"""Train SO3LR-T-M SAPT heads on a mastiff-classical store.

    python -m so3lr.sapt.train --arm A0 --store-root <root> --out <run_dir> ...

Resumable: ``<out>/ckpt/last.pkl`` holds parameters, optimizer state, epoch and
the position inside the epoch's permutation; a TERM signal or ``--deadline-s``
saves it and exits 0 with ``<out>/STOPPED_EARLY``. A finished run writes
``<out>/results.json`` and ``<out>/test_predictions.npz`` and exits 0. The
selected checkpoint (``ckpt/best.pkl``) is the epoch with the lowest validation
MAE of the summed component channels (``sel_metric``); MSE is not used for selection.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import queue
import signal
import threading
import time
from pathlib import Path

import numpy as np

ARMS = {
    # id: path, backbone training, pretrained, backbone lr scale, charge anchor
    "A0": dict(path="A", train_backbone="none", pretrained=True),
    "A1": dict(path="A", train_backbone="last", pretrained=True),
    "A2": dict(path="A", train_backbone="all", pretrained=True, backbone_lr_scale=0.1, charge_anchor=100.0),
    "B1": dict(path="B", train_backbone="last", pretrained=True),
    "C1": dict(path="C", train_backbone="all", pretrained=False),
    "E0": dict(path="E", train_backbone="none", pretrained=True),
    "E1": dict(path="E", train_backbone="last", pretrained=True),
    "E2": dict(path="E", train_backbone="all", pretrained=True, backbone_lr_scale=0.1, charge_anchor=100.0),
    "E3": dict(path="E", train_backbone="all", pretrained=False),
}

STOP = {"flag": False}


def _on_term(signum, frame):
    STOP["flag"] = True


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--store-root", required=True, help="classical store directory holding train/val/test")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--test-split", default="test")
    p.add_argument("--manifest-sha256", default=None,
                   help="JSON {split: sha256} pinning each split manifest")
    p.add_argument("--no-verify", action="store_true", help="skip array sha256 checks (size still checked)")
    p.add_argument("--model-dir", required=True, help="released model directory (hyperparameters.json, params.pkl)")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-dimers", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--w-total", type=float, default=1.0, help="weight of the summed-total MSE term")
    p.add_argument("--charge-anchor", type=float, default=None, help="override the arm's lambda_q")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-dimers", type=int, default=None, help="smoke: use the first N training rows")
    p.add_argument("--max-eval-dimers", type=int, default=None, help="smoke: cap val/test rows")
    p.add_argument("--max-steps-per-epoch", type=int, default=None)
    p.add_argument("--deadline-s", type=float, default=None, help="save and stop after this many seconds")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--save-every-s", type=float, default=900.0,
                   help="also write ckpt/last.pkl mid-epoch this often (preemption)")
    p.add_argument("--no-per-residue-embed", action="store_true")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-id", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-tags", default="", help="comma separated")
    p.add_argument("--wandb-notes", default=None)
    p.add_argument("--wandb-config-json", default=None, help="extra config merged into wandb.config")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- data
class Prefetch:
    """Build numpy batches on a background thread."""

    def __init__(self, store, order, batch_dimers, three_pass, per_residue_embed, depth=6):
        self.q = queue.Queue(maxsize=depth)
        self.n = int(np.ceil(len(order) / batch_dimers))
        chunks = [order[k * batch_dimers:(k + 1) * batch_dimers] for k in range(self.n)]
        from so3lr.sapt.batch import make_batch

        def work():
            try:
                for c in chunks:
                    self.q.put((c, make_batch(store, c, three_pass=three_pass,
                                              per_residue_embed=per_residue_embed)))
            except Exception as exc:          # surface in the consumer
                self.q.put(exc)
            self.q.put(None)

        threading.Thread(target=work, daemon=True).start()

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item


def _atomic_pickle(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# -------------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    arm = dict(ARMS[args.arm])
    path = arm["path"]
    lam_q = args.charge_anchor if args.charge_anchor is not None else arm.get("charge_anchor", 0.0)
    bb_scale = arm.get("backbone_lr_scale", 1.0)

    import jax
    import jax.numpy as jnp
    import optax
    import flax

    from so3lr.sapt.store import DimerStore
    from so3lr.sapt.batch import make_batch
    from so3lr.sapt.model import (SAPTModel, OUTPUTS, THREE_PASS, load_backbone,
                                  merge_pretrained, reference_components)

    out = Path(args.out)
    if (out / "results.json").exists():
        print(f"{out}/results.json exists; nothing to do", flush=True)
        return 0
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _on_term)
    t_start = time.time()

    pins = json.loads(args.manifest_sha256) if args.manifest_sha256 else {}
    verify = not args.no_verify
    stores = {s: DimerStore(args.store_root, s, verify=verify, expect_manifest_sha256=pins.get(s))
              for s in {args.train_split, args.val_split, args.test_split}}
    train, val, test = stores[args.train_split], stores[args.val_split], stores[args.test_split]

    three = THREE_PASS[path]
    pre = not args.no_per_residue_embed
    outputs = OUTPUTS[path]
    nout = len(outputs)

    net, bb_params, model_prov = load_backbone(args.model_dir, pretrained=arm["pretrained"])
    model = SAPTModel(backbone=net, path=path)
    example = make_batch(val, np.arange(min(4, len(val))), three_pass=three, per_residue_embed=pre)
    params = model.init(jax.random.PRNGKey(args.seed), example)
    extra = []
    if bb_params is not None:
        params, extra = merge_pretrained(params, bb_params)
    params = flax.core.unfreeze(params)
    ref_backbone = jax.tree_util.tree_map(jnp.array, params["params"]["backbone"]) if lam_q > 0 else None

    # --- parameter partition
    n_layers = len([k for k in params["params"]["backbone"] if k.startswith("layers_")])
    last_layer = f"layers_{n_layers - 1}"

    def label(path_tuple):
        top = path_tuple[1]
        if top == "heads" or top != "backbone":
            return "heads"
        if arm["train_backbone"] == "all":
            return "backbone"
        if arm["train_backbone"] == "last" and path_tuple[2] == last_layer:
            return "backbone"
        return "frozen"

    labels = flax.traverse_util.path_aware_map(lambda pth, _: label(("params",) + pth), params["params"])
    labels = {"params": labels}
    trainable_mask = jax.tree_util.tree_map(lambda l: l != "frozen", labels)

    n_tr = len(train) if args.max_train_dimers is None else min(len(train), args.max_train_dimers)
    steps_per_epoch = int(np.ceil(n_tr / args.batch_dimers))
    if args.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    total_steps = max(1, steps_per_epoch * args.epochs)

    def sched(scale):
        return optax.warmup_cosine_decay_schedule(
            0.0, args.lr * scale, min(args.warmup_steps, total_steps // 10 + 1),
            total_steps, args.lr * scale * args.min_lr_ratio)

    tx = optax.chain(
        optax.clip_by_global_norm(args.clip),
        optax.multi_transform({
            "heads": optax.adamw(sched(1.0), weight_decay=args.weight_decay),
            "backbone": optax.adamw(sched(bb_scale), weight_decay=args.weight_decay),
            "frozen": optax.set_to_zero(),
        }, labels),
    )
    opt_state = tx.init(params)

    def freeze(p):
        return jax.tree_util.tree_map(lambda v, m: v if m else jax.lax.stop_gradient(v), p, trainable_mask)

    def loss_fn(p, batch):
        pred, aux = model.apply(freeze(p), batch)
        ref = reference_components(batch["targets"], path)
        err = pred - ref
        comp = jnp.mean(err ** 2, axis=0)
        tot = jnp.mean((pred.sum(-1) - batch["targets"].sum(-1)) ** 2)
        loss = comp.sum() + args.w_total * tot
        anchor = jnp.zeros(())
        if lam_q > 0:
            ref_p = dict(p)
            ref_p = {"params": dict(p["params"], backbone=ref_backbone)}
            _, aux_ref = model.apply(jax.lax.stop_gradient(ref_p), batch)
            m = batch["node_mask"].astype(pred.dtype)
            anchor = jnp.sum(m * (aux["partial_charges"] - aux_ref["partial_charges"]) ** 2) / jnp.sum(m)
            loss = loss + lam_q * anchor
        return loss, dict(loss=loss, mse=comp, mse_total=tot, anchor=anchor)

    @jax.jit
    def train_step(p, o, batch):
        (_, met), g = jax.value_and_grad(loss_fn, has_aux=True)(p, batch)
        upd, o = tx.update(g, o, p)
        p = optax.apply_updates(p, upd)
        met["gnorm"] = optax.global_norm(g)
        return p, o, met

    @jax.jit
    def predict(p, batch):
        return model.apply(p, batch)[0]

    def evaluate(p, store, cap=None, keep=False):
        n = len(store) if cap is None else min(cap, len(store))
        preds, refs, rows = [], [], []
        for ids, b in Prefetch(store, np.arange(n), args.batch_dimers, three, pre):
            preds.append(np.asarray(predict(p, b)))
            refs.append(b["targets"])
            rows.append(store.pickle_row[ids])
        pred = np.concatenate(preds)
        ref4 = np.concatenate(refs)
        refc = np.asarray(reference_components(ref4, path))
        res = {f"mae_{c}": float(np.mean(np.abs(pred[:, k] - refc[:, k]))) for k, c in enumerate(outputs)}
        res["mae_total"] = float(np.mean(np.abs(pred.sum(-1) - ref4.sum(-1))))
        if path in ("A", "C"):
            res["mae_elst+ind"] = float(np.mean(np.abs(pred[:, 0] + pred[:, 2] - ref4[:, 0] - ref4[:, 2])))
        res["sel_metric"] = float(sum(res[f"mae_{c}"] for c in outputs))
        res["n"] = int(n)
        if keep:
            return res, dict(pred=pred, ref=ref4, pickle_row=np.concatenate(rows), outputs=np.array(outputs))
        return res

    # --- resume
    state = dict(epoch=0, pos=0, step=0, best=None, history=[])
    last = out / "ckpt" / "last.pkl"
    if last.exists():
        with open(last, "rb") as fh:
            ck = pickle.load(fh)
        if ck["arm"] != args.arm or ck["seed"] != args.seed:
            raise SystemExit(f"{last} belongs to arm {ck['arm']} seed {ck['seed']}")
        params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
        opt_state = jax.tree_util.tree_map(jnp.asarray, ck["opt_state"])
        state = ck["state"]
        print(f"resumed epoch {state['epoch']} pos {state['pos']} step {state['step']}", flush=True)

    def save_last():
        _atomic_pickle(dict(arm=args.arm, seed=args.seed, params=jax.device_get(params),
                            opt_state=jax.device_get(opt_state), state=state), last)

    config = dict(arm=args.arm, path=path, arm_spec=arm, charge_anchor=lam_q, args=vars(args),
                  outputs=list(outputs), model=model_prov, new_backbone_params=extra,
                  data={s: st.provenance() for s, st in stores.items()},
                  steps_per_epoch=steps_per_epoch, total_steps=total_steps,
                  jax=dict(version=jax.__version__, devices=[str(d) for d in jax.devices()]))
    (out / "config.json").write_text(json.dumps(config, indent=1, default=str))

    wb = None
    if args.wandb_project:
        import wandb
        wcfg = dict(config)
        if args.wandb_config_json:
            wcfg.update(json.loads(Path(args.wandb_config_json).read_text()))
        wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity, id=args.wandb_id,
                        name=args.wandb_name, resume="allow", config=wcfg, notes=args.wandb_notes,
                        tags=[t for t in args.wandb_tags.split(",") if t], dir=str(out))

    mlog = open(out / "metrics.jsonl", "a")

    def log(rec):
        rec["time"] = time.time()
        mlog.write(json.dumps(rec) + "\n")
        mlog.flush()
        if wb is not None:
            wb.log({k: v for k, v in rec.items() if isinstance(v, (int, float))}, step=state["step"])

    print(f"arm {args.arm} path {path} outputs {outputs} steps/epoch {steps_per_epoch} "
          f"train rows {n_tr} devices {jax.devices()}", flush=True)
    stopped = False
    while state["epoch"] < args.epochs and not stopped:
        ep = state["epoch"]
        order = np.random.default_rng(args.seed * 100003 + ep).permutation(n_tr)
        order = order[: steps_per_epoch * args.batch_dimers]
        acc, t_ep = [], time.time()
        t_saved = time.time()
        for ids, b in Prefetch(train, order[state["pos"] * args.batch_dimers:], args.batch_dimers, three, pre):
            params, opt_state, met = train_step(params, opt_state, b)
            state["pos"] += 1
            state["step"] += 1
            acc.append(jax.device_get(met))
            if state["step"] % args.log_every == 0:
                m = {k: np.mean([a[k] for a in acc[-args.log_every:]], axis=0) for k in acc[-1]}
                rec = dict(kind="train", epoch=ep, step=state["step"], loss=float(m["loss"]),
                           mse_total=float(m["mse_total"]), gnorm=float(m["gnorm"]),
                           anchor=float(m["anchor"]),
                           **{f"mse_{c}": float(m["mse"][k]) for k, c in enumerate(outputs)})
                log(rec)
                print(json.dumps(rec), flush=True)
            if not np.isfinite(acc[-1]["loss"]):
                raise FloatingPointError(f"non-finite loss at step {state['step']}")
            if time.time() - t_saved > args.save_every_s:
                save_last()
                t_saved = time.time()
            over = args.deadline_s is not None and time.time() - t_start > args.deadline_s
            if STOP["flag"] or over:
                stopped = True
                break
        if stopped:
            save_last()
            (out / "STOPPED_EARLY").write_text(f"epoch {ep} pos {state['pos']} step {state['step']}\n")
            print(f"stopping early at epoch {ep} pos {state['pos']}", flush=True)
            break
        vres = evaluate(params, val, args.max_eval_dimers)
        tr_loss = float(np.mean([a["loss"] for a in acc])) if acc else float("nan")
        rec = dict(kind="val", epoch=ep, step=state["step"], train_loss=tr_loss,
                   epoch_s=time.time() - t_ep, **{f"val_{k}": v for k, v in vres.items()})
        improved = state["best"] is None or vres["sel_metric"] < state["best"]["sel_metric"]
        if improved:
            state["best"] = dict(epoch=ep, **vres)
            _atomic_pickle(dict(arm=args.arm, seed=args.seed, epoch=ep, params=jax.device_get(params),
                                val=vres), out / "ckpt" / "best.pkl")
        rec["improved"] = bool(improved)
        state["history"].append(rec)
        log(rec)
        print(json.dumps(rec), flush=True)
        state["epoch"] = ep + 1
        state["pos"] = 0
        save_last()

    if stopped:
        mlog.close()
        if wb is not None:
            wb.finish(exit_code=0)
        return 0

    results = dict(arm=args.arm, seed=args.seed, best=state["best"], history=state["history"])
    if not args.skip_test:
        with open(out / "ckpt" / "best.pkl", "rb") as fh:
            best = pickle.load(fh)
        bp = jax.tree_util.tree_map(jnp.asarray, best["params"])
        tres, rows = evaluate(bp, test, args.max_eval_dimers, keep=True)
        results["test"] = dict(epoch=best["epoch"], **tres)
        np.savez_compressed(out / "test_predictions.npz", **rows)
        log(dict(kind="test", epoch=best["epoch"], **{f"test_{k}": v for k, v in tres.items()}))
        print(json.dumps(results["test"]), flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    mlog.close()
    if wb is not None:
        if "test" in results:
            wb.summary.update({f"test/{k}": v for k, v in results["test"].items()})
        wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
