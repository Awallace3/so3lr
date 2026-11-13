"""Pack dimers into padded SO3krates graph batches.

Two layouts share one function:

* ``three_pass=True`` (Paths A/B/C): every dimer contributes three graphs, in the
  order AB, A, B. Each graph is an ordinary SO3LR structure with its own total
  charge, so the released charge embedding and charge conservation apply
  unchanged to every pass.
* ``three_pass=False`` (Path E): one AB graph per dimer.

Every graph carries ``residue_segments`` / ``residue_charge`` (A = residue 0,
B = residue 1; a monomer graph is one residue), so partial charges are conserved
per monomer in the AB pass too. With ``per_residue_embed=True`` the charge and
spin embeddings also read the monomer's own charge / unpaired count
(``residue_charge_embed`` / ``residue_unpaired_embed``), so an AB graph at
infinite separation embeds exactly like its two monomer graphs.

Both layouts carry the A-B pair list of the AB graph (``pair_i`` in A, ``pair_j``
in B, one direction) and the per-node bookkeeping the heads need:

* ``node_dimer``  -- dimer index of every node (padding: ``num_dimers``)
* ``node_sign``   -- +1 on AB-graph nodes, -1 on monomer-pass nodes, 0 on padding,
  so ``segment_sum(node_sign * eps, node_dimer)`` is the AB - A - B difference
* ``mon_node``    -- for an AB-graph node, the node of the same atom in its
  monomer pass (three-pass only; identity elsewhere)
* ``lr_on_ab``    -- 1 for long-range edges inside an AB graph

Training and evaluation pad every batch to one fixed set of ``caps`` (nodes,
edges, long-range edges, A-B pairs, dimers), so ``jit`` compiles once per
layout. ``pack`` fills a batch with dimers in the given order until the next one
would overflow any cap; ``estimate_caps`` sizes the caps from the store so an
average batch holds ``batch_dimers`` dimers. Without ``caps`` (tests, ad hoc
calls) shapes are rounded up to coarse buckets instead. One padding graph owns
all padding nodes, padding edges are self-loops on the last node (``d = 0``,
which every SO3LR module masks), and padded dimer rows have ``dimer_mask = 0``.
"""

from __future__ import annotations

import math

import numpy as np

THEORY_LEVEL = 5          # jnp.eye(16)[5] -- the level the released T-M uses
NUM_THEORY_LEVELS = 16


def _bucket(n, quantum):
    """Round ``n + 1`` up to a multiple of ``quantum`` (the +1 reserves padding)."""
    return int(math.ceil((n + 1) / quantum) * quantum)


def _radius_edges(R, cutoff):
    d = np.sqrt(((R[:, None, :] - R[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    i, j = np.nonzero(d < cutoff)
    return i, j


def _all_pairs(n):
    i, j = np.nonzero(~np.eye(n, dtype=bool))
    return i, j


def _unpaired(z, charge):
    return int((int(z.sum()) - int(charge)) % 2)


def dimer_cost(store, row, three_pass, cutoff=4.5):
    """(nodes, edges, lr, pairs) one dimer adds to a batch."""
    z, R, (na, nb), _, _ = store.dimer(int(row))
    n = na + nb
    e = len(_radius_edges(R, cutoff)[0])
    lr = n * (n - 1)
    nodes = n
    if three_pass:
        e += len(_radius_edges(R[:na], cutoff)[0]) + len(_radius_edges(R[na:], cutoff)[0])
        lr += na * (na - 1) + nb * (nb - 1)
        nodes = 2 * n
    return np.array([nodes, e, lr, na * nb], np.int64)


CAP_KEYS = ("nodes", "edges", "lr", "pairs")


def estimate_caps(store, three_pass, batch_dimers, sample=2000, seed=0, cutoff=4.5):
    """Caps for ``batch_dimers`` average-sized dimers, never below the largest dimer.

    Means come from a random sample; the per-dimer maxima of nodes, lr and pairs are
    exact (from ``natoms``) and the edge maximum is taken over the sample plus the
    500 largest dimers, with 25% headroom.
    """
    rng = np.random.default_rng(seed)
    n = len(store)
    ids = rng.choice(n, size=min(sample, n), replace=False)
    nat = np.asarray(store.natoms, np.int64)
    big = np.argsort(nat.sum(1))[-500:]
    costs = np.array([dimer_cost(store, i, three_pass, cutoff) for i in np.r_[ids, big]])
    mean = costs[:len(ids)].mean(0)
    na, nb = nat[:, 0], nat[:, 1]
    tot = na + nb
    exact_max = np.array([
        (2 if three_pass else 1) * tot.max(),
        0,
        (tot * (tot - 1) + (na * (na - 1) + nb * (nb - 1) if three_pass else 0)).max(),
        (na * nb).max(),
    ])
    single_max = np.maximum(exact_max, costs.max(0))
    single_max[1] = int(costs[:, 1].max() * 1.25)
    caps = np.maximum(np.ceil(mean * batch_dimers), single_max) + 1   # +1: padding slot
    quantum = np.array([64, 1024, 4096, 512])
    caps = (np.ceil(caps / quantum) * quantum).astype(int)
    out = dict(zip(CAP_KEYS, caps.tolist()))
    out["dimers"] = int(2 * batch_dimers)
    out["mean_cost"] = dict(zip(CAP_KEYS, mean.round(1).tolist()))
    return out


def pack(store, order, caps, three_pass, cutoff=4.5, skipped=None):
    """Yield id arrays from ``order`` whose summed cost fits ``caps``.

    A dimer that cannot fit even alone is skipped and appended to ``skipped``
    (a list the caller reports); with caps from ``estimate_caps`` this should
    never happen.
    """
    limit = np.array([caps[k] - 1 for k in CAP_KEYS])
    cur, used = [], np.zeros(4, np.int64)
    for row in order:
        c = dimer_cost(store, row, three_pass, cutoff)
        if (c > limit).any():
            if skipped is not None:
                skipped.append(int(row))
            continue
        if cur and ((used + c > limit).any() or len(cur) >= caps["dimers"]):
            yield np.asarray(cur)
            cur, used = [], np.zeros(4, np.int64)
        cur.append(int(row))
        used += c
    if cur:
        yield np.asarray(cur)


def make_batch(store, ids, *, three_pass, cutoff=4.5, quanta=None,
               per_residue_embed=True, caps=None):
    """Return a dict of numpy arrays for the dimers ``ids`` of ``store``.

    With ``caps`` every array is padded to exactly those sizes (ValueError if the
    dimers do not fit); otherwise sizes are rounded up to ``quanta`` buckets.
    """
    q = dict(nodes=128, edges=2048, lr=8192, pairs=1024)
    if quanta:
        q.update(quanta)
    pos, Z, bseg, rseg, ndim, nsign, monn = [], [], [], [], [], [], []
    tot_q, unp, res_q, res_u = [], [], [], []
    ei, ej, li, lj, lab = [], [], [], [], []
    pi, pj, pseg = [], [], []
    targets, natoms, charges = [], [], []
    n_node = 0
    g = 0

    def add_graph(z, R, residues, charge, residue_charge, residue_unp, sign, d):
        nonlocal n_node, g
        n = len(z)
        base = n_node
        pos.append(R)
        Z.append(z)
        bseg.append(np.full(n, g))
        rseg.append(residues)
        ndim.append(np.full(n, d))
        nsign.append(np.full(n, sign, dtype=np.float32))
        monn.append(np.arange(base, base + n))   # identity; fixed up below
        tot_q.append(charge)
        unp.append(_unpaired(z, charge))
        res_q.append(residue_charge)
        res_u.append(residue_unp)
        i, j = _radius_edges(R, cutoff)
        ei.append(i + base)
        ej.append(j + base)
        i, j = _all_pairs(n)
        li.append(i + base)
        lj.append(j + base)
        lab.append(np.full(len(i), sign > 0, dtype=np.float32))
        n_node += n
        g += 1
        return base

    for d, row in enumerate(ids):
        z, R, (na, nb), (qa, qb), t = store.dimer(int(row))
        residues = np.r_[np.zeros(na, np.int64), np.ones(nb, np.int64)]
        ua, ub = _unpaired(z[:na], qa), _unpaired(z[na:], qb)
        ab = add_graph(z, R, residues, qa + qb, (qa, qb), (ua, ub), 1.0, d)
        a_idx = ab + np.arange(na)
        b_idx = ab + na + np.arange(nb)
        ii, jj = np.meshgrid(a_idx, b_idx, indexing="ij")
        pi.append(ii.ravel())
        pj.append(jj.ravel())
        pseg.append(np.full(ii.size, d))
        if three_pass:
            a0 = add_graph(z[:na], R[:na], np.zeros(na, np.int64), qa, (qa, 0), (ua, 0), -1.0, d)
            b0 = add_graph(z[na:], R[na:], np.zeros(nb, np.int64), qb, (qb, 0), (ub, 0), -1.0, d)
            monn[-3][:na] = a0 + np.arange(na)
            monn[-3][na:] = b0 + np.arange(nb)
        targets.append(t)
        natoms.append((na, nb))
        charges.append((qa, qb))

    n_real = len(ids)
    num_dimers = caps["dimers"] if caps else n_real
    if n_real > num_dimers:
        raise ValueError(f"{n_real} dimers > cap {num_dimers}")

    def size(raw, key):
        if caps is None:
            return _bucket(raw, q[key])
        if raw + 1 > caps[key]:
            raise ValueError(f"{key}: {raw} + padding > cap {caps[key]}")
        return caps[key]

    N = size(n_node, "nodes")
    G = (3 * num_dimers + 1) if (caps and three_pass) else ((num_dimers + 1) if caps else g + 1)
    pad_node = N - 1

    def cat(parts, fill, size, dtype):
        a = np.concatenate(parts) if parts else np.zeros(0, dtype)
        out = np.full((size,) + a.shape[1:], fill, dtype=dtype)
        out[:len(a)] = a
        return out, len(a)

    positions, _ = cat(pos, 0.0, N, np.float32)
    # padding atoms sit far apart so nothing but self-loops ever touches them
    positions[n_node:, 0] = 1.0e4 + 10.0 * np.arange(N - n_node)
    atomic_numbers, _ = cat(Z, 1, N, np.int32)
    batch_segments, _ = cat(bseg, g, N, np.int32)
    residue_segments, _ = cat(rseg, 0, N, np.int32)
    node_dimer, _ = cat(ndim, num_dimers, N, np.int32)
    node_sign, _ = cat(nsign, 0.0, N, np.float32)
    mon_node, _ = cat(monn, pad_node, N, np.int32)
    mon_node[n_node:] = np.arange(n_node, N)
    node_mask = np.zeros(N, bool)
    node_mask[:n_node] = True

    e_raw = sum(len(a) for a in ei)
    idx_i, _ = cat(ei, pad_node, size(e_raw, "edges"), np.int32)
    idx_j, _ = cat(ej, pad_node, len(idx_i), np.int32)
    l_raw = sum(len(a) for a in li)
    idx_i_lr, _ = cat(li, pad_node, size(l_raw, "lr"), np.int32)
    idx_j_lr, _ = cat(lj, pad_node, len(idx_i_lr), np.int32)
    lr_on_ab, _ = cat(lab, 0.0, len(idx_i_lr), np.float32)
    p_raw = sum(len(a) for a in pi)
    pair_i, _ = cat(pi, pad_node, size(p_raw, "pairs"), np.int32)
    pair_j, _ = cat(pj, pad_node, len(pair_i), np.int32)
    pair_dimer, _ = cat(pseg, num_dimers, len(pair_i), np.int32)
    pair_mask = np.zeros(len(pair_i), np.float32)
    pair_mask[:p_raw] = 1.0

    graph_mask = np.zeros(G, bool)
    graph_mask[:g] = True
    total_charge = np.zeros(G, np.float32)
    total_charge[:g] = tot_q
    num_unpaired = np.zeros(G, np.float32)
    num_unpaired[:g] = unp
    residue_charge = np.zeros((G, 2), np.float32)
    residue_charge[:g] = res_q
    residue_unpaired = np.zeros((G, 2), np.float32)
    residue_unpaired[:g] = res_u
    theory_mask = np.zeros((G, NUM_THEORY_LEVELS), np.float32)
    theory_mask[:, THEORY_LEVEL] = 1.0

    out = dict(
        positions=positions, atomic_numbers=atomic_numbers,
        idx_i=idx_i, idx_j=idx_j, idx_i_lr=idx_i_lr, idx_j_lr=idx_j_lr,
        batch_segments=batch_segments, node_mask=node_mask, graph_mask=graph_mask,
        total_charge=total_charge, num_unpaired_electrons=num_unpaired,
        theory_mask=theory_mask,
        node_dimer=node_dimer, node_sign=node_sign, mon_node=mon_node,
        lr_on_ab=lr_on_ab, pair_i=pair_i, pair_j=pair_j, pair_dimer=pair_dimer,
        pair_mask=pair_mask,
        targets=_pad_rows(np.asarray(targets, np.float32).reshape(n_real, 4), num_dimers),
        dimer_natoms=_pad_rows(np.asarray(natoms, np.int32).reshape(n_real, 2), num_dimers),
        dimer_charges=_pad_rows(np.asarray(charges, np.float32).reshape(n_real, 2), num_dimers),
        dimer_mask=_pad_rows(np.ones(n_real, np.float32), num_dimers),
        residue_segments=residue_segments, residue_charge=residue_charge,
    )
    if per_residue_embed:
        out["residue_charge_embed"] = residue_charge
        out["residue_unpaired_embed"] = residue_unpaired
    return out


def _pad_rows(a, n):
    out = np.zeros((n,) + a.shape[1:], a.dtype)
    out[:len(a)] = a
    return out


def backbone_inputs(batch):
    """The subset of ``batch`` the released StackNet reads."""
    keys = ("positions", "atomic_numbers", "idx_i", "idx_j", "idx_i_lr", "idx_j_lr",
            "batch_segments", "node_mask", "graph_mask", "total_charge",
            "num_unpaired_electrons", "theory_mask", "residue_segments",
            "residue_charge", "residue_charge_embed", "residue_unpaired_embed")
    return {k: batch[k] for k in keys if k in batch}
