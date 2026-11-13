"""SO3LR-T-M with SAPT component heads.

``SAPTModel`` nests the released StackNet under ``params['backbone']`` exactly as
it ships, so a pretrained checkpoint loads without renaming and "frozen" means
"the optimizer sees zero updates for that subtree". Every other top-level
entry of ``params['params']`` is a new head parameter.

Paths (proposal-0-2026-11-13, sections 4-8):

* ``A`` -- three passes (AB, A, B). elst = A-B Coulomb of the monomer-pass
  charges; ind = E_C[q(AB)] - E_C[q(A) (+) q(B)]; exch = frozen NLH + learned
  NLH-form screened Coulomb over A-B pairs; disp = QDO over A-B pairs from the
  dimer-pass alpha / C6 ratios with learned gamma and t6/t8/t10; plus a
  four-channel per-atom residual summed as AB - A - B.
* ``B`` -- A with elst and ind summed into one output and a three-channel residual.
* ``C`` -- A's map on a randomly initialised backbone (the caller skips the
  pretrained load; nothing in this module differs).
* ``E`` -- one dimer pass; every energy is an A-B pair sum. elst+ind = Coulomb
  of the per-monomer-conserved dimer charges + pair residual; exch = frozen NLH +
  softplus Born-Mayer-Slater pair; disp = QDO over A-B pairs + attractive pair
  residual.

All component outputs are kcal/mol per dimer. Distances are Angstrom.
"""

from __future__ import annotations

from typing import Any, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ase.units import Bohr, Hartree
from jax.ops import segment_sum

from so3lr.mlff.nn.observable.dispersion_ref_data import NLH_AA, NLH_BB
from so3lr.mlff.nn.observable.observable_sparse import (
    gamma_cubic_fit,
    mixing_rules,
    switching_fn,
)
from so3lr.mlff.sph_ops import init_sph_fn

EV_TO_KCAL = 23.060547830619
KE_KCAL = 332.0637133         # kcal/mol * Angstrom / e^2
ZMAX = 118

# Output channels per path (order of the component vector the loss sees).
OUTPUTS = {
    "A": ("elst", "exch", "ind", "disp"),
    "C": ("elst", "exch", "ind", "disp"),
    "B": ("elst+ind", "exch", "disp"),
    "E": ("elst+ind", "exch", "disp"),
}
THREE_PASS = {"A": True, "B": True, "C": True, "E": False}

BACKBONE_QUANTITIES = ("x", "ev", "partial_charges")


def _isp(y):
    """Inverse softplus."""
    return float(np.log(np.expm1(y)))


def _erf_coulomb(qq, r, sigma):
    return KE_KCAL * qq * jax.scipy.special.erf(r / sigma) / r


def _nlh_pair(z_i, z_j, r):
    """NLH screened nuclear repulsion for one direction of a pair (kcal/mol).

    Same tables and switch as NLHRepulsionSparse; the released module counts each
    undirected edge twice with a 1/2, here each A-B pair appears once.
    """
    zi = jnp.minimum(z_i, z_j) - 1
    zj = jnp.maximum(z_i, z_j) - 1
    a = jnp.asarray(NLH_AA, r.dtype)[zi, zj]
    b = jnp.asarray(NLH_BB, r.dtype)[zi, zj]
    y = jnp.sum(a * jnp.exp(-b * r[:, None]), axis=-1)
    w = switching_fn(r, x_on=0.0, x_off=1.5)
    return w * KE_KCAL * z_i * z_j / r * y


def _cos_cut(r, rc):
    return jnp.where(r < rc, 0.5 * (jnp.cos(jnp.pi * r / rc) + 1.0), 0.0)


class MLP(nn.Module):
    widths: Sequence[int]
    out: int
    zero_last: bool = True
    bias_init: Any = None

    @nn.compact
    def __call__(self, y):
        for w in self.widths:
            y = nn.silu(nn.Dense(w)(y))
        init = nn.initializers.zeros_init() if self.zero_last else nn.initializers.lecun_normal()
        binit = (nn.initializers.zeros_init() if self.bias_init is None
                 else (lambda key, shape, dtype=jnp.float32: jnp.asarray(self.bias_init, dtype) * jnp.ones(shape, dtype)))
        return nn.Dense(self.out, kernel_init=init, bias_init=binit)(y)


class ElementTable(nn.Module):
    """Per-element learned values (ZMAX+1, n) initialised to a constant row."""
    init_row: Sequence[float]

    @nn.compact
    def __call__(self, z):
        row = jnp.asarray(self.init_row, jnp.float32)
        table = self.param("table", lambda key, shape: jnp.tile(row, (ZMAX + 1, 1)),
                           (ZMAX + 1, len(self.init_row)))
        return table[z]


class QDOPairDispersion(nn.Module):
    """-sum_n t_2n C_2n / (R^2n + (gamma * 5.08 alpha^1/7)^2n) over given pairs (kcal/mol)."""

    @nn.compact
    def __call__(self, z, a0_ratios, c6_ratios, i, j, r, node_mask):
        # padding atoms can carry zero/garbage ratios; 0 * NaN in a masked sum still
        # poisons the gradient, so give them the free-atom values before mixing
        a0_ratios = jnp.where(node_mask, a0_ratios, 1.0)
        c6_ratios = jnp.where(node_mask, c6_ratios, 1.0)
        t = nn.softplus(self.param("t_raw", lambda k, s: jnp.full(s, _isp(1.0)), (3,)))
        g = nn.softplus(self.param("gamma_raw", lambda k, s: jnp.full(s, _isp(1.0)), ()))
        alpha_ij, c6 = mixing_rules(z, i, j, a0_ratios, c6_ratios)
        gam = gamma_cubic_fit(alpha_ij)
        c8 = 5.0 / gam * c6
        c10 = 245.0 / 8.0 / gam ** 2 * c6
        p = g * 2.0 * 2.54 * alpha_ij ** (1.0 / 7.0)
        R = r / Bohr
        v = (t[0] * c6 / (R ** 6 + p ** 6) + t[1] * c8 / (R ** 8 + p ** 8)
             + t[2] * c10 / (R ** 10 + p ** 10))
        return -v * Hartree * EV_TO_KCAL


class SAPTModel(nn.Module):
    backbone: nn.Module
    path: str = "A"
    readout_widths: Sequence[int] = (128, 64)
    pair_widths: Sequence[int] = (128, 64)
    sigma_init: float = 1.0          # per-atom erf width s_i (A); sigma_ij = sqrt(s_i^2 + s_j^2)
    exch_b_init: Sequence[float] = (2.0, 2.75, 3.5)   # Path A NLH-form exponents (1/A)
    exch_a_init: float = 0.05
    exch_z_init: float = 4.0         # Path A effective charge Z~_i at init
    bms_a_init: float = 10.0         # Path E Born-Mayer-Slater prefactor (kcal/mol)
    bms_b_init: float = 2.2          # Path E exponent (1/A)
    pair_cutoff: float = 5.0         # Path E f_cut radius for the pair residuals (A)
    num_rbf: int = 16

    @property
    def outputs(self):
        return OUTPUTS[self.path]

    @nn.compact
    def __call__(self, batch):
        if self.path not in OUTPUTS:
            raise ValueError(f"unknown path {self.path!r}")
        bb_keys = ("positions", "atomic_numbers", "idx_i", "idx_j", "idx_i_lr", "idx_j_lr",
                   "batch_segments", "node_mask", "graph_mask", "total_charge",
                   "num_unpaired_electrons", "theory_mask", "residue_segments",
                   "residue_charge", "residue_charge_embed", "residue_unpaired_embed")
        bb = self.backbone({k: batch[k] for k in bb_keys if k in batch})
        x = bb["x"]
        q = bb["partial_charges"]
        a0r = bb["a0_ratios"]
        c6r = bb["c6_ratios"]
        z = batch["atomic_numbers"]
        R = batch["positions"]
        num_dimers = batch["targets"].shape[0]
        nseg = num_dimers + 1                      # last segment collects padding

        pi, pj, pm = batch["pair_i"], batch["pair_j"], batch["pair_mask"]
        # padding pairs are self-loops; shift them to 10 A so norms and their
        # gradients stay finite (they are masked out of every sum)
        shift = jnp.array([10.0, 0.0, 0.0], R.dtype)
        rvec = R[pj] - R[pi] + (1.0 - pm)[:, None] * shift
        rp = jnp.linalg.norm(rvec, axis=-1)
        pseg = batch["pair_dimer"]

        def pair_sum(e):
            return segment_sum(e * pm, pseg, nseg)[:num_dimers]

        if THREE_PASS[self.path]:
            comps, aux = self._three_pass(batch, x, q, a0r, c6r, z, R, rp, pair_sum, nseg, num_dimers)
        else:
            comps, aux = self._dimer_pass(batch, bb, x, q, a0r, c6r, z, rvec, rp, pair_sum)
        aux["partial_charges"] = q
        return jnp.stack(comps, axis=-1), aux

    # ------------------------------------------------------------------ A/B/C
    def _three_pass(self, batch, x, q, a0r, c6r, z, R, rp, pair_sum, nseg, num_dimers):
        pi, pj = batch["pair_i"], batch["pair_j"]
        mon = batch["mon_node"]
        x_m = x[mon]                               # permanent (monomer-pass) features on AB nodes
        q_m = q[mon]

        s = nn.softplus(ElementTable((_isp(self.sigma_init),), name="sigma_elem")(z)[:, 0]
                        + MLP(self.readout_widths, 1, name="sigma_mlp")(x_m)[:, 0])
        sig_p = jnp.sqrt(s[pi] ** 2 + s[pj] ** 2)
        elst = pair_sum(_erf_coulomb(q_m[pi] * q_m[pj], rp, sig_p))

        # induction: charge relaxation over every pair of the AB graph
        li, lj, lab = batch["idx_i_lr"], batch["idx_j_lr"], batch["lr_on_ab"]
        shift = jnp.array([10.0, 0.0, 0.0], R.dtype)
        rl = jnp.linalg.norm(R[lj] - R[li] + (1.0 - lab)[:, None] * shift, axis=-1)
        sig_l = jnp.sqrt(s[li] ** 2 + s[lj] ** 2)
        dqq = q[li] * q[lj] - q_m[li] * q_m[lj]
        e_ind = 0.5 * lab * _erf_coulomb(dqq, rl, sig_l)
        ind = segment_sum(e_ind, batch["node_dimer"][li], nseg)[:num_dimers]

        # exchange: frozen NLH + learned NLH-form screened Coulomb, A-B pairs
        zt = nn.softplus(ElementTable((_isp(self.exch_z_init),), name="exch_z_elem")(z)[:, 0]
                         + MLP(self.readout_widths, 1, name="exch_z_mlp")(x_m)[:, 0])
        nb = len(self.exch_b_init)
        b = nn.softplus(ElementTable(tuple(_isp(v) for v in self.exch_b_init), name="exch_b_elem")(z)
                        + MLP(self.readout_widths, nb, name="exch_b_mlp")(x_m))
        a = nn.softplus(self.param("exch_a_raw", lambda k, sh: jnp.full(sh, _isp(self.exch_a_init)), (nb,)))
        screen = jnp.sum(a * jnp.exp(-0.5 * (b[pi] + b[pj]) * rp[:, None]), axis=-1)
        exch = pair_sum(_nlh_pair(z[pi], z[pj], rp) + KE_KCAL * zt[pi] * zt[pj] / rp * screen)

        # dispersion: QDO over A-B pairs from the dimer-pass ratios
        disp = pair_sum(QDOPairDispersion(name="qdo")(z, a0r, c6r, pi, pj, rp, batch["node_mask"]))

        # per-atom residual, AB - A - B
        nout = len(self.outputs)
        eps = MLP(self.readout_widths, nout, name="residual")(x)
        eps = eps * batch["node_sign"][:, None]
        d_eps = segment_sum(eps, batch["node_dimer"], nseg)[:num_dimers]

        # learnable scales on the classical priors (init 1): T-M charges are not an
        # energy minimum, so E_C[q(AB)] - E_C[q(A)+q(B)] need not have induction's sign
        k_cl = self.param("classical_scale", nn.initializers.ones_init(), (4,))
        elst, exch, ind, disp = k_cl[0] * elst, k_cl[1] * exch, k_cl[2] * ind, k_cl[3] * disp
        if self.path == "B":
            comps = [elst + ind + d_eps[:, 0], exch + d_eps[:, 1], disp + d_eps[:, 2]]
        else:
            comps = [elst + d_eps[:, 0], exch + d_eps[:, 1], ind + d_eps[:, 2], disp + d_eps[:, 3]]
        aux = dict(elst_classical=elst, ind_classical=ind, exch_classical=exch, disp_classical=disp)
        return comps, aux

    # ---------------------------------------------------------------------- E
    def _dimer_pass(self, batch, bb, x, q, a0r, c6r, z, rvec, rp, pair_sum):
        pi, pj = batch["pair_i"], batch["pair_j"]
        unit = rvec / rp[:, None]
        ev = bb["ev"]                               # (N, 24): degrees 1..4 concatenated
        ylm_f = jnp.concatenate([init_sph_fn(l)(unit) for l in (1, 2, 3, 4)], -1)
        ylm_b = jnp.concatenate([init_sph_fn(l)(-unit) for l in (1, 2, 3, 4)], -1)
        bounds = np.cumsum([0, 3, 5, 7, 9])
        proj = []
        for l in range(4):
            sl = slice(bounds[l], bounds[l + 1])
            ai = jnp.sum(ev[pi][:, sl] * ylm_f[:, sl], -1)
            aj = jnp.sum(ev[pj][:, sl] * ylm_b[:, sl], -1)
            proj += [ai + aj, ai * aj]
        centers = jnp.linspace(0.5, self.pair_cutoff + 3.0, self.num_rbf)
        rbf = jnp.exp(-((rp[:, None] - centers) ** 2) / (2 * 0.5 ** 2))
        h = jnp.concatenate([x[pi] + x[pj], x[pi] * x[pj], rbf, jnp.stack(proj, -1)], -1)
        fcut = _cos_cut(rp, self.pair_cutoff)

        s = nn.softplus(ElementTable((_isp(self.sigma_init),), name="sigma_elem")(z)[:, 0]
                        + MLP(self.readout_widths, 1, name="sigma_mlp")(x)[:, 0])
        sig_p = jnp.sqrt(s[pi] ** 2 + s[pj] ** 2)
        u_ei = MLP(self.pair_widths, 1, name="eli_pair")(h)[:, 0]
        eli = pair_sum(_erf_coulomb(q[pi] * q[pj], rp, sig_p) + fcut * u_ei)

        ea = ElementTable((0.5 * _isp(self.bms_a_init),), name="bms_a_elem")(z)[:, 0]
        eb = ElementTable((0.5 * _isp(self.bms_b_init),), name="bms_b_elem")(z)[:, 0]
        ab_raw = MLP(self.pair_widths, 2, name="exch_pair")(h)
        A = nn.softplus(ea[pi] + ea[pj] + ab_raw[:, 0])
        B = nn.softplus(eb[pi] + eb[pj] + ab_raw[:, 1])
        y = B * rp
        exch = pair_sum(_nlh_pair(z[pi], z[pj], rp) + A * (1 + y + y * y / 3.0) * jnp.exp(-y))

        u_d = MLP(self.pair_widths, 1, name="disp_pair", bias_init=-5.0)(h)[:, 0]
        disp_qdo = QDOPairDispersion(name="qdo")(z, a0r, c6r, pi, pj, rp, batch["node_mask"])
        disp = pair_sum(disp_qdo - fcut * nn.softplus(u_d))

        aux = dict(eli_classical=pair_sum(_erf_coulomb(q[pi] * q[pj], rp, sig_p)),
                   disp_classical=pair_sum(disp_qdo))
        return [eli, exch, disp], aux


def reference_components(targets, path):
    """Map store targets [elst, exch, ind, disp] onto the path's output channels."""
    if OUTPUTS[path][0] == "elst+ind":
        return jnp.stack([targets[:, 0] + targets[:, 2], targets[:, 1], targets[:, 3]], -1)
    return targets


def load_backbone(model_dir, pretrained=True, example_inputs=None, seed=0):
    """Build the released StackNet from ``model_dir`` and its parameters.

    Returns ``(net, params_or_None, provenance)``. ``pretrained=False`` (Path C/E3)
    returns ``None`` for the parameters so the caller initialises from scratch.
    """
    import hashlib
    import json
    import pathlib
    import pickle

    from ml_collections import config_dict

    from so3lr.mlff.config.from_config import make_so3krates_sparse_from_config
    from so3lr.mlff.utils import training_utils

    d = pathlib.Path(model_dir)
    hyper = json.loads((d / "hyperparameters.json").read_text())
    cfg = config_dict.ConfigDict(hyper)
    net = make_so3krates_sparse_from_config(
        cfg, output_intermediate_quantities=list(BACKBONE_QUANTITIES))
    raw = (d / "params.pkl").read_bytes()
    prov = {"model_dir": str(d), "params_sha256": hashlib.sha256(raw).hexdigest(),
            "pretrained": bool(pretrained)}
    params = None
    if pretrained:
        params = training_utils.expand_observables_to_theory_levels(pickle.loads(raw))
    return net, params, prov


def merge_pretrained(fresh, backbone_params):
    """Put the released backbone parameters into a freshly initialised tree.

    Fails if any released array is missing or has a different shape, so a
    silently re-initialised backbone cannot pass as pretrained.
    """
    import flax

    fresh = flax.core.unfreeze(fresh)
    src = flax.core.unfreeze(backbone_params)["params"]
    dst = fresh["params"]["backbone"]
    flat_src = flax.traverse_util.flatten_dict(src)
    flat_dst = flax.traverse_util.flatten_dict(dst)
    for k, v in flat_src.items():
        if k not in flat_dst:
            raise KeyError(f"released parameter {'/'.join(k)} not in the model")
        if tuple(flat_dst[k].shape) != tuple(np.shape(v)):
            raise ValueError(f"{'/'.join(k)}: {np.shape(v)} != {flat_dst[k].shape}")
        flat_dst[k] = jnp.asarray(v, flat_dst[k].dtype)
    extra = sorted('/'.join(k) for k in set(flat_dst) - set(flat_src))
    fresh["params"]["backbone"] = flax.traverse_util.unflatten_dict(flat_dst)
    return fresh, extra
