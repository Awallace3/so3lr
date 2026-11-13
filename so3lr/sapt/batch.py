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

Shapes are padded up to bucket sizes so ``jit`` recompiles only a handful of
times; one padding graph owns all padding nodes, and padding edges are
self-loops on the last node (``d = 0``, which every SO3LR module masks).
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


def make_batch(store, ids, *, three_pass, cutoff=4.5, quanta=None,
               per_residue_embed=True):
    """Return a dict of numpy arrays for the dimers ``ids`` of ``store``."""
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

    num_dimers = len(ids)
    N = _bucket(n_node, q["nodes"])
    G = g + 1
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
    idx_i, _ = cat(ei, pad_node, _bucket(e_raw, q["edges"]), np.int32)
    idx_j, _ = cat(ej, pad_node, len(idx_i), np.int32)
    l_raw = sum(len(a) for a in li)
    idx_i_lr, _ = cat(li, pad_node, _bucket(l_raw, q["lr"]), np.int32)
    idx_j_lr, _ = cat(lj, pad_node, len(idx_i_lr), np.int32)
    lr_on_ab, _ = cat(lab, 0.0, len(idx_i_lr), np.float32)
    p_raw = sum(len(a) for a in pi)
    pair_i, _ = cat(pi, pad_node, _bucket(p_raw, q["pairs"]), np.int32)
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
        targets=np.asarray(targets, np.float32).reshape(num_dimers, 4),
        dimer_natoms=np.asarray(natoms, np.int32).reshape(num_dimers, 2),
        dimer_charges=np.asarray(charges, np.float32).reshape(num_dimers, 2),
        residue_segments=residue_segments, residue_charge=residue_charge,
    )
    if per_residue_embed:
        out["residue_charge_embed"] = residue_charge
        out["residue_unpaired_embed"] = residue_unpaired
    return out


def backbone_inputs(batch):
    """The subset of ``batch`` the released StackNet reads."""
    keys = ("positions", "atomic_numbers", "idx_i", "idx_j", "idx_i_lr", "idx_j_lr",
            "batch_segments", "node_mask", "graph_mask", "total_charge",
            "num_unpaired_electrons", "theory_mask", "residue_segments",
            "residue_charge", "residue_charge_embed", "residue_unpaired_embed")
    return {k: batch[k] for k in keys if k in batch}
