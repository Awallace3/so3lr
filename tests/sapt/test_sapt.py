"""SO3LR-SAPT heads: separation limit, pretrained loading, batching.

Uses a synthetic charged dimer so the tests need no store on disk.
"""
import functools
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from so3lr.sapt.batch import make_batch
from so3lr.sapt.model import (KE_KCAL, SAPTModel, THREE_PASS, load_backbone,
                              merge_pretrained)

MODEL_DIR = pathlib.Path(__file__).resolve().parents[2] / "so3lr" / "models" / "so3lr-tauri-m-2606"


class _Store:
    """Two dimers: NH4+ ... H2O (charged) and H2O ... H2O (neutral)."""

    def __init__(self, shift=0.0):
        nh4 = np.array([[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                        [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]], float)
        w = np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]], float)
        off = np.array([2.9 + shift, 0.4, 0.1])
        self.rows = [
            (np.array([7, 1, 1, 1, 1, 8, 1, 1]), np.r_[nh4, w + off], (5, 3), (1, 0)),
            (np.array([8, 1, 1, 8, 1, 1]), np.r_[w, w + off], (3, 3), (0, 0)),
        ]

    def dimer(self, d):
        z, R, n, q = self.rows[d]
        return z, R, np.array(n), np.array(q), np.zeros(4)


@functools.lru_cache(maxsize=None)
def _backbone():
    return load_backbone(MODEL_DIR)


def _model(path, store, per_residue_embed=True, perturb=True):
    net, bbp, _ = _backbone()
    b = make_batch(store, [0, 1], three_pass=THREE_PASS[path], per_residue_embed=per_residue_embed)
    m = SAPTModel(backbone=net, path=path)
    p = m.init(jax.random.PRNGKey(0), b)
    p, _ = merge_pretrained(p, bbp)
    if perturb:   # make the zero-initialised residual heads non-trivial
        keys = iter(jax.random.split(jax.random.PRNGKey(1), 1000))
        p = jax.tree_util.tree_map_with_path(
            lambda kp, v: v + 0.05 * jax.random.normal(next(keys), v.shape)
            if "backbone" not in jax.tree_util.keystr(kp) else v, p)
    return m, p


@pytest.mark.parametrize("path", ["A", "B", "E"])
def test_separation_limit_is_the_ion_coulomb_tail(path):
    """At R = 1e4 A every term vanishes except k_e Q_A Q_B / R in the elst channel."""
    far = _Store(shift=1.0e4)
    m, p = _model(path, far)
    b = make_batch(far, [0, 1], three_pass=THREE_PASS[path])
    y = np.asarray(m.apply(p, b)[0])
    # NH4+ ... H2O: Q_B = 0, so even the ion tail is zero up to dipole terms ~1/R^2
    np.testing.assert_allclose(y, 0.0, atol=2e-3)


def test_per_monomer_embedding_is_what_closes_the_limit():
    """Without it the AB pass embeds Q_A + Q_B and the residual leaks at infinity."""
    far = _Store(shift=1.0e4)
    m, p = _model("A", far, per_residue_embed=False)
    b = make_batch(far, [0, 1], three_pass=True, per_residue_embed=False)
    y = np.asarray(m.apply(p, b)[0])
    assert np.abs(y[1]).max() < 2e-3            # neutral dimer unaffected
    assert np.abs(y[0]).max() > 1e-2            # charged dimer leaks


def test_released_backbone_loads_unchanged():
    net, bbp, prov = _backbone()
    assert prov["params_sha256"] == "7293f2c12e0d807f73c0d7f2b70c5bee5779af815dae9feacd2091b031753426"
    m, p = _model("A", _Store(), perturb=False)
    src = jax.tree_util.tree_leaves(bbp["params"])
    dst = jax.tree_util.tree_leaves(p["params"]["backbone"])
    assert len(src) == len(dst)
    for a, c in zip(src, dst):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(c))


def test_pairs_cover_exactly_the_cross_monomer_atoms():
    b = make_batch(_Store(), [0, 1], three_pass=True)
    real = b["pair_mask"] > 0
    assert real.sum() == 5 * 3 + 3 * 3
    seg = b["pair_dimer"][real]
    assert (np.bincount(seg) == [15, 9]).all()
    # monomer-pass nodes carry sign -1, AB nodes +1, and every AB node maps to a monomer node
    ab = (b["node_sign"] > 0)
    assert ab.sum() == 14 and (b["node_sign"] < 0).sum() == 14
    np.testing.assert_array_equal(b["atomic_numbers"][b["mon_node"][ab]], b["atomic_numbers"][ab])
