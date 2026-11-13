"""Read dimers from a ``mastiff-classical-store-v1`` split.

The FF-1 / DXA arms train on these stores, so reading them directly gives the
SO3LR-SAPT arms the exact same rows, split membership and labels. Only the
geometry/label arrays are loaded; the classical-feature columns are ignored.

Layout (one split directory): atoms are stored per dimer as ``[A..., B...]``.

- ``z``            (n_atoms,)   int16   atomic numbers
- ``R``            (n_atoms, 3) float64 positions in Angstrom
- ``atom_offsets`` (n + 1,)     int64   dimer d owns atoms [off[d], off[d+1])
- ``natoms``       (n, 2)       int16   atoms in monomer A, B
- ``charges``      (n, 2)       int16   net charge of monomer A, B (e)
- ``targets``      (n, 4)       float64 [elst, exch, ind, disp] in kcal/mol
- ``pickle_row``   (n,)         int64   row in the source pickle (provenance)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT = "mastiff-classical-store-v1"
FIELDS = ("z", "R", "atom_offsets", "natoms", "charges", "targets", "pickle_row")
COMPONENTS = ("elst", "exch", "ind", "disp")


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


class DimerStore:
    """One split of a classical store, memory-mapped.

    ``verify=True`` checks byte size and SHA-256 of every loaded array against
    the split manifest; ``expect_manifest_sha256`` pins the manifest itself so a
    lock can fail closed on a different store.
    """

    def __init__(self, root, split, verify=True, expect_manifest_sha256=None,
                 mmap=True):
        self.dir = Path(root) / split
        manifest_path = self.dir / "manifest.json"
        self.manifest_sha256 = sha256_file(manifest_path)
        if expect_manifest_sha256 and self.manifest_sha256 != expect_manifest_sha256:
            raise ValueError(f"{self.dir}: manifest sha256 {self.manifest_sha256} "
                             f"!= expected {expect_manifest_sha256}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format") != FORMAT or self.manifest.get("split") != split:
            raise ValueError(f"{self.dir} is not a {FORMAT} '{split}' split")
        files = self.manifest["files"]
        arrays = {}
        for name in FIELDS:
            path = self.dir / f"{name}.npy"
            spec = files[name]
            if path.stat().st_size != spec["bytes"]:
                raise ValueError(f"{path}: size {path.stat().st_size} != {spec['bytes']}")
            if verify and sha256_file(path) != spec["sha256"]:
                raise ValueError(f"{path}: sha256 mismatch")
            arrays[name] = np.load(path, mmap_mode="r" if mmap else None,
                                   allow_pickle=False)
        self.z = arrays["z"]
        self.R = arrays["R"]
        self.atom_offsets = np.asarray(arrays["atom_offsets"], dtype=np.int64)
        self.natoms = np.asarray(arrays["natoms"], dtype=np.int64)
        self.charges = np.asarray(arrays["charges"], dtype=np.int64)
        self.targets = np.asarray(arrays["targets"], dtype=np.float64)
        self.pickle_row = np.asarray(arrays["pickle_row"], dtype=np.int64)
        self.n = len(self.pickle_row)
        counts = np.diff(self.atom_offsets)
        if not np.array_equal(counts, self.natoms.sum(1)):
            raise ValueError(f"{self.dir}: natoms does not match atom_offsets")
        if self.targets.shape != (self.n, 4):
            raise ValueError(f"{self.dir}: targets shape {self.targets.shape}")
        if self.manifest.get("records") not in (None, self.n):
            raise ValueError(f"{self.dir}: {self.n} rows != manifest records")

    def __len__(self):
        return self.n

    def n_atoms(self, ids=None):
        counts = np.diff(self.atom_offsets)
        return counts if ids is None else counts[np.asarray(ids)]

    def dimer(self, d):
        """Return (z, R, natoms(2,), charges(2,), targets(4,)) for one dimer."""
        lo, hi = self.atom_offsets[d], self.atom_offsets[d + 1]
        return (np.asarray(self.z[lo:hi], dtype=np.int64),
                np.asarray(self.R[lo:hi], dtype=np.float64),
                self.natoms[d], self.charges[d], self.targets[d])

    def provenance(self):
        files = self.manifest["files"]
        return {
            "path": str(self.dir),
            "dataset_id": self.manifest.get("dataset_id"),
            "split": self.manifest.get("split"),
            "records": self.n,
            "manifest_sha256": self.manifest_sha256,
            "arrays": {k: files[k]["sha256"] for k in FIELDS},
        }
