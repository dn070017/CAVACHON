#!/usr/bin/env python3
"""
Diagnose whether z_Shared captures the information shared by RNA and ATAC.

  (a) shared ceiling: the CCA subspace between z_RNA_All and z_ATAC_All,
      compared with z_Shared (LR / kNN on cell type, R² of the subspace)
  (b) leakage: how much of the other modality's root and of the shared
      subspace the residual latents (z_RNA_Only, z_ATAC_Only) explain
  (c) integration ceiling: z_hat_Shared vs the concat of its inputs and
      of its inputs plus the ancestors' z_hat
  (d) collapse: active units and participation ratio of each z

Only obs and the needed obsm keys are read (with h5py).

  python diagnose_shared.py [--input mdata.h5mu] [--label cell_type]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from anndata.io import read_elem
except ImportError:
    from anndata.experimental import read_elem

DEFAULT_INPUT = "../ablation_study/ablation_study/ceren_z_hat_dist/mdata/mdata.h5mu"
KEYS = [
    "z_RNA_All", "z_ATAC_All", "z_RNA_Only", "z_ATAC_Only", "z_Shared",
    "z_hat_RNA_Only", "z_hat_ATAC_Only", "z_hat_Shared",
]


def load(path: Path, keys, label: str):
    """Return ({key: array}, labels), all aligned on the label's obs_names."""
    embeddings, y = {}, None
    with h5py.File(path, "r") as f:
        groups = [f] + [f["mod"][m] for m in f.get("mod", {})]
        for group in groups:
            obs = read_elem(group["obs"]) if "obs" in group else None
            if y is None and obs is not None and label in obs.columns:
                y = obs[label]
            for key in keys:
                if key not in embeddings and "obsm" in group and key in group["obsm"]:
                    emb = read_elem(group["obsm"][key])
                    emb = emb.toarray() if hasattr(emb, "toarray") else np.asarray(emb)
                    embeddings[key] = pd.DataFrame(emb, index=obs.index)
    missing = [k for k in keys if k not in embeddings]
    if missing:
        raise KeyError(f"obsm keys not found: {missing}")
    if y is None:
        raise KeyError(f"obs column '{label}' not found")
    y = y.dropna()
    common = y.index
    for emb in embeddings.values():
        common = common.intersection(emb.index)
    o = {k: v.loc[common].values.astype(np.float64) for k, v in embeddings.items()}
    return o, np.asarray(y.loc[common].astype(str))


def lr(X, y):
    """5-fold balanced accuracy; C=1e3 so that L2 barely depends on scale."""
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1e3, max_iter=5000))
    return cross_val_score(model, X, y, cv=5, scoring="balanced_accuracy").mean()


def knn_purity(X, y, k=15):
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    return (y[idx] == y[:, None]).mean()


def r2(X, Y):
    """Fraction of the variance of Y linearly explained by X (5-fold)."""
    return cross_val_score(LinearRegression(), X, Y, cv=5, scoring="r2").mean()


def report(name, X, y):
    print(f"  {name:<34s} dim={X.shape[1]:>3d}  LR(bal)={lr(X, y):.3f}  kNN={knn_purity(X, y):.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--label", default="cell_type")
    parser.add_argument("--n-cca", type=int, default=10)
    args = parser.parse_args()

    o, y = load(Path(args.input), KEYS, args.label)
    print(f"{len(y)} cells, {len(np.unique(y))} classes\n")

    zr, za = o["z_RNA_All"], o["z_ATAC_All"]
    zs, zro, zao = o["z_Shared"], o["z_RNA_Only"], o["z_ATAC_Only"]

    # (a) shared ceiling: CCA subspace between the two roots
    cca = CCA(n_components=args.n_cca, max_iter=2000).fit(zr, za)
    ur, ua = cca.transform(zr, za)
    corr = [np.corrcoef(ur[:, i], ua[:, i])[0, 1] for i in range(args.n_cca)]
    print("(a) shared ceiling")
    print("  canonical corr:", np.round(corr, 3))
    shared_ref = np.hstack([ur, ua])
    report("CCA shared subspace (ceiling)", shared_ref, y)
    report("z_Shared", zs, y)
    report("z_RNA_All", zr, y)
    report("z_ATAC_All", za, y)
    print(f"  R² z_Shared -> shared subspace      {r2(zs, shared_ref):.3f}")
    print(f"  R² shared subspace -> z_Shared      {r2(shared_ref, zs):.3f}")

    # (b) leakage: residuals should not explain the other modality's root
    print("\n(b) leakage (should be ~0)")
    print(f"  R² z_RNA_Only  -> z_ATAC_All        {r2(zro, za):.3f}")
    print(f"  R² z_ATAC_Only -> z_RNA_All         {r2(zao, zr):.3f}")
    print(f"  R² residuals   -> shared subspace   {r2(np.hstack([zro, zao]), shared_ref):.3f}")
    print(f"  R² z_Shared + residuals -> shared   {r2(np.hstack([zs, zro, zao]), shared_ref):.3f}")

    # (c) integration ceiling of z_hat_Shared
    print("\n(c) integration")
    inputs = np.hstack([zs, zao, zro])
    report("z_hat_Shared", o["z_hat_Shared"], y)
    report("concat inputs (z_Shared, residuals)", inputs, y)
    report("inputs + ancestors' z_hat",
           np.hstack([inputs, o["z_hat_ATAC_Only"], o["z_hat_RNA_Only"]]), y)
    print(f"  R² inputs -> z_hat_Shared           {r2(inputs, o['z_hat_Shared']):.3f}")

    # (d) collapse: per-dim variance of the posterior means across cells
    # (a unit is active when Var(mu_d) > 0.01, Burda et al. 2016)
    print("\n(d) active units (Var(mu_d) > 0.01) and participation ratio")
    for key in ["z_RNA_All", "z_ATAC_All", "z_RNA_Only", "z_ATAC_Only", "z_Shared"]:
        var = o[key].var(axis=0)
        eig = np.clip(np.linalg.eigvalsh(np.cov(o[key], rowvar=False)), 0, None)
        pr = eig.sum() ** 2 / (eig ** 2).sum()
        print(f"  {key:<12s} active={int((var > 0.01).sum()):>2d}/{len(var)}  PR={pr:5.2f}  "
              f"var={np.array2string(np.sort(var)[::-1], precision=2, max_line_width=200)}")


if __name__ == "__main__":
    main()
