#!/usr/bin/env python3
"""
Interactive mdata embedding evaluator.

Loads a MuData file containing embeddings from different models,
interactively discovers where embeddings live (obsm keys), resolves
y_label / cluster_label, computes clustering metrics and KNN purity,
and writes InteractiveVisualization HTMLs.

Supports:
  - YAML config (--config) + CLI overrides
  - Required: --input (mdata.h5mu or adata.h5ad) , --outdir
  - .h5ad input is wrapped as a single-modality MuData (named by
    --modality, else uns['cavachon']['name'], else the file stem),
    so no modality prompt is shown
  - Interactive discovery scanning mdata.obsm, mdata.mod[mod].obsm,
    .varm, .uns with suggestions for z* / *z / *embed* patterns
  - Explicit --modality / --use_rep / --y-label / --cluster-label
  - Multiple embeddings in one run, each evaluated in turn:
      CLI:  --use-rep z_A z_B [--cluster-label cl_A cl_B]  (one label for
            all, or one per embedding, paired by position)
      YAML: embeddings:
              - use_rep: z_A
                cluster_label: cl_A
              - use_rep: z_B      # no cluster_label -> default / fallback
            (use_rep may also be a list)
      Interactive: prompts for embeddings one at a time (Enter to finish),
            asking for each embedding's cluster label
  - Fallback scanpy Leiden/Louvain if cluster_label missing
  - Logistic regression accuracy of embedding -> y_label (fit on all cells)
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

# Lazy imports to keep help fast
# muon, scanpy, sklearn imported inside functions


SUGGEST_PATTERNS = ["z*", "*z", "*embed*", "*latent*", "*rep*", "X_*"]
SUGGEST_SUBSTR = ["z_", "embed", "latent", "rep"]


def load_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML not installed; cannot read --config YAML")
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive embedding evaluator for MuData/mdata. "
        "Supports YAML + CLI; interactively discovers embedding keys."
    )
    p.add_argument("--input", dest="input", help="Path to mdata.h5mu / .h5ad")
    p.add_argument("--config", help="YAML config file (optional)")
    p.add_argument("--outdir", help="Output directory")
    p.add_argument("--modality", help="Modality to evaluate (e.g. rna, atac); for .h5ad input, the name given to it")
    p.add_argument("--use-rep", dest="use_rep", nargs="+",
                   help="obsm key(s) for embedding (e.g. z_rna X_emb); each is evaluated in turn")
    p.add_argument("--plane", nargs="+", help="Alias for --use-rep (explicit plane/key)")
    p.add_argument("--y-label", dest="y_label", help="obs column for cell identity (y)")
    p.add_argument("--cluster-label", dest="cluster_label", nargs="+",
                   help="obs column for cluster: one for all embeddings, or one per --use-rep (paired by position)")
    p.add_argument("--method", default=None, choices=["pca", "tsne", "umap"],
                   help="Embedding viz method for InteractiveVisualization")
    p.add_argument("--k", type=int, default=None, help="K for KNN purity (default 15)")
    p.add_argument("--ks", nargs="*", type=int, help="List of K values for purity curve")
    p.add_argument("--resolution", type=float, default=0.8, help="Leiden resolution fallback")
    p.add_argument("--color-schema", dest="color_schema", help="YAML file mapping obs column values to hex colors")
    p.add_argument("--non-interactive", action="store_true", help="Fail instead of prompting")
    p.add_argument("--evaluate-all", action="store_true",
                   help="Evaluate all suggested embeddings instead of picking one")
    args = p.parse_args(argv)
    # plane alias
    if args.plane and not args.use_rep:
        args.use_rep = args.plane
    return args


def merge_config_and_cli(args: argparse.Namespace) -> dict:
    cfg = {}
    if args.config:
        cfg = load_yaml(Path(args.config))
    # YAML keys: input, outdir, modality, use_rep/plane, y_label, cluster_label, method, k, etc.
    merged = dict(cfg)
    for key in ["input", "outdir", "modality", "use_rep", "y_label", "cluster_label", "method", "k", "resolution", "color_schema"]:
        val = getattr(args, key, None)
        if val is not None:
            merged[key] = val
    # also handle plane
    if getattr(args, "plane", None) and "use_rep" not in merged:
        merged["use_rep"] = args.plane
    # explicit CLI --use-rep overrides a YAML 'embeddings' list
    if args.use_rep is not None:
        merged.pop("embeddings", None)
    if args.ks is not None:
        merged["ks"] = args.ks
    merged["non_interactive"] = args.non_interactive or merged.get("non_interactive", False)
    merged["evaluate_all"] = args.evaluate_all or merged.get("evaluate_all", False)
    return merged


def as_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def normalize_embedding_specs(embeddings, use_rep, cluster_label):
    """Turn YAML 'embeddings' / use_rep / cluster_label into per-embedding specs.

    Returns (specs, default_cluster_label). Each spec has 'key' and, when it
    was bound to one, 'cluster_label'. Specs without 'cluster_label' use
    default_cluster_label.
    """
    specs: List[Dict] = []
    if embeddings:
        for item in as_list(embeddings):
            if isinstance(item, dict):
                key = item.get("use_rep") or item.get("key") or item.get("plane")
                if not key:
                    raise ValueError(f"embeddings entry {item} has no 'use_rep'")
                spec = {"key": key}
                if item.get("cluster_label"):
                    spec["cluster_label"] = item["cluster_label"]
                specs.append(spec)
            else:
                specs.append({"key": str(item)})
        labels = as_list(cluster_label)
        return specs, (labels[0] if len(labels) == 1 else None)

    reps = as_list(use_rep)
    labels = as_list(cluster_label)
    if len(labels) > 1:
        if len(labels) != len(reps):
            raise ValueError(
                f"{len(labels)} cluster labels given for {len(reps)} embeddings; "
                "give one cluster label for all, or one per embedding"
            )
        return [{"key": r, "cluster_label": c} for r, c in zip(reps, labels)], None
    return [{"key": r} for r in reps], (labels[0] if labels else None)


def load_mdata(path: Path, modality: Optional[str] = None):
    # .h5ad: wrap the AnnData into a single-modality MuData so the
    # modality is resolved automatically and ClusterAnalysis still works
    if path.suffix.lower() == ".h5ad":
        import anndata
        import mudata  # type: ignore

        adata = anndata.read_h5ad(str(path))
        if not modality:
            cav = adata.uns.get("cavachon")
            modality = cav.get("name") if isinstance(cav, dict) and cav.get("name") else path.stem
        print(f"Loaded AnnData {adata.shape}; treating it as modality '{modality}'")
        return mudata.MuData({modality: adata})

    # Try muon, then mudata, then anndata
    errors = []
    try:
        import muon as mu  # type: ignore
        return mu.read_h5mu(str(path))
    except Exception as e:
        errors.append(f"muon.read_h5mu: {e}")
    try:
        import mudata  # type: ignore
        return mudata.read_h5mu(str(path))
    except Exception as e:
        errors.append(f"mudata.read_h5mu: {e}")
    try:
        import anndata
        return anndata.read_h5ad(str(path))
    except Exception as e:
        errors.append(f"anndata.read_h5ad: {e}")
    raise RuntimeError(f"Failed to load mdata from {path}. Tried muon/mudata/anndata:\n" + "\n".join(errors))


def discover_candidates(mdata) -> List[Dict]:
    """Scan mdata for embedding candidates."""
    candidates = []
    # helper to score suggestion
    def is_suggested(key: str) -> bool:
        kl = key.lower()
        for pat in SUGGEST_PATTERNS:
            if fnmatch.fnmatch(kl, pat.lower()):
                return True
        for sub in SUGGEST_SUBSTR:
            if sub in kl:
                return True
        return False

    # Global mdata
    for store_name in ["obsm", "varm", "uns", "obsp"]:
        store = getattr(mdata, store_name, None)
        if store is None:
            continue
        try:
            keys = list(store.keys())
        except Exception:
            continue
        for k in keys:
            # estimate shape if possible
            shape = None
            try:
                v = store[k]
                shape = getattr(v, "shape", None)
                if callable(shape):
                    shape = None
            except Exception:
                pass
            candidates.append(dict(
                modality="__global__",
                store=store_name,
                key=k,
                suggested=is_suggested(k),
                shape=str(shape) if shape else "?",
            ))

    # Per-modality (MuData)
    mods = getattr(mdata, "mod", None)
    if mods is None:
        # may be mudata attr mod or single AnnData -> treat as one modality named "rna"
        mods = {}
    try:
        mod_names = list(mods.keys())
    except Exception:
        mod_names = []

    for mod in mod_names:
        try:
            adata = mods[mod]
        except Exception:
            continue
        for store_name in ["obsm", "varm", "uns", "obsp", "layers"]:
            store = getattr(adata, store_name, None)
            if store is None:
                continue
            try:
                keys = list(store.keys())
            except Exception:
                continue
            for k in keys:
                shape = None
                try:
                    v = store[k]
                    shape = getattr(v, "shape", None)
                except Exception:
                    pass
                candidates.append(dict(
                    modality=mod,
                    store=store_name,
                    key=k,
                    suggested=is_suggested(k),
                    shape=str(shape) if shape else "?",
                ))
        # also check X itself as candidate
        candidates.append(dict(modality=mod, store="X", key="X", suggested=False, shape=str(getattr(adata, "shape", "?"))))

    # Also check if mdata is AnnData directly
    if not candidates and hasattr(mdata, "obsm"):
        for k in list(getattr(mdata, "obsm", {}).keys()):
            candidates.append(dict(modality="__adata__", store="obsm", key=k, suggested=is_suggested(k), shape="?"))

    return candidates


def print_candidates(candidates: List[Dict]):
    print("\nDiscovered embedding / matrix locations:")
    print(f"{'#':>3}  {'modality':<16} {'store':<6} {'key':<32} {'suggested':<9} {'shape'}")
    print("-" * 90)
    for i, c in enumerate(candidates):
        flag = "★" if c["suggested"] else " "
        print(f"{i:>3}  {c['modality']:<16} {c['store']:<6} {c['key']:<32} {flag:<9} {c['shape']}")
    suggested = [c for c in candidates if c["suggested"] and c["store"] == "obsm"]
    if suggested:
        print("\nSuggested obsm embeddings (z* / *z / *embed*):")
        for c in suggested:
            print(f"  - {c['modality']}:{c['store']}:{c['key']} {c['shape']}")


def interactive_select(candidates: List[Dict], prompt: str, allow_skip=False) -> Optional[Dict]:
    print_candidates(candidates)
    suggested_idx = [i for i, c in enumerate(candidates) if c["suggested"] and c["store"] == "obsm"]
    hint = ""
    if suggested_idx:
        hint = f" [suggested: {', '.join(str(i) for i in suggested_idx)}]"
    print(f"\n{prompt}{hint}")
    print("Enter index (e.g. 3), or 'modality:key' (e.g. rna:z_rna), or 'all' for all suggested.")
    if allow_skip:
        print("Press Enter to skip.")
    raw = input("> ").strip()
    if not raw and allow_skip:
        return None
    if raw.lower() == "all":
        return {"all": True}  # type: ignore
    # try int
    try:
        idx = int(raw)
        if 0 <= idx < len(candidates):
            return candidates[idx]
    except ValueError:
        pass
    # try modality:key
    if ":" in raw:
        mod, key = raw.split(":", 1)
        for c in candidates:
            if c["modality"] == mod and c["key"] == key:
                return c
        # also allow store:key
        for c in candidates:
            if c["key"] == key:
                return c
    # try plain key
    for c in candidates:
        if c["key"] == raw:
            return c
    print(f"Could not resolve '{raw}', please try again.")
    return interactive_select(candidates, prompt, allow_skip=allow_skip)


def resolve_modality(mdata, requested: Optional[str], non_interactive: bool) -> str:
    mods = getattr(mdata, "mod", None)
    if mods is None or len(list(mods.keys())) == 0:
        # single AnnData
        return "__adata__"
    mod_names = list(mods.keys())
    if requested and requested in mod_names:
        return requested
    if len(mod_names) == 1:
        print(f"Using sole modality: {mod_names[0]}")
        return mod_names[0]
    if non_interactive:
        raise ValueError(f"Multiple modalities {mod_names}; --modality required in non-interactive mode")
    print(f"Available modalities: {mod_names}")
    while True:
        raw = input(f"Select modality {mod_names} [default {mod_names[0]}]: ").strip()
        if not raw:
            return mod_names[0]
        if raw in mod_names:
            return raw
        print(f"Invalid modality '{raw}'")


def get_adata(mdata, modality: str):
    if modality in ("__global__", "__adata__"):
        return mdata
    mods = getattr(mdata, "mod", None)
    if mods is not None and modality in mods:
        return mods[modality]
    return mdata


def resolve_obs_column(adata, requested: Optional[str], role: str, non_interactive: bool, fallback_options: Optional[List[str]] = None) -> Optional[str]:
    obs_cols = list(adata.obs.columns) if hasattr(adata, "obs") else []
    if requested and requested in obs_cols:
        print(f"Using {role} = '{requested}'")
        return requested
    if requested and requested not in obs_cols:
        print(f"Warning: requested {role} '{requested}' not in obs columns {obs_cols}")
    if non_interactive:
        if requested:
            return requested
        return None
    print(f"\nobs columns: {obs_cols}")
    if fallback_options:
        print(f"Hint for {role}: {fallback_options}")
    while True:
        raw = input(f"Enter {role} obs column (or leave empty to skip/fallback): ").strip()
        if not raw:
            return None
        if raw in obs_cols:
            return raw
        print(f"Column '{raw}' not found. Available: {obs_cols}")


def ensure_cluster_label(adata, modality: str, cluster_label: Optional[str], use_rep_key: str, non_interactive: bool, resolution: float, outdir: Path) -> str:
    if cluster_label and cluster_label in list(adata.obs.columns):
        print(f"Cluster label '{cluster_label}' present ({adata.obs[cluster_label].nunique()} unique)")
        return cluster_label
    if cluster_label:
        print(f"Cluster label '{cluster_label}' not found; will compute fallback clustering.")
    else:
        print("No cluster label specified; will compute fallback clustering if needed.")
    # ask if should fallback
    if non_interactive:
        # auto fallback
        pass
    else:
        ans = input("Run Leiden clustering fallback? [Y/n]: ").strip().lower()
        if ans in ("n", "no"):
            raise ValueError("No cluster_label available and fallback declined.")

    # Need embedding
    if use_rep_key not in adata.obsm:
        raise ValueError(f"use_rep '{use_rep_key}' not in adata.obsm for clustering fallback. Available: {list(adata.obsm.keys())}")

    import scanpy as sc  # type: ignore
    # Ensure neighbors computed on that rep
    print(f"Computing neighbors on obsm['{use_rep_key}'] and Leiden (resolution={resolution})...")
    # scanpy expects adata with that obsm
    try:
        sc.pp.neighbors(adata, use_rep=use_rep_key, n_neighbors=15)
        try:
            sc.tl.leiden(adata, resolution=resolution, key_added="leiden_fallback")
            new_label = "leiden_fallback"
        except Exception as e:
            print(f"Leiden failed ({e}), trying Louvain...")
            sc.tl.louvain(adata, resolution=resolution, key_added="louvain_fallback")
            new_label = "louvain_fallback"
        print(f"Fallback clustering written to obs['{new_label}'] ({adata.obs[new_label].nunique()} clusters)")
        return new_label
    except Exception as e:
        print(f"Scanpy graph clustering failed ({e}), falling back to sklearn KMeans...")
        from sklearn.cluster import KMeans

        n_clusters = 8
        emb = adata.obsm[use_rep_key]
        if hasattr(emb, "toarray"):
            emb = emb.toarray()
        emb = np.asarray(emb)
        y_candidates = [c for c in ["cell_type", "celltype", "label", "truth"] if c in adata.obs.columns]
        if y_candidates:
            n_clusters = int(adata.obs[y_candidates[0]].nunique())
            n_clusters = max(2, min(n_clusters, 20))
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(emb)
        new_label = "kmeans_fallback"
        adata.obs[new_label] = pd.Categorical(kmeans.labels_.astype(str))
        print(f"Fallback KMeans written to obs['{new_label}'] ({n_clusters} clusters)")
        return new_label


def compute_metrics(
    adata,
    y_label: str,
    cluster_label: str,
    embedding: np.ndarray,
    k: int = 15,
    ks: Optional[List[int]] = None,
    *,
    mdata=None,
    modality: Optional[str] = None,
    use_rep_key: Optional[str] = None,
) -> Dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score

    y_true = adata.obs[y_label].astype(str).values
    y_pred = adata.obs[cluster_label].astype(str).values

    metrics: Dict = {}
    metrics["ari"] = float(adjusted_rand_score(y_true, y_pred))
    metrics["nmi"] = float(normalized_mutual_info_score(y_true, y_pred))
    metrics["homogeneity"] = float(homogeneity_score(y_true, y_pred))
    metrics["completeness"] = float(completeness_score(y_true, y_pred))
    metrics["v_measure"] = float(v_measure_score(y_true, y_pred))
    metrics["n_cells"] = int(adata.n_obs)
    metrics["n_true_classes"] = int(len(np.unique(y_true)))
    metrics["n_pred_clusters"] = int(len(np.unique(y_pred)))

    # KNN purity via Cavachon ClusterAnalysis when possible, fallback to sklearn
    purity_computed = False
    if mdata is not None and modality is not None and use_rep_key is not None:
        try:
            from cavachon.tools.cluster_analysis import ClusterAnalysis

            n_list: List[int] = [k] + (ks or [])
            # ClusterAnalysis expects modality key and obs column with true labels
            ca = ClusterAnalysis(mdata, model=None)  # type: ignore[arg-type]
            df_knn = ca.compute_neighbors_with_same_annotations(
                modality=modality,
                use_cluster=y_label,
                use_rep=use_rep_key,
                n_neighbors=n_list,
            )
            # df columns: Number of Neighbors, Cluster, % of KNN Cells with the Same Cluster
            # mean across all rows for each K approximates purity mean
            for kk in n_list:
                if kk >= len(y_true):
                    continue
                sub = df_knn[df_knn["Number of Neighbors"] == kk]
                if len(sub):
                    col = "% of KNN Cells with the Same Cluster"
                    metrics[f"purity_k{kk}_mean"] = float(sub[col].mean())
                    metrics[f"purity_k{kk}_median"] = float(sub[col].median())
                    metrics[f"purity_k{kk}_std"] = float(sub[col].std(ddof=0))
            if ks:
                curve = {str(kk): float(metrics[f"purity_k{kk}_mean"]) for kk in ks if f"purity_k{kk}_mean" in metrics}
                if curve:
                    metrics["purity_curve"] = curve
            metrics["_knn_source"] = "cavachon.ClusterAnalysis"
            metrics["_knn_table_rows"] = int(len(df_knn))
            purity_computed = True
        except Exception as e:
            print(f"  [warn] Cavachon KNN purity failed ({e}); falling back to sklearn", file=sys.stderr)

    if not purity_computed:
        from sklearn.neighbors import NearestNeighbors

        if embedding.shape[0] != len(y_true):
            raise ValueError(f"Embedding rows {embedding.shape[0]} != n_obs {len(y_true)}")
        nn = NearestNeighbors(n_neighbors=k+1, algorithm="auto").fit(embedding)
        _, indices = nn.kneighbors(embedding)
        neigh = indices[:, 1:]
        y_true_arr = np.array(y_true)
        same = (y_true_arr[neigh] == y_true_arr[:, None]).sum(axis=1) / k
        metrics[f"purity_k{k}_mean"] = float(same.mean())
        metrics[f"purity_k{k}_median"] = float(np.median(same))
        metrics[f"purity_k{k}_std"] = float(same.std())
        if ks:
            curve = {}
            for kk in ks:
                if kk >= len(y_true):
                    continue
                nn2 = NearestNeighbors(n_neighbors=kk+1).fit(embedding)
                _, idx2 = nn2.kneighbors(embedding)
                neigh2 = idx2[:, 1:]
                same2 = (y_true_arr[neigh2] == y_true_arr[:, None]).sum(axis=1) / kk
                curve[str(kk)] = float(same2.mean())
            metrics["purity_curve"] = curve
        metrics["_knn_source"] = "sklearn.NearestNeighbors"

    # Predictive power of the embedding for y_label: multinomial logistic
    # regression fit and scored on all cells (training accuracy, no split)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        y_fit = clf.fit(embedding, y_true).predict(embedding)
        metrics["logreg_accuracy"] = float(accuracy_score(y_true, y_fit))
        metrics["logreg_balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_fit))
    except Exception as e:
        print(f"  [warn] logistic regression failed ({e})", file=sys.stderr)

    df = pd.DataFrame({"y": y_true, "c": y_pred})
    cluster_purity = df.groupby("c")["y"].apply(lambda s: s.value_counts().iloc[0] / len(s)).mean()
    metrics["cluster_majority_purity_mean"] = float(cluster_purity)

    return metrics


def visualize_two(adata, use_rep: str, method: str, y_label: str, cluster_label: str, outdir: Path, modality: str, mdata=None, color_discrete_map: Optional[Dict] = None):
    from cavachon.tools.interactive_visualization import InteractiveVisualization

    outdir.mkdir(parents=True, exist_ok=True)
    cav = adata.uns.get("cavachon")
    if not isinstance(cav, dict):
        adata.uns["cavachon"] = {"name": modality}
    elif "name" not in cav:
        cav["name"] = modality
    safe_rep = use_rep.replace("/", "_")
    f1 = str(outdir / f"{modality}_{safe_rep}_{method}_by_{y_label}.html")
    f2 = str(outdir / f"{modality}_{safe_rep}_{method}_by_{cluster_label}.html")

    print(f"  -> Visualizing {modality}:{use_rep} ({method}) colored by {y_label} -> {f1}")
    InteractiveVisualization.embedding(
        adata=adata,
        method=method,
        use_rep=use_rep,
        color=y_label,
        filename=f1,
        title=f"{modality}:{use_rep} by {y_label}",
        color_discrete_map=color_discrete_map,
        width=800,
        height=800,
    )
    print(f"  -> Visualizing {modality}:{use_rep} ({method}) colored by {cluster_label} -> {f2}")
    InteractiveVisualization.embedding(
        adata=adata,
        method=method,
        use_rep=use_rep,
        color=cluster_label,
        filename=f2,
        title=f"{modality}:{use_rep} by {cluster_label}",
        color_discrete_map=color_discrete_map,
        width=800,
        height=800,
    )
    files = [f1, f2]
    if mdata is not None:
        try:
            f3 = str(outdir / f"{modality}_{safe_rep}_knn_purity_by_{y_label}.html")
            print(f"  -> KNN purity bar (ClusterAnalysis → InteractiveVisualization) -> {f3}")
            from cavachon.tools.cluster_analysis import ClusterAnalysis

            analysis = ClusterAnalysis(mdata, model=None)  # type: ignore[arg-type]
            analysis_result = analysis.compute_neighbors_with_same_annotations(
                modality=modality,
                use_cluster=y_label,
                use_rep=use_rep,
            )
            fig = InteractiveVisualization.bar(
                analysis_result,
                x="Cluster",
                y="Cell Enrichment Score",
                group="Cluster",
                title=f"{modality}:{use_rep} KNN Purity by {y_label}",
                xaxis_title=y_label,
                yaxis_title="Mean Cell Enrichment Score",
                color_discrete_map=color_discrete_map or {},
            )
            mean_purity = (
                analysis_result.groupby("Cluster")["Cell Enrichment Score"]
                .mean()
                .sort_values(ascending=False)
            )
            fig.update_xaxes(categoryorder="array", categoryarray=list(mean_purity.index))
            fig.show()
            if f3.endswith("html"):
                fig.write_html(f3)
            else:
                fig.write_image(f3)
            files.append(f3)
        except Exception as e:
            print(f"  [warn] neighbors_with_same_annotations failed ({e})", file=sys.stderr)
    return files


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = merge_config_and_cli(args)

    input_path = cfg.get("input")
    outdir = cfg.get("outdir")
    if not input_path or not outdir:
        print("ERROR: --input and --outdir are required (via CLI or YAML).", file=sys.stderr)
        print("Example: python scripts/evaluate_embeddings.py --input mdata.h5mu --outdir out/eval", file=sys.stderr)
        print("   or: python scripts/evaluate_embeddings.py --config config.yaml", file=sys.stderr)
        return 2

    input_path = Path(input_path)
    outdir = Path(outdir)
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        return 2
    outdir.mkdir(parents=True, exist_ok=True)

    modality_req = cfg.get("modality")
    y_label_req = cfg.get("y_label") or cfg.get("yLabel") or cfg.get("cell_type")
    try:
        embedding_specs, cluster_label_req = normalize_embedding_specs(
            cfg.get("embeddings"),
            cfg.get("use_rep"),
            cfg.get("cluster_label") or cfg.get("clusterLabel"),
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    method = cfg.get("method") or "tsne"
    k = int(cfg.get("k") or 15)
    ks = cfg.get("ks")
    resolution = float(cfg.get("resolution") or 0.8)
    non_interactive = bool(cfg.get("non_interactive"))
    evaluate_all = bool(cfg.get("evaluate_all"))
    color_schema_path = cfg.get("color_schema")
    color_schema: Dict = {}
    if color_schema_path:
        color_schema_path = Path(color_schema_path)
        if not color_schema_path.exists():
            print(f"Warning: --color-schema not found: {color_schema_path}", file=sys.stderr)
        else:
            color_schema = load_yaml(color_schema_path)
            print(f"Loaded color schema from {color_schema_path}")

    print(f"Loading mdata from {input_path} ...")
    mdata = load_mdata(input_path, modality_req)
    # Quick summary
    try:
        mods = list(getattr(mdata, "mod", {}).keys())
        print(f"Loaded MuData with modalities: {mods}, n_obs={getattr(mdata, 'n_obs', '?')}")
    except Exception:
        print(f"Loaded AnnData: {getattr(mdata, 'shape', '?')}")

    candidates = discover_candidates(mdata)
    if not candidates:
        print("No candidates discovered; is this a valid mdata?", file=sys.stderr)
        return 2

    # Resolve modality
    modality = resolve_modality(mdata, modality_req, non_interactive)

    adata = get_adata(mdata, modality)
    print(f"\nSelected modality: {modality} (n_obs={getattr(adata, 'n_obs', '?')}, n_vars={getattr(adata, 'n_vars', '?')})")
    print(f"obs columns: {list(adata.obs.columns)[:20]}{' ...' if len(list(adata.obs.columns))>20 else ''}")
    print(f"obsm keys: {list(adata.obsm.keys())}")

    # Resolve embedding key(s)
    mod_obsm = list(adata.obsm.keys()) if hasattr(adata, "obsm") else []
    global_obsm = list(getattr(mdata, "obsm", {}).keys())

    # Build suggested list respecting patterns
    def is_suggested_key(key: str) -> bool:
        kl = key.lower()
        return any(fnmatch.fnmatch(kl, p.lower()) for p in SUGGEST_PATTERNS) or any(s in kl for s in SUGGEST_SUBSTR)

    suggested_keys = [k for k in mod_obsm if is_suggested_key(k)]
    global_suggested_keys = [k for k in global_obsm if is_suggested_key(k)]

    if not mod_obsm and not global_obsm:
        print(f"No obsm embeddings in modality '{modality}' or globally. Did you store embeddings under mdata.mod['{modality}'].obsm or mdata.obsm?")
        print("All discovered candidates:")
        print_candidates(candidates)
        return 2

    use_rep_selections: List[Dict] = []

    for spec in embedding_specs:
        key = spec["key"]
        in_mod = key in mod_obsm
        in_global = key in global_obsm
        if in_mod or in_global:
            is_global = in_global and not in_mod
            use_rep_selections.append({**spec, "is_global": is_global})
            source = "global mdata.obsm" if is_global else f"modality {modality}"
            bound = f" with cluster_label '{spec['cluster_label']}'" if spec.get("cluster_label") else ""
            print(f"Using explicit use_rep '{key}' from {source}{bound}")
        else:
            available = mod_obsm + [f"__global__:{k}" for k in global_obsm]
            print(f"Warning: use_rep '{key}' not in obsm keys {available}", file=sys.stderr)
            if non_interactive:
                return 2

    if not use_rep_selections:
        if evaluate_all and (suggested_keys or global_suggested_keys):
            use_rep_selections = [{"key": k, "is_global": False} for k in suggested_keys] + [{"key": k, "is_global": True} for k in global_suggested_keys]
            print(f"--evaluate-all: will evaluate {len(use_rep_selections)} suggested keys")
        elif non_interactive:
            if suggested_keys:
                use_rep_selections = [{"key": suggested_keys[0], "is_global": False}]
                print(f"Non-interactive: picking first suggested '{suggested_keys[0]}' from {suggested_keys}")
            elif global_suggested_keys:
                use_rep_selections = [{"key": global_suggested_keys[0], "is_global": True}]
                print(f"Non-interactive: picking first global suggested '{global_suggested_keys[0]}' from {global_suggested_keys}")
            elif mod_obsm:
                use_rep_selections = [{"key": mod_obsm[0], "is_global": False}]
                print(f"Non-interactive: picking first obsm '{mod_obsm[0]}'")
            else:
                use_rep_selections = [{"key": global_obsm[0], "is_global": True}]
                print(f"Non-interactive: picking first global obsm '{global_obsm[0]}'")
        else:
            # interactive
            print(f"\nSuggested embeddings for modality '{modality}': {suggested_keys or '(none)'}")
            print(f"Suggested global embeddings: {global_suggested_keys or '(none)'}")
            print(f"All modality obsm keys: {mod_obsm}")
            print(f"All global obsm keys: {global_obsm}")
            filt = [c for c in candidates if c["store"] == "obsm" and (c["modality"] == modality or c["modality"] == "__global__")]
            # Prompt repeatedly; each embedding is bound to its own cluster label
            while True:
                n = len(use_rep_selections) + 1
                sel = interactive_select(
                    filt,
                    f"Select embedding #{n} for modality '{modality}' (use_rep)"
                    + (" — Enter to finish" if use_rep_selections else ""),
                    allow_skip=bool(use_rep_selections),
                )
                if sel is None:
                    break
                if "all" in sel:  # type: ignore
                    use_rep_selections += [{"key": k, "is_global": False} for k in suggested_keys] + [{"key": k, "is_global": True} for k in global_suggested_keys]
                    break
                sel_cluster = resolve_obs_column(
                    adata, None, f"cluster_label for '{sel['key']}'", False,
                    fallback_options=["leiden", "louvain", "cluster", "cluster_rna", "cluster_atac"],
                )
                use_rep_selections.append({
                    "key": sel["key"],
                    "is_global": sel["modality"] == "__global__",
                    "cluster_label": sel_cluster,
                })
                print("Selected so far: " + ", ".join(
                    f"{s['key']} ({s.get('cluster_label') or 'fallback clustering'})" for s in use_rep_selections
                ))
            if not use_rep_selections:
                print("No selection made.", file=sys.stderr)
                return 2

    # Resolve y_label and cluster_label interactively
    y_label = resolve_obs_column(adata, y_label_req, "y_label (cell identity)", non_interactive,
                                  fallback_options=["cell_type", "celltype", "label", "truth"])
    if y_label is None:
        # try common fallbacks automatically
        for cand in ["cell_type", "celltype", "cell_type_integrated", "label", "truth", "annotation"]:
            if cand in list(adata.obs.columns):
                y_label = cand
                print(f"Auto-picked y_label='{y_label}'")
                break
    if y_label is None:
        print("No y_label resolved. Need a cell identity column in obs for purity/ARI.", file=sys.stderr)
        if non_interactive:
            return 2
        # ask again without fallback
        y_label = resolve_obs_column(adata, None, "y_label (cell identity)", False)
        if y_label is None:
            print("Aborting: y_label required.", file=sys.stderr)
            return 2

    # Default cluster label, only for embeddings not bound to their own
    cluster_label = None
    if any("cluster_label" not in s for s in use_rep_selections):
        cluster_label = resolve_obs_column(adata, cluster_label_req, "cluster_label", non_interactive,
                                            fallback_options=["leiden", "louvain", "cluster", "cluster_rna", "cluster_atac"])
    # We'll ensure fallback per-embedding if needed

    # Evaluate each embedding
    all_results = {}
    summary_rows = []
    for sel in use_rep_selections:
        use_rep = sel["key"]
        is_global = sel["is_global"]
        if is_global and use_rep in getattr(mdata, "obsm", {}):
            adata.obsm[use_rep] = mdata.obsm[use_rep]
        print(f"\n{'='*70}")
        print(f"Evaluating {modality}:{use_rep} (global={is_global}, method={method}, k={k})")
        print(f"{'='*70}")
        # Ensure cluster label exists (per embedding, since fallback uses that rep)
        cur_cluster_label = sel["cluster_label"] if "cluster_label" in sel else cluster_label
        if cur_cluster_label is None or cur_cluster_label not in list(adata.obs.columns):
            # compute fallback using this embedding
            try:
                cur_cluster_label = ensure_cluster_label(adata, modality, cur_cluster_label, use_rep, non_interactive, resolution, outdir)
            except Exception as e:
                print(f"Failed to obtain cluster label for {use_rep}: {e}", file=sys.stderr)
                continue

        # Extract embedding matrix
        try:
            emb = adata.obsm[use_rep]
        except KeyError:
            print(f"obsm key '{use_rep}' not found, skipping.", file=sys.stderr)
            continue
        # support sparse?
        if hasattr(emb, "toarray"):
            emb = emb.toarray()
        emb = np.asarray(emb)
        if emb.ndim != 2:
            print(f"Embedding {use_rep} has ndim {emb.ndim}, shape {emb.shape}; skipping.", file=sys.stderr)
            continue
        print(f"Embedding shape: {emb.shape}, y_label='{y_label}' ({adata.obs[y_label].nunique()} classes), cluster_label='{cur_cluster_label}' ({adata.obs[cur_cluster_label].nunique()} clusters)")

        metrics = compute_metrics(
            adata, y_label, cur_cluster_label, emb, k=k, ks=ks,
            mdata=mdata, modality=modality, use_rep_key=use_rep,
        )
        print("Metrics:")
        for kk, vv in metrics.items():
            if kk == "purity_curve":
                print(f"  {kk}: {vv}")
            else:
                print(f"  {kk}: {vv:.4f}" if isinstance(vv, float) else f"  {kk}: {vv}")

        try:
            color_discrete_map = color_schema.get(y_label) if isinstance(color_schema, dict) else None
            viz_files = visualize_two(adata, use_rep, method, y_label, cur_cluster_label, outdir, modality, mdata=mdata, color_discrete_map=color_discrete_map)
        except Exception as e:
            print(f"Visualization failed for {use_rep}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            viz_files = []

        # same embedding may be listed with different cluster labels
        result_key = use_rep if use_rep not in all_results else f"{use_rep}:{cur_cluster_label}"
        all_results[result_key] = {
            "modality": modality,
            "use_rep": use_rep,
            "embedding_name": use_rep,
            "embedding_dim": int(emb.shape[1]),
            "outdir": str(outdir),
            "y_label": y_label,
            "cluster_label": cur_cluster_label,
            "method": method,
            "metrics": metrics,
            "visualizations": viz_files,
        }
        row = {
            "modality": modality,
            "use_rep": use_rep,
            "embedding_name": use_rep,
            "embedding_dim": int(emb.shape[1]),
            "outdir": str(outdir),
            "y_label": y_label,
            "cluster_label": cur_cluster_label,
            **metrics,
        }
        summary_rows.append(row)

    # Write summary
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        # order columns
        cols = [
            "modality", "use_rep", "embedding_name", "embedding_dim", "outdir",
            "y_label", "cluster_label", "n_cells", "n_true_classes", "n_pred_clusters",
            f"purity_k{k}_mean", f"purity_k{k}_median", f"purity_k{k}_std", "ari", "nmi",
            "homogeneity", "completeness", "v_measure", "cluster_majority_purity_mean",
            "logreg_accuracy", "logreg_balanced_accuracy",
        ]
        cols = [c for c in cols if c in df.columns]
        df = df[cols + [c for c in df.columns if c not in cols]]
        csv_path = outdir / "embedding_evaluation_summary.csv"
        json_path = outdir / "embedding_evaluation_summary.json"
        df.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nSummary CSV: {csv_path}")
        print(f"Summary JSON: {json_path}")
        print(df.to_string(index=False))
    else:
        print("No embeddings evaluated successfully.", file=sys.stderr)
        return 1

    print(f"\nDone. Outputs in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

