"""Unit tests for Workflow.perform_clustering_analysis,
visualize_embedding, visualize_conditional_attribution_scores and
visualize_knn.

Package boundary: ``cavachon.workflow``. ClusterAnalysis,
InteractiveVisualization and AnnDataUtils are MagicMocks (conftest
``deps``); os.makedirs is a recorder. mdata is a tiny in-memory MuData.
"""

import os
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from test.workflow.conftest import make_config, make_mdata, make_workflow


def clustering_cfg(component, modality="RNA", use_rep="z", min_n_obs=5):
    return SimpleNamespace(
        component=component, modality=modality, use_rep=use_rep, min_n_obs=min_n_obs
    )


def clustering_workflow(configs, components=None):
    config = make_config()
    config.dataset.batch_size = 16
    config.analysis.clustering = configs
    wf = make_workflow(config)
    wf.model.components = components or {}
    return wf


# --------------------------------------------------------------------------
# perform_clustering_analysis
# --------------------------------------------------------------------------


def test_clustering_z_uses_cluster_log_probability(deps):
    wf = clustering_workflow([clustering_cfg("comp", min_n_obs=9)])
    wf.perform_clustering_analysis()

    deps.ClusterAnalysis.assert_called_once_with(wf.mdata, wf.model)
    analysis = deps.ClusterAnalysis.return_value
    analysis.compute_cluster_log_probability.assert_called_once_with(
        modality="RNA",
        component="comp",
        batch_size=16,
        min_n_obs=9,
        batch_effect_colnames=wf.batch_effect_colnames,
        distribution_names=wf.distribution_names,
    )
    analysis.compute_cluster_log_probability_z_hat.assert_not_called()
    analysis.compute_integrated_cluster_log_probability.assert_not_called()


def test_clustering_z_hat_with_learned_prior(deps):
    comp = SimpleNamespace(learn_z_hat_priors=True, z_hat_prior_parameterizer=object())
    wf = clustering_workflow(
        [clustering_cfg("child", use_rep="z_hat")], {"child": comp}
    )
    wf.perform_clustering_analysis()

    analysis = deps.ClusterAnalysis.return_value
    analysis.compute_cluster_log_probability_z_hat.assert_called_once_with(
        modality="RNA",
        component="child",
        batch_size=16,
        min_n_obs=5,
        batch_effect_colnames=wf.batch_effect_colnames,
        distribution_names=wf.distribution_names,
    )
    analysis.compute_integrated_cluster_log_probability.assert_not_called()
    analysis.compute_cluster_log_probability.assert_not_called()


@pytest.mark.parametrize(
    "component",
    [
        SimpleNamespace(learn_z_hat_priors=True, z_hat_prior_parameterizer=None),
        SimpleNamespace(learn_z_hat_priors=False, z_hat_prior_parameterizer=object()),
        SimpleNamespace(),  # neither attribute present
    ],
    ids=["prior-missing", "not-learned", "no-attributes"],
)
def test_clustering_z_hat_falls_back_to_integrated(deps, component):
    wf = clustering_workflow(
        [clustering_cfg("child", use_rep="z_hat", min_n_obs=3)], {"child": component}
    )
    wf.perform_clustering_analysis()

    analysis = deps.ClusterAnalysis.return_value
    analysis.compute_integrated_cluster_log_probability.assert_called_once_with(
        modality="RNA", component="child", batch_size=16, min_n_obs=3
    )
    analysis.compute_cluster_log_probability_z_hat.assert_not_called()


def test_clustering_runs_each_config_in_order(deps):
    comp = SimpleNamespace(learn_z_hat_priors=True, z_hat_prior_parameterizer=object())
    wf = clustering_workflow(
        [clustering_cfg("parent"), clustering_cfg("child", use_rep="z_hat")],
        {"child": comp},
    )
    wf.perform_clustering_analysis()

    names = [c[0] for c in deps.ClusterAnalysis.return_value.mock_calls]
    assert names == [
        "compute_cluster_log_probability",
        "compute_cluster_log_probability_z_hat",
    ]


def test_clustering_nothing_configured(deps):
    wf = clustering_workflow([])
    wf.perform_clustering_analysis()
    assert deps.ClusterAnalysis.return_value.mock_calls == []


# --------------------------------------------------------------------------
# visualize_embedding
# --------------------------------------------------------------------------


def embedding_cfg(interactive, color_by="cell_type", use_rep="z_comp", method="umap"):
    return SimpleNamespace(
        embedding_method=method,
        color_by=color_by,
        modality="RNA",
        use_rep=use_rep,
        interactive=interactive,
    )


def embedding_workflow(configs):
    config = make_config()
    config.analysis.visualize_embedding = configs
    return make_workflow(config, mdata=make_mdata())


@pytest.mark.parametrize("interactive, ext", [(True, "html"), (False, "png")])
def test_visualize_embedding_arguments_and_filename(deps, makedirs, interactive, ext):
    wf = embedding_workflow([embedding_cfg(interactive)])
    wf.visualize_embedding()

    makedirs.assert_called_once_with("/out/embeddings", exist_ok=True)
    deps.InteractiveVisualization.embedding.assert_called_once_with(
        adata=mock.ANY,
        title="z_comp of RNA colored with cell_type umap",
        method="umap",
        use_rep="z_comp",
        color="cell_type",
        width=800,
        height=760,
        filename=f"/out/embeddings/z_comp_of_rna_colored_with_cell_type_umap.{ext}",
    )
    kwargs = deps.InteractiveVisualization.embedding.call_args.kwargs
    # the modality AnnData itself (so obsm written by the plot persists)
    assert kwargs["adata"] is wf.mdata.mod["RNA"]


def test_visualize_embedding_nothing_configured_still_makes_dir(deps, makedirs):
    wf = embedding_workflow([])
    wf.visualize_embedding()
    makedirs.assert_called_once_with("/out/embeddings", exist_ok=True)
    deps.InteractiveVisualization.embedding.assert_not_called()


@pytest.mark.xfail(
    strict=True,
    reason="BUG: color_by 'obsm_key/index' puts '/' into the embedding filename",
)
def test_visualize_embedding_obsm_color_stays_in_outdir(deps):
    # InteractiveVisualization.embedding documents color='<obsm_key>/<index>'
    # (colour by a column of an obsm matrix). The '/' ends up in file_label
    # and turns the output into a non-existent subdirectory of outdir.
    wf = embedding_workflow([embedding_cfg(True, color_by="z_comp/0")])
    wf.visualize_embedding()

    filename = deps.InteractiveVisualization.embedding.call_args.kwargs["filename"]
    assert os.path.dirname(filename) == "/out/embeddings"


# --------------------------------------------------------------------------
# visualize_conditional_attribution_scores
# --------------------------------------------------------------------------


def attribution_cfg(with_respect_to):
    return SimpleNamespace(
        modality="RNA",
        component="child",
        with_respect_to=with_respect_to,
        use_cluster="cluster_child",
    )


def test_conditional_attribution_one_plot_per_wrt(deps, makedirs):
    config = make_config()
    config.dataset.batch_size = 8
    config.analysis.conditional_attribution_scores = [
        attribution_cfg(["parent", "child"])
    ]
    wf = make_workflow(config)

    wf.visualize_conditional_attribution_scores()

    makedirs.assert_called_once_with("/out/attribution", exist_ok=True)
    merged = deps.AnnDataUtils.merge_mdata_on_obs_annotation
    assert (
        merged.call_args_list
        == [mock.call(wf.mdata, "cluster_child", wf.batch_effect_colnames)] * 2
    )
    calls = deps.InteractiveVisualization.attribution_score.call_args_list
    assert len(calls) == 2
    for call, wrt in zip(calls, ["parent", "child"]):
        title = f"RNA child attribution wrt {wrt} colored by cluster_child"
        assert call == mock.call(
            mdata=merged.return_value,
            model=wf.model,
            component="child",
            modality="RNA",
            with_respect_to=[wrt],
            use_cluster="cluster_child",
            batch_size=8,
            title=title,
            batch_effect_colnames=wf.batch_effect_colnames,
            distribution_names=wf.distribution_names,
            batch_effect_encoders=wf.dataloader.batch_effect_encoders,
            width=800,
            height=760,
            filename=f"/out/attribution/rna_child_attribution_wrt_{wrt}_colored_by_cluster_child.html",
        )


def test_conditional_attribution_empty_wrt_or_config(deps, makedirs):
    config = make_config()
    config.analysis.conditional_attribution_scores = [attribution_cfg([])]
    wf = make_workflow(config)
    wf.visualize_conditional_attribution_scores()

    deps.InteractiveVisualization.attribution_score.assert_not_called()
    deps.AnnDataUtils.merge_mdata_on_obs_annotation.assert_not_called()


# --------------------------------------------------------------------------
# visualize_knn (not called by run)
# --------------------------------------------------------------------------


def test_visualize_knn_targets_only_z_representations(deps, makedirs):
    mdata = make_mdata(n_vars={"RNA": 2, "ATAC": 2}, n_obs=3)
    mdata.mod["RNA"].obsm["z_a"] = np.zeros((3, 2))
    mdata.mod["RNA"].obsm["x_params"] = np.zeros((3, 2))
    mdata.mod["ATAC"].obsm["z_b"] = np.zeros((3, 2))
    wf = make_workflow(mdata=mdata)

    wf.visualize_knn(use_cluster="cluster_a", n_neighbors=[5, 10])

    calls = deps.InteractiveVisualization.neighbors_with_same_annotations.call_args_list
    assert [(c.kwargs["modality"], c.kwargs["use_rep"]) for c in calls] == [
        ("RNA", "z_a"),
        ("ATAC", "z_b"),
    ]
    first = calls[0]
    assert first.args == (mdata, wf.model)
    title = "z_a of RNA colored with cluster_a [5, 10] neighbors"
    assert first.kwargs["title"] == title
    assert first.kwargs["n_neighbors"] == [5, 10]
    assert first.kwargs["group_by_cluster"] is True
    assert first.kwargs["filename"] == (
        "/out/knn/z_a_of_rna_colored_with_cluster_a_[5,_10]_neighbors.html"
    )
    assert all(
        c == mock.call("/out/knn", exist_ok=True) for c in makedirs.call_args_list
    )
