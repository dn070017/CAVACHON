"""Unit tests for Workflow.perform_differential_analysis and
perform_hierarchical_differential_analysis.

Package boundary: ``cavachon.workflow``. DifferentialAnalysis and
HierarchicalDifferentialAnalysis are MagicMocks (conftest ``deps``) that
return small in-memory DataFrames; os.makedirs and DataFrame.to_csv are
recorders, so the asserted output paths are never written.
"""

import inspect
import os
from types import SimpleNamespace

import pandas as pd
import pytest

from cavachon.environment.constants import Constants
from test.workflow.conftest import make_config, make_mdata, make_workflow

GROUP_A = Constants.DIFFERENTIAL_ANALYSIS_GROUP_A_COLUMN
GROUP_B = Constants.DIFFERENTIAL_ANALYSIS_GROUP_B_COLUMN


def deg_table(k_ab=(5.0, 0.1, -4.0), k_ba=(0.0, 0.2, 0.0)):
    return pd.DataFrame(
        {"K(A>B|Z)": list(k_ab), "K(B>A|Z)": list(k_ba)},
        index=[f"gene{i}" for i in range(len(k_ab))],
    )


def deg_cfg(**overrides):
    cfg = dict(
        modality="RNA",
        component="comp",
        use_cluster="cluster_comp",
        group_a="",
        group_b="",
        z_sampling_size=3,
        x_sampling_size=50,
        batch_size=12,
        keep_only_significant=False,
        sort_output=True,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def deg_workflow(configs, obs=None):
    config = make_config()
    config.analysis.differential_analysis = configs
    mdata = make_mdata({"RNA": obs or {"cluster_comp": ["a", "a", "b", "b", "c", "c"]}})
    return make_workflow(config, mdata=mdata)


# --------------------------------------------------------------------------
# perform_differential_analysis
# --------------------------------------------------------------------------


def test_deg_constructs_analysis_with_workflow_state(deps, makedirs):
    deps.DifferentialAnalysis.return_value.across_clusters_pairwise.return_value = {}
    wf = deg_workflow([deg_cfg()])
    wf.perform_differential_analysis()

    makedirs.assert_called_once_with("/out/differential_analysis", exist_ok=True)
    deps.DifferentialAnalysis.assert_called_once_with(
        wf.mdata,
        wf.model,
        wf.batch_effect_colnames,
        wf.distribution_names,
        wf.dataloader.batch_effect_encoders,
    )


def test_deg_positional_order_matches_signature():
    from cavachon.tools.differential_analysis import DifferentialAnalysis

    names = list(inspect.signature(DifferentialAnalysis.__init__).parameters)[1:6]
    assert names == [
        "mdata",
        "model",
        "batch_effect_colnames",
        "distribution_names",
        "batch_effect_encoders",
    ]


def test_deg_pairwise_mode(deps, to_csv_calls):
    # GroupA/GroupB are set by DifferentialAnalysis; the workflow keeps them
    table = deg_table()
    table[GROUP_A] = "a"
    table[GROUP_B] = "b"
    deps.DifferentialAnalysis.return_value.across_clusters_pairwise.return_value = {
        "a/b": table
    }
    wf = deg_workflow([deg_cfg(keep_only_significant=True, sort_output=False)])

    wf.perform_differential_analysis()

    analysis = deps.DifferentialAnalysis.return_value
    analysis.across_clusters_pairwise.assert_called_once_with(
        "comp",
        "RNA",
        "cluster_comp",
        z_sampling_size=3,
        x_sampling_size=50,
        batch_size=12,
        keep_only_significant=True,
        sort_output=False,
    )
    analysis.between_two_groups.assert_not_called()
    assert list(table[GROUP_A]) == ["a"] * 3
    assert list(table[GROUP_B]) == ["b"] * 3
    assert wf.differential_analysis_results == {
        ("RNA", "comp", "cluster_comp"): {"a/b": table}
    }
    assert len(to_csv_calls) == 1
    _, path, kwargs = to_csv_calls[0]
    assert path == "/out/differential_analysis/rna_comp_cluster_comp_a_b.tsv"
    assert kwargs == {"sep": "\t"}


@pytest.mark.parametrize("only_a", [True, False])
def test_deg_single_pair_needs_both_groups(deps, only_a):
    deps.DifferentialAnalysis.return_value.across_clusters_pairwise.return_value = {}
    cfg = deg_cfg(group_a="a" if only_a else "", group_b="" if only_a else "b")
    deg_workflow([cfg]).perform_differential_analysis()

    analysis = deps.DifferentialAnalysis.return_value
    analysis.between_two_groups.assert_not_called()
    analysis.across_clusters_pairwise.assert_called_once()


def test_deg_single_pair_mode(deps, to_csv_calls):
    table = deg_table()
    deps.DifferentialAnalysis.return_value.between_two_groups.return_value = table
    wf = deg_workflow([deg_cfg(group_a="a", group_b="c")])

    wf.perform_differential_analysis()

    call = deps.DifferentialAnalysis.return_value.between_two_groups.call_args
    assert list(call.kwargs["group_a_index"]) == ["cell0", "cell1"]
    assert list(call.kwargs["group_b_index"]) == ["cell4", "cell5"]
    rest = {k: v for k, v in call.kwargs.items() if not k.endswith("_index")}
    assert rest == dict(
        component="comp",
        modality="RNA",
        z_sampling_size=3,
        x_sampling_size=50,
        batch_size=12,
        sort_output=True,
        group_a_label="a",
        group_b_label="c",
    )
    deps.DifferentialAnalysis.return_value.across_clusters_pairwise.assert_not_called()
    # keyed by (modality, component, use_cluster) like the pairwise mode
    assert wf.differential_analysis_results == {
        ("RNA", "comp", "cluster_comp"): {"a/c": table}
    }
    assert [p for _, p, _ in to_csv_calls] == [
        "/out/differential_analysis/rna_comp_cluster_comp_a_c.tsv"
    ]


def test_deg_single_pair_integer_cluster_labels(deps):
    # regression: W2 - config group labels are strings; an integer-coded
    # annotation column never matched them, giving empty groups.
    deps.DifferentialAnalysis.return_value.between_two_groups.return_value = deg_table()
    wf = deg_workflow(
        [deg_cfg(group_a="0", group_b="2")],
        obs={"cluster_comp": [0, 0, 1, 1, 2, 2]},
    )

    wf.perform_differential_analysis()

    call = deps.DifferentialAnalysis.return_value.between_two_groups.call_args
    assert list(call.kwargs["group_a_index"]) == ["cell0", "cell1"]
    assert list(call.kwargs["group_b_index"]) == ["cell4", "cell5"]


@pytest.mark.parametrize(
    "keep, expected", [(True, ["gene0", "gene2"]), (False, ["gene0", "gene1", "gene2"])]
)
def test_deg_single_pair_keep_only_significant(deps, to_csv_calls, keep, expected):
    deps.DifferentialAnalysis.return_value.between_two_groups.return_value = deg_table()
    wf = deg_workflow([deg_cfg(group_a="a", group_b="b", keep_only_significant=keep)])
    wf.perform_differential_analysis()

    written, _, _ = to_csv_calls[0]
    assert list(written.index) == expected


def test_deg_results_accumulate_and_all_are_written(deps, to_csv_calls):
    analysis = deps.DifferentialAnalysis.return_value
    analysis.across_clusters_pairwise.return_value = {
        "a/b": deg_table(),
        "a/c": deg_table(),
    }
    analysis.between_two_groups.return_value = deg_table()
    wf = deg_workflow([deg_cfg(), deg_cfg(group_a="b", group_b="c")])

    wf.perform_differential_analysis()

    assert sorted(p for _, p, _ in to_csv_calls) == [
        "/out/differential_analysis/rna_comp_cluster_comp_a_b.tsv",
        "/out/differential_analysis/rna_comp_cluster_comp_a_c.tsv",
        "/out/differential_analysis/rna_comp_cluster_comp_b_c.tsv",
    ]


def test_deg_filename_has_no_spaces(deps, to_csv_calls):
    # regression: fixed bug, see test_report/UNIT_TEST_REPORT.md
    # cluster labels written by ClusterAnalysis look like 'Cluster 001'.
    # Only the target prefix gets .replace(' ', '_'); the cluster-pair part
    # does not, unlike the HDEG filenames.
    deps.DifferentialAnalysis.return_value.across_clusters_pairwise.return_value = {
        "Cluster 001/Cluster 002": deg_table()
    }
    deg_workflow([deg_cfg()]).perform_differential_analysis()

    _, path, _ = to_csv_calls[0]
    assert " " not in os.path.basename(path)


def test_deg_pairwise_label_with_slash(deps, to_csv_calls):
    # regression: fixed bug, see test_report/UNIT_TEST_REPORT.md
    # The tool already sets GroupA/GroupB; the workflow overwrites them with
    # cluster_pair.split('/'), which raises ValueError for a label such as
    # 'CD4/CD8 T'.
    table = deg_table()
    table[GROUP_A] = "CD4/CD8 T"
    table[GROUP_B] = "B"
    deps.DifferentialAnalysis.return_value.across_clusters_pairwise.return_value = {
        "CD4/CD8 T/B": table
    }
    deg_workflow([deg_cfg()]).perform_differential_analysis()

    written, _, _ = to_csv_calls[0]
    assert set(written[GROUP_A]) == {"CD4/CD8 T"}
    assert set(written[GROUP_B]) == {"B"}


def test_deg_single_pair_configs_on_different_annotations_are_kept(deps, to_csv_calls):
    # regression: fixed bug, see test_report/UNIT_TEST_REPORT.md
    # Two single-pair comparisons of the same labels on two different
    # annotation columns map to the same key (modality, component, a, b)
    # and the same file, so the first result is silently lost.
    analysis = deps.DifferentialAnalysis.return_value
    first, second = deg_table(), deg_table(k_ab=(9.0, 9.0, 9.0))
    analysis.between_two_groups.side_effect = [first, second]
    obs = {
        "cluster_comp": ["a", "a", "b", "b", "b", "b"],
        "celltype": ["a", "b", "a", "b", "a", "b"],
    }
    wf = deg_workflow(
        [
            deg_cfg(group_a="a", group_b="b"),
            deg_cfg(group_a="a", group_b="b", use_cluster="celltype"),
        ],
        obs=obs,
    )
    wf.perform_differential_analysis()

    assert len({p for _, p, _ in to_csv_calls}) == 2


@pytest.mark.parametrize(
    "method",
    ["perform_differential_analysis", "perform_hierarchical_differential_analysis"],
)
def test_unconfigured_analysis_creates_no_source_directory(deps, makedirs, method):
    # regression: fixed bug, see test_report/UNIT_TEST_REPORT.md
    # ApplicationConfig accepts enrichment/differential attribution without a
    # DEG/HDEG config when outdir/<analysis> exists. Every run creates that
    # directory (empty), so a later enrichment-only run passes validation
    # and then fails at the very end in EnrichmentAnalysis.run_directory.
    wf = deg_workflow([])
    getattr(wf, method)()
    makedirs.assert_not_called()


# --------------------------------------------------------------------------
# perform_hierarchical_differential_analysis
# --------------------------------------------------------------------------


def hdeg_cfg(**overrides):
    cfg = dict(
        modality="RNA",
        component="Child Comp",
        use_cluster="cluster_child",
        donor_cluster="",
        recipient_cluster="",
        donor_components=None,
        n_samples=4,
        seed=7,
        batch_size=21,
        sort_output=False,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def hdeg_workflow(configs):
    config = make_config()
    config.analysis.hierarchical_differential_analysis = configs
    return make_workflow(config)


def test_hdeg_constructs_analysis(deps, makedirs):
    deps.HierarchicalDifferentialAnalysis.return_value.across_clusters_pairwise.return_value = {}
    wf = hdeg_workflow([hdeg_cfg()])
    wf.perform_hierarchical_differential_analysis()

    makedirs.assert_called_once_with(
        "/out/hierarchical_differential_analysis", exist_ok=True
    )
    deps.HierarchicalDifferentialAnalysis.assert_called_once_with(
        wf.mdata,
        wf.model,
        wf.batch_effect_colnames,
        wf.distribution_names,
        wf.dataloader.batch_effect_encoders,
    )


def test_hdeg_single_pair_mode(deps, to_csv_calls):
    table = deg_table()
    analysis = deps.HierarchicalDifferentialAnalysis.return_value
    analysis.between_clusters.return_value = table
    cfg = hdeg_cfg(
        donor_cluster="Cluster 001",
        recipient_cluster="T/NK",
        donor_components=["p1", "p2"],
    )
    hdeg_workflow([cfg]).perform_hierarchical_differential_analysis()

    analysis.between_clusters.assert_called_once_with(
        donor_cluster="Cluster 001",
        recipient_cluster="T/NK",
        component="Child Comp",
        modality="RNA",
        use_cluster="cluster_child",
        n_samples=4,
        seed=7,
        batch_size=21,
        donor_components=["p1", "p2"],
        sort_output=False,
    )
    analysis.across_clusters_pairwise.assert_not_called()
    assert len(to_csv_calls) == 1
    written, path, kwargs = to_csv_calls[0]
    assert path == (
        "/out/hierarchical_differential_analysis/"
        "rna_from_child_comp_substitute_p1_p2_cluster_001_to_t_nk.tsv"
    )
    assert kwargs == {"sep": "\t"}
    pd.testing.assert_frame_equal(written, table)


def test_hdeg_pairwise_mode(deps, to_csv_calls):
    analysis = deps.HierarchicalDifferentialAnalysis.return_value
    analysis.across_clusters_pairwise.return_value = {
        "Cluster 001->Cluster 002": deg_table(),
        "Cluster 002->Cluster 001": deg_table(),
    }
    hdeg_workflow([hdeg_cfg()]).perform_hierarchical_differential_analysis()

    analysis.across_clusters_pairwise.assert_called_once_with(
        component="Child Comp",
        modality="RNA",
        use_cluster="cluster_child",
        donor_components=None,
        n_samples=4,
        seed=7,
        batch_size=21,
        sort_output=False,
    )
    analysis.between_clusters.assert_not_called()
    prefix = (
        "/out/hierarchical_differential_analysis/rna_from_child_comp_substitute_all_"
    )
    assert [p for _, p, _ in to_csv_calls] == [
        prefix + "cluster_001_to_cluster_002.tsv",
        prefix + "cluster_002_to_cluster_001.tsv",
    ]


def test_hdeg_pairwise_when_only_donor_given(deps):
    analysis = deps.HierarchicalDifferentialAnalysis.return_value
    analysis.across_clusters_pairwise.return_value = {}
    hdeg_workflow(
        [hdeg_cfg(donor_cluster="a")]
    ).perform_hierarchical_differential_analysis()
    analysis.between_clusters.assert_not_called()
    analysis.across_clusters_pairwise.assert_called_once()
