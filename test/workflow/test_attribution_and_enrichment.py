"""Unit tests for Workflow.perform_differential_attribution_analysis and
perform_enrichment_analysis.

Package boundary: ``cavachon.workflow``. DifferentialAttributionAnalysis,
EnrichmentAnalysis (incl. the find_deg_tables / _output_name static
helpers), InteractiveVisualization and _safe_filename are MagicMocks
(conftest ``deps``; _safe_filename maps v -> 'Safe-v'). DEG tables are
in-memory DataFrames handed out by the find_deg_tables mock; os.makedirs
and DataFrame.to_csv are recorders.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from cavachon.environment.constants import Constants
from test.workflow.conftest import make_config, make_mdata, make_workflow

GROUP_A = Constants.DIFFERENTIAL_ANALYSIS_GROUP_A_COLUMN
GROUP_B = Constants.DIFFERENTIAL_ANALYSIS_GROUP_B_COLUMN
DONOR = Constants.HIERARCHICAL_DIFFERENTIAL_ANALYSIS_DONOR_CLUSTER_COLUMN
RECIPIENT = Constants.HIERARCHICAL_DIFFERENTIAL_ANALYSIS_RECIPIENT_CLUSTER_COLUMN

ATTR_ROOT = "/out/differential_attribution_analysis/safe-comp/safe-rna"


def attr_cfg(**overrides):
    cfg = dict(
        analysis_type="deg",
        modality="RNA",
        component="comp",
        use_cluster="cluster_comp",
        with_respect_to=["parent", "comp"],
        column="K(A>B|Z)",
        threshold=2.5,
        threshold_operator="gt",
        use_absolute=False,
        min_num_degs=4,
        n_steps=16,
        recursive=False,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def labelled_table(col_a, col_b, a, b):
    return pd.DataFrame(
        {"K(A>B|Z)": [5.0, 4.0], col_a: [a, a], col_b: [b, b]}, index=["g1", "g2"]
    )


def attr_workflow(configs, cluster=("x", "x", "y", "y", "z", "z")):
    config = make_config()
    config.dataset.batch_size = 40
    config.analysis.differential_attribution_analysis = configs
    mdata = make_mdata({"RNA": {"cluster_comp": list(cluster)}})
    return make_workflow(config, mdata=mdata)


@pytest.fixture
def tables(deps):
    """Configure find_deg_tables / _output_name; returns setter."""
    deps.EnrichmentAnalysis._output_name.side_effect = lambda root, f: f"out-{f.stem}"

    def set_tables(items):
        deps.EnrichmentAnalysis.find_deg_tables.side_effect = lambda *a, **k: iter(
            list(items)
        )

    return set_tables


@pytest.fixture
def attributed(deps):
    frame = pd.DataFrame({"LogFC": [1.0], "abs_contrib_parent": [0.5]}, index=["g1"])
    deps.DifferentialAttributionAnalysis.return_value.from_deg_file.return_value = frame
    return frame


# --------------------------------------------------------------------------
# perform_differential_attribution_analysis
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "analysis_type, source",
    [("deg", "differential_analysis"), ("hdeg", "hierarchical_differential_analysis")],
)
def test_attribution_searches_source_directory(deps, tables, analysis_type, source):
    tables([])
    wf = attr_workflow([attr_cfg(analysis_type=analysis_type)])
    wf.perform_differential_attribution_analysis()

    deps.EnrichmentAnalysis.find_deg_tables.assert_called_once_with(
        f"/out/{source}", column="K(A>B|Z)", recursive=False
    )


def test_attribution_constructs_analysis(deps, tables):
    tables([])
    wf = attr_workflow([attr_cfg()])
    wf.perform_differential_attribution_analysis()

    deps.DifferentialAttributionAnalysis.assert_called_once_with(
        wf.mdata,
        wf.model,
        wf.batch_effect_colnames,
        wf.distribution_names,
        wf.dataloader.batch_effect_encoders,
    )


def test_attribution_no_tables_skips_and_continues(deps, tables, capsys, makedirs):
    tables([])
    wf = attr_workflow([attr_cfg(), attr_cfg(analysis_type="hdeg")])
    wf.perform_differential_attribution_analysis()

    out = capsys.readouterr().out
    assert "No DEG/HDEG tables found in /out/differential_analysis" in out
    assert "No DEG/HDEG tables found in /out/hierarchical_differential_analysis" in out
    assert deps.EnrichmentAnalysis.find_deg_tables.call_count == 2
    deps.DifferentialAttributionAnalysis.return_value.from_deg_file.assert_not_called()
    makedirs.assert_not_called()


def test_attribution_deg_call_arguments(deps, tables, attributed):
    path = Path("/out/differential_analysis/rna_comp_cluster_comp_x_z.tsv")
    tables([(path, labelled_table(GROUP_A, GROUP_B, "x", "z"))])
    wf = attr_workflow([attr_cfg()])
    wf.perform_differential_attribution_analysis()

    call = deps.DifferentialAttributionAnalysis.return_value.from_deg_file.call_args
    assert list(call.kwargs.pop("group_a_index")) == ["cell0", "cell1"]
    assert list(call.kwargs.pop("group_b_index")) == ["cell4", "cell5"]
    assert call.kwargs == dict(
        deg_file=str(path),
        component="comp",
        modality="RNA",
        with_respect_to=["parent", "comp"],
        column="K(A>B|Z)",
        threshold=2.5,
        threshold_operator="gt",
        use_absolute=False,
        min_num_degs=4,
        batch_size=40,
        n_steps=16,
    )


def test_attribution_hdeg_uses_donor_and_recipient(deps, tables, attributed):
    path = Path("/out/hierarchical_differential_analysis/f.tsv")
    tables([(path, labelled_table(DONOR, RECIPIENT, "y", "x"))])
    wf = attr_workflow([attr_cfg(analysis_type="hdeg")])
    wf.perform_differential_attribution_analysis()

    call = deps.DifferentialAttributionAnalysis.return_value.from_deg_file.call_args
    assert list(call.kwargs["group_a_index"]) == ["cell2", "cell3"]
    assert list(call.kwargs["group_b_index"]) == ["cell0", "cell1"]


def test_attribution_writes_table_and_radviz(
    deps, tables, attributed, makedirs, to_csv_calls
):
    path = Path("/out/differential_analysis/sub/t1.tsv")
    tables([(path, labelled_table(GROUP_A, GROUP_B, "x", "y"))])
    wf = attr_workflow([attr_cfg()])
    wf.perform_differential_attribution_analysis()

    deps._safe_filename.assert_any_call("comp")
    deps._safe_filename.assert_any_call("RNA")
    makedirs.assert_called_once_with(ATTR_ROOT, exist_ok=True)
    deps.EnrichmentAnalysis._output_name.assert_called_once_with(
        Path("/out/differential_analysis"), path
    )
    assert len(to_csv_calls) == 1
    written, out_path, kwargs = to_csv_calls[0]
    assert out_path == f"{ATTR_ROOT}/out-t1.tsv"
    assert kwargs == {"sep": "\t"}
    pd.testing.assert_frame_equal(written, attributed)
    deps.InteractiveVisualization.differential_attribution_radviz.assert_called_once_with(
        data=attributed,
        with_respect_to=["parent", "comp"],
        filename=f"{ATTR_ROOT}/out-t1_radviz.html",
        title="Differential attribution: comp (RNA)",
    )


def test_attribution_single_wrt_skips_radviz(deps, tables, attributed, to_csv_calls):
    tables(
        [
            (
                Path("/out/differential_analysis/t.tsv"),
                labelled_table(GROUP_A, GROUP_B, "x", "y"),
            )
        ]
    )
    wf = attr_workflow([attr_cfg(with_respect_to=["parent"])])
    wf.perform_differential_attribution_analysis()

    assert len(to_csv_calls) == 1
    deps.InteractiveVisualization.differential_attribution_radviz.assert_not_called()


@pytest.mark.parametrize("result", [None, pd.DataFrame()], ids=["none", "empty"])
def test_attribution_empty_result_writes_nothing(
    deps, tables, makedirs, to_csv_calls, result
):
    deps.DifferentialAttributionAnalysis.return_value.from_deg_file.return_value = (
        result
    )
    tables(
        [
            (
                Path("/out/differential_analysis/t1.tsv"),
                labelled_table(GROUP_A, GROUP_B, "x", "y"),
            ),
            (
                Path("/out/differential_analysis/t2.tsv"),
                labelled_table(GROUP_A, GROUP_B, "y", "z"),
            ),
        ]
    )
    wf = attr_workflow([attr_cfg()])
    wf.perform_differential_attribution_analysis()

    assert (
        deps.DifferentialAttributionAnalysis.return_value.from_deg_file.call_count == 2
    )
    makedirs.assert_not_called()
    assert to_csv_calls == []
    deps.InteractiveVisualization.differential_attribution_radviz.assert_not_called()


@pytest.mark.xfail(
    strict=True,
    reason="BUG: attribution runs on DEG tables of every component/modality in the directory",
)
def test_attribution_only_uses_tables_of_configured_component(deps, tables, attributed):
    # The DEG directory holds tables for every differential_analysis entry
    # (prefix '<modality>_<component>_<use_cluster>_'). An attribution entry
    # for component 'comp_b' also attributes comp_a's table, through
    # comp_b's generative path and comp_b's cluster column, writing a
    # wrong result into the comp_b output directory.
    root = Path("/out/differential_analysis")
    tables(
        [
            (
                root / "rna_comp_a_cluster_comp_a_x_y.tsv",
                labelled_table(GROUP_A, GROUP_B, "x", "y"),
            ),
            (
                root / "rna_comp_b_cluster_comp_b_x_y.tsv",
                labelled_table(GROUP_A, GROUP_B, "x", "y"),
            ),
        ]
    )
    wf = attr_workflow([attr_cfg(component="comp_b", use_cluster="cluster_comp")])
    wf.perform_differential_attribution_analysis()

    calls = (
        deps.DifferentialAttributionAnalysis.return_value.from_deg_file.call_args_list
    )
    assert [Path(c.kwargs["deg_file"]).name for c in calls] == [
        "rna_comp_b_cluster_comp_b_x_y.tsv"
    ]


@pytest.mark.xfail(
    strict=True,
    reason="BUG: str() group labels never match non-string cluster columns (empty groups)",
)
def test_attribution_non_string_cluster_labels(deps, tables, attributed):
    # A DEG table read back from TSV has integer GroupA/GroupB when the
    # annotation column holds integers. The workflow str()s the label but
    # compares it to the raw obs column, so both group indices are empty and
    # from_deg_file is still called.
    tables(
        [
            (
                Path("/out/differential_analysis/t.tsv"),
                labelled_table(GROUP_A, GROUP_B, 0, 2),
            )
        ]
    )
    wf = attr_workflow([attr_cfg()], cluster=(0, 0, 1, 1, 2, 2))
    wf.perform_differential_attribution_analysis()

    call = deps.DifferentialAttributionAnalysis.return_value.from_deg_file.call_args
    assert list(call.kwargs["group_a_index"]) == ["cell0", "cell1"]
    assert list(call.kwargs["group_b_index"]) == ["cell4", "cell5"]


# --------------------------------------------------------------------------
# perform_enrichment_analysis
# --------------------------------------------------------------------------


def enrich_cfg(analysis_type="deg", gene_sets="KEGG_2019_Mouse"):
    return SimpleNamespace(
        analysis_type=analysis_type,
        gene_sets=gene_sets,
        organism="Mouse",
        column="K(B>A|Z)",
        recursive=False,
        terms=["Apoptosis"],
        metric="NOM p-val",
        threshold=0.01,
        threshold_operator="lt",
        min_num_degs=3,
        permutation_num=50,
        min_size=5,
        max_size=60,
    )


@pytest.mark.parametrize(
    "analysis_type, source",
    [("deg", "differential_analysis"), ("hdeg", "hierarchical_differential_analysis")],
)
def test_enrichment_arguments(deps, analysis_type, source):
    config = make_config()
    config.analysis.enrichment_analysis = [enrich_cfg(analysis_type)]
    make_workflow(config).perform_enrichment_analysis()

    deps._safe_filename.assert_called_once_with("KEGG_2019_Mouse")
    deps.EnrichmentAnalysis.assert_called_once_with(
        gene_sets="KEGG_2019_Mouse", organism="Mouse"
    )
    deps.EnrichmentAnalysis.return_value.run_directory.assert_called_once_with(
        input_dir=f"/out/{source}",
        outdir="/out/enrichment_analysis/safe-kegg_2019_mouse",
        column="K(B>A|Z)",
        recursive=False,
        terms=["Apoptosis"],
        metric="NOM p-val",
        threshold=0.01,
        threshold_operator="lt",
        min_num_degs=3,
        permutation_num=50,
        min_size=5,
        max_size=60,
    )


def test_enrichment_one_analysis_per_config(deps):
    config = make_config()
    config.analysis.enrichment_analysis = [
        enrich_cfg("deg", "A"),
        enrich_cfg("hdeg", "B"),
    ]
    make_workflow(config).perform_enrichment_analysis()

    assert deps.EnrichmentAnalysis.call_args_list == [
        mock.call(gene_sets="A", organism="Mouse"),
        mock.call(gene_sets="B", organism="Mouse"),
    ]
    runs = deps.EnrichmentAnalysis.return_value.run_directory.call_args_list
    assert [(c.kwargs["input_dir"], c.kwargs["outdir"]) for c in runs] == [
        ("/out/differential_analysis", "/out/enrichment_analysis/safe-a"),
        ("/out/hierarchical_differential_analysis", "/out/enrichment_analysis/safe-b"),
    ]


def test_enrichment_nothing_configured(deps, makedirs):
    make_workflow().perform_enrichment_analysis()
    deps.EnrichmentAnalysis.assert_not_called()
    makedirs.assert_not_called()


def test_enrichment_errors_propagate(deps):
    deps.EnrichmentAnalysis.return_value.run_directory.side_effect = ValueError(
        "No DEG/HDEG tables"
    )
    config = make_config()
    config.analysis.enrichment_analysis = [enrich_cfg()]
    with pytest.raises(ValueError, match="No DEG/HDEG tables"):
        make_workflow(config).perform_enrichment_analysis()
