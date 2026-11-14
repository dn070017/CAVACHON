"""Unit tests for cavachon.tools.enrichment_analysis.

gseapy, InteractiveVisualization and DataFrameUtils are MagicMocks
(conftest ``ea_deps``; ``filter_by_threshold`` compares like the real one).
Path.mkdir, tempfile.TemporaryDirectory, shutil.copytree and pd.read_csv
are recorders (conftest ``fs``); Path.is_dir / glob / rglob are patched
per test. Nothing touches the network or the disk.
"""

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from cavachon.tools.enrichment_analysis import EnrichmentAnalysis, _safe_filename
from test.tools.visualization_enrichment.conftest import (
    STAGING_DIR,
    make_prerank_result,
)

LIBRARY = {"SET_A": ["g1", "g2"], "SET_B": ["g3"]}


def mkdir_paths(fs):
    return [c[0] for c in fs.mkdir.calls]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


class TestInitAndLibrary:
    def test_defaults(self):
        analysis = EnrichmentAnalysis()
        assert analysis.gene_sets == "KEGG_2019_Mouse"
        assert analysis.organism == "Mouse"

    def test_get_library_without_organism(self, ea_deps):
        ea_deps.gseapy.get_library.return_value = LIBRARY
        assert EnrichmentAnalysis.get_library("LIB") is LIBRARY
        ea_deps.gseapy.get_library.assert_called_once_with(name="LIB")

    def test_get_library_with_organism(self, ea_deps):
        EnrichmentAnalysis.get_library("LIB", organism="Human")
        ea_deps.gseapy.get_library.assert_called_once_with(name="LIB", organism="Human")


@pytest.mark.parametrize(
    "value,expected",
    [
        ("KEGG pathway: A/B", "KEGG_pathway_A_B"),
        ("__x--y__", "x--y"),
        ("a!!!b", "a_b"),
        ("!!!", ""),
        ("plain", "plain"),
    ],
)
def test_safe_filename(value, expected):
    assert _safe_filename(value) == expected


def test_output_name():
    name = EnrichmentAnalysis._output_name(Path("/in"), Path("/in/sub dir/deg.v1.tsv"))
    assert name == "sub_dir__deg_v1"


# ---------------------------------------------------------------------------
# _run_prerank
# ---------------------------------------------------------------------------


def test_run_prerank_calls_gseapy(ea_deps, fs):
    rnk = pd.Series([1.0, -1.0], index=["g1", "g2"])
    out = EnrichmentAnalysis()._run_prerank(
        rnk, LIBRARY, "/out/dir", permutation_num=10, seed=1
    )
    assert out is ea_deps.gseapy.prerank.return_value
    ea_deps.gseapy.prerank.assert_called_once_with(
        rnk=rnk, gene_sets=LIBRARY, outdir="/out/dir", permutation_num=10, seed=1
    )
    assert mkdir_paths(fs) == [Path("/out/dir")]
    assert fs.mkdir.calls[0][2] == {"parents": True, "exist_ok": True}


# ---------------------------------------------------------------------------
# _visualize / visualize
# ---------------------------------------------------------------------------


class TestVisualize:
    def test_default_terms_from_threshold(self, ea_deps, fs):
        result = make_prerank_result()
        figures = EnrichmentAnalysis().visualize(result, "/viz")

        iv = ea_deps.InteractiveVisualization
        iv.prerank_ringplot.assert_called_once_with(
            result,
            metric="FDR q-val",
            threshold=0.05,
            filename="/viz/enrichment_ringplot.html",
        )
        assert [c.kwargs for c in iv.prerank_enrichment_score.call_args_list] == [
            {"term": "T1", "filename": "/viz/enrichment_score_T1.html"},
            {"term": "T2", "filename": "/viz/enrichment_score_T2.html"},
        ]
        assert figures == {
            "ringplot": "RINGPLOT",
            "enrichment_score:T1": "ES:T1",
            "enrichment_score:T2": "ES:T2",
        }
        assert mkdir_paths(fs) == [Path("/viz")]

    def test_threshold_operator_forwarded(self, ea_deps):
        result = make_prerank_result()
        figures = EnrichmentAnalysis().visualize(
            result, "/viz", metric="NES", threshold=1.0, threshold_operator="geq"
        )
        call = ea_deps.DataFrameUtils.filter_by_threshold.call_args
        pd.testing.assert_series_equal(call.args[0], result.res2d["NES"])
        assert call.args[1:] == (1.0, "geq")
        assert list(figures) == ["ringplot", "enrichment_score:T1"]

    def test_explicit_terms_and_safe_filenames(self, ea_deps):
        result = make_prerank_result(results={"A/B c": {}, "T1": {}})
        figures = EnrichmentAnalysis().visualize(result, Path("/viz"), terms=["A/B c"])
        ea_deps.DataFrameUtils.filter_by_threshold.assert_not_called()
        ea_deps.InteractiveVisualization.prerank_enrichment_score.assert_called_once_with(
            result, term="A/B c", filename="/viz/enrichment_score_A_B_c.html"
        )
        assert list(figures) == ["ringplot", "enrichment_score:A/B c"]

    def test_index_used_when_no_term_column(self, ea_deps):
        res2d = pd.DataFrame({"FDR q-val": [0.01, 0.9]}, index=["X", "Y"])
        result = make_prerank_result(res2d=res2d, results={"X": {}, "Y": {}})
        figures = EnrichmentAnalysis().visualize(result, "/viz")
        assert list(figures) == ["ringplot", "enrichment_score:X"]

    def test_unknown_term_raises(self):
        result = make_prerank_result()
        with pytest.raises(ValueError, match="not present in enrichment results: nope"):
            EnrichmentAnalysis().visualize(result, "/viz", terms=["nope"])

    def test_missing_result_table_raises(self):
        result = make_prerank_result()
        result.res2d = None
        with pytest.raises(ValueError, match="result table"):
            EnrichmentAnalysis().visualize(result, "/viz")

    def test_missing_term_results_raises(self):
        result = make_prerank_result()
        result.results = None
        with pytest.raises(ValueError, match="term results"):
            EnrichmentAnalysis().visualize(result, "/viz")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@pytest.fixture
def degs():
    # negative values pass the default "FDR q-val <= 0.05" threshold (see the
    # xfail test below about this threshold being applied to the DEG table)
    return pd.DataFrame(
        {"K(A>B|Z)": [-1.0, -2.0, np.nan, -3.0], "other": [1, 2, 3, 4]},
        index=["g1", "g2", "g3", "g4"],
    )


@pytest.fixture
def prerank(ea_deps):
    result = make_prerank_result()
    ea_deps.gseapy.prerank.return_value = result
    ea_deps.gseapy.get_library.return_value = LIBRARY
    return result


class TestRun:
    def test_missing_column_raises(self, degs, ea_deps):
        with pytest.raises(ValueError, match="Column is not present in DEG table: K"):
            EnrichmentAnalysis().run(degs, "K", "/out")
        ea_deps.gseapy.prerank.assert_not_called()

    def test_no_ranking_values_raises(self, ea_deps):
        table = pd.DataFrame({"K": [np.nan, np.nan]}, index=["a", "b"])
        with pytest.raises(ValueError, match="no ranking values"):
            EnrichmentAnalysis().run(table, "K", "/out")
        ea_deps.gseapy.prerank.assert_not_called()

    def test_too_few_degs_skips(self, degs, ea_deps, fs, capsys):
        out = EnrichmentAnalysis().run(degs, "K(A>B|Z)", "/out", min_num_degs=4)
        assert out is None
        assert "only 3 DEGs" in capsys.readouterr().out
        ea_deps.gseapy.prerank.assert_not_called()
        ea_deps.gseapy.get_library.assert_not_called()
        assert fs.mkdir.calls == []

    def test_full_run(self, degs, ea_deps, fs, prerank):
        analysis = EnrichmentAnalysis(gene_sets="LIB", organism="Human")
        out = analysis.run(degs, "K(A>B|Z)", "/out", min_num_degs=3, permutation_num=7)

        assert out is prerank
        ea_deps.gseapy.get_library.assert_called_once_with(name="LIB", organism="Human")
        call = ea_deps.gseapy.prerank.call_args
        pd.testing.assert_series_equal(call.kwargs["rnk"], degs["K(A>B|Z)"].dropna())
        assert call.kwargs["gene_sets"] is LIBRARY
        assert call.kwargs["outdir"] == STAGING_DIR
        assert call.kwargs["permutation_num"] == 7
        assert fs.tempdirs[0].kwargs == {"prefix": "enrichment-analysis-"}

        assert mkdir_paths(fs) == [Path(STAGING_DIR), Path("/out"), Path("/out")]
        fs.copytree.assert_called_once_with(
            STAGING_DIR, Path("/out"), dirs_exist_ok=True
        )
        iv = ea_deps.InteractiveVisualization
        assert iv.prerank_ringplot.call_args.kwargs["filename"] == (
            "/out/enrichment_ringplot.html"
        )
        assert iv.prerank_enrichment_score.call_count == 2

    def test_mapping_gene_sets_skip_library_download(self, degs, ea_deps, prerank):
        EnrichmentAnalysis(gene_sets=LIBRARY).run(degs, "K(A>B|Z)", "/out")
        ea_deps.gseapy.get_library.assert_not_called()
        assert ea_deps.gseapy.prerank.call_args.kwargs["gene_sets"] is LIBRARY

    def test_no_significant_terms_skips_output(
        self, degs, ea_deps, fs, prerank, capsys
    ):
        prerank.res2d["FDR q-val"] = [0.5, 0.6, 0.7]
        out = EnrichmentAnalysis().run(degs, "K(A>B|Z)", "/out")
        assert out is prerank
        assert (
            "No enrichment pathways passed FDR q-val <= 0.05" in capsys.readouterr().out
        )
        fs.copytree.assert_not_called()
        assert Path("/out") not in mkdir_paths(fs)
        ea_deps.InteractiveVisualization.prerank_ringplot.assert_not_called()

    def test_missing_metric_raises(self, degs, prerank):
        with pytest.raises(ValueError, match="does not contain metric: nope"):
            EnrichmentAnalysis().run(degs, "K(A>B|Z)", "/out", metric="nope")

    def test_missing_result_table_raises(self, degs, prerank):
        prerank.res2d = None
        with pytest.raises(ValueError, match="result table"):
            EnrichmentAnalysis().run(degs, "K(A>B|Z)", "/out")

    def test_terms_forwarded(self, degs, ea_deps, prerank):
        EnrichmentAnalysis().run(degs, "K(A>B|Z)", "/out", terms=["T3"])
        ea_deps.InteractiveVisualization.prerank_enrichment_score.assert_called_once_with(
            prerank, term="T3", filename="/out/enrichment_score_T3.html"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: run() applies the term-metric threshold/operator (e.g. "
        "FDR q-val <= 0.05) to the DEG ranking column, dropping ranked genes",
    )
    def test_full_ranking_passed_to_prerank(self, ea_deps, prerank):
        table = pd.DataFrame({"K(A>B|Z)": [5.0, 1.0, -2.0]}, index=["g1", "g2", "g3"])
        EnrichmentAnalysis(gene_sets=LIBRARY).run(table, "K(A>B|Z)", "/out")
        rnk = ea_deps.gseapy.prerank.call_args.kwargs["rnk"]
        assert list(rnk.index) == ["g1", "g2", "g3"]


# ---------------------------------------------------------------------------
# _read_table / find_deg_tables / run_directory
# ---------------------------------------------------------------------------


class TestReadTable:
    def test_reads_and_coerces(self, fs):
        fs.read_csv.return_value = pd.DataFrame({"K": ["1.5", "x", "2"]})
        table = EnrichmentAnalysis._read_table("/f.tsv", "K")
        fs.read_csv.assert_called_once_with(
            "/f.tsv", sep=None, engine="python", index_col=0
        )
        np.testing.assert_allclose(table["K"], [1.5, np.nan, 2.0])

    def test_missing_column(self, fs):
        fs.read_csv.return_value = pd.DataFrame({"other": [1.0]})
        assert EnrichmentAnalysis._read_table("/f.tsv", "K") is None

    def test_no_numeric_values(self, fs):
        fs.read_csv.return_value = pd.DataFrame({"K": ["a", "b"]})
        assert EnrichmentAnalysis._read_table("/f.tsv", "K") is None


@pytest.fixture
def fake_dir(monkeypatch, fs):
    """/in holds a.tsv (usable), b.csv (no column) and sub/c.txt (usable)."""
    files = {
        "*.tsv": [Path("/in/a.tsv")],
        "*.csv": [Path("/in/b.csv")],
        "*.txt": [Path("/in/sub/c.txt")],
    }
    tables = {
        "/in/a.tsv": pd.DataFrame({"K(A>B|Z)": [1.0]}, index=["g"]),
        "/in/b.csv": pd.DataFrame({"x": [1.0]}),
        "/in/sub/c.txt": pd.DataFrame({"K(A>B|Z)": [2.0]}, index=["g"]),
    }
    is_dir = mock.MagicMock(return_value=True)
    rglob = mock.MagicMock(side_effect=lambda pattern: iter(files[pattern]))
    glob = mock.MagicMock(
        side_effect=lambda pattern: iter(
            p for p in files[pattern] if p.parent == Path("/in")
        )
    )
    monkeypatch.setattr(Path, "is_dir", lambda self: is_dir(self))
    monkeypatch.setattr(Path, "rglob", lambda self, p: rglob(p))
    monkeypatch.setattr(Path, "glob", lambda self, p: glob(p))
    fs.read_csv.side_effect = lambda f, **kw: tables[str(f)].copy()
    return mock.Mock(is_dir=is_dir, rglob=rglob, glob=glob)


class TestFindDegTables:
    def test_not_a_directory(self, fake_dir):
        fake_dir.is_dir.return_value = False
        with pytest.raises(ValueError, match="does not exist"):
            list(EnrichmentAnalysis.find_deg_tables("/in"))

    def test_recursive(self, fake_dir):
        found = list(EnrichmentAnalysis.find_deg_tables("/in"))
        assert [f for f, _ in found] == [Path("/in/a.tsv"), Path("/in/sub/c.txt")]
        assert [c.args[0] for c in fake_dir.rglob.call_args_list] == [
            "*.tsv",
            "*.csv",
            "*.txt",
        ]
        fake_dir.glob.assert_not_called()

    def test_non_recursive(self, fake_dir):
        found = list(EnrichmentAnalysis.find_deg_tables("/in", recursive=False))
        assert [f for f, _ in found] == [Path("/in/a.tsv")]
        fake_dir.rglob.assert_not_called()

    def test_custom_column(self, fake_dir):
        found = list(EnrichmentAnalysis.find_deg_tables("/in", column="x"))
        assert [f for f, _ in found] == [Path("/in/b.csv")]


class TestRunDirectory:
    def test_runs_each_table(self, fake_dir, fs, monkeypatch):
        analysis = EnrichmentAnalysis()
        run = mock.MagicMock(side_effect=["R1", None])
        monkeypatch.setattr(analysis, "run", run)

        out = analysis.run_directory(
            "/in", terms=["T"], threshold=0.1, min_num_degs=2, seed=3
        )

        assert out == {Path("/in/a.tsv"): "R1", Path("/in/sub/c.txt"): None}
        assert mkdir_paths(fs) == [Path("/in/enrichment_analysis")]
        outdirs = [c.kwargs["outdir"] for c in run.call_args_list]
        assert outdirs == [
            Path("/in/enrichment_analysis/a"),
            Path("/in/enrichment_analysis/sub__c"),
        ]
        kwargs = run.call_args_list[0].kwargs
        assert kwargs["column"] == "K(A>B|Z)"
        assert kwargs["terms"] == ["T"]
        assert kwargs["threshold"] == 0.1
        assert kwargs["threshold_operator"] == "leq"
        assert kwargs["metric"] == "FDR q-val"
        assert kwargs["min_num_degs"] == 2
        assert kwargs["seed"] == 3
        pd.testing.assert_frame_equal(
            kwargs["deg_table"], pd.DataFrame({"K(A>B|Z)": [1.0]}, index=["g"])
        )

    def test_explicit_outdir(self, fake_dir, fs, monkeypatch):
        analysis = EnrichmentAnalysis()
        run = mock.MagicMock()
        monkeypatch.setattr(analysis, "run", run)
        analysis.run_directory("/in", outdir="/res", recursive=False)
        assert mkdir_paths(fs) == [Path("/res")]
        assert [c.kwargs["outdir"] for c in run.call_args_list] == [Path("/res/a")]

    def test_no_tables_raises(self, fake_dir, fs, monkeypatch):
        analysis = EnrichmentAnalysis()
        run = mock.MagicMock()
        monkeypatch.setattr(analysis, "run", run)
        with pytest.raises(
            ValueError, match="No DEG/HDEG tables with ranking column 'zz'"
        ):
            analysis.run_directory("/in", column="zz")
        run.assert_not_called()
        assert fs.mkdir.calls == []
