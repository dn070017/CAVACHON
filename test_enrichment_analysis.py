from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from gseapy.gsea import Prerank

from cavachon.config import AnalysisConfig, AnalysisEnrichmentConfig, ApplicationConfig
from cavachon.tools import EnrichmentAnalysis
from cavachon.workflow.workflow import Workflow


def test_run_uses_deg_column_and_visualizes_result(tmp_path, monkeypatch):
    calls = {}
    expected_result = SimpleNamespace(
        res2d=pd.DataFrame({"FDR q-val": [0.01]}), results={}
    )

    def fake_prerank(**kwargs):
        calls.update(kwargs)
        return expected_result

    monkeypatch.setattr(
        "cavachon.tools.enrichment_analysis.gseapy.prerank", fake_prerank
    )
    monkeypatch.setattr(
        "cavachon.tools.enrichment_analysis.EnrichmentAnalysis._visualize",
        lambda *args, **kwargs: calls.update(visualize=kwargs),
    )

    deg_table = pd.DataFrame(
        {"K(A>B|Z)": [2.0, -1.0]}, index=pd.Index(["GeneA", "GeneB"])
    )
    output_dir = tmp_path / "enrichment"
    result = EnrichmentAnalysis(gene_sets={"pathway": ["GeneA"]}).run(
        deg_table=deg_table,
        column="K(A>B|Z)",
        outdir=output_dir,
        permutation_num=10,
    )

    assert result is expected_result
    assert output_dir.is_dir()
    assert calls["rnk"].equals(deg_table["K(A>B|Z)"])
    assert calls["gene_sets"] == {"pathway": ["GeneA"]}
    assert calls["outdir"] != str(output_dir)
    assert calls["permutation_num"] == 10
    assert calls["visualize"]["outdir"] == output_dir


def test_visualize_writes_ringplot_and_requested_term(tmp_path, monkeypatch):
    result = SimpleNamespace(
        res2d=pd.DataFrame(
            {"Term": ["Pathway / A"], "FDR q-val": [0.01]}, index=pd.Index([0])
        ),
        results={"Pathway / A": object()},
    )
    calls = []

    def fake_ringplot(prerank_result, **kwargs):
        calls.append(("ringplot", prerank_result, kwargs))
        return "ringplot"

    def fake_score(prerank_result, **kwargs):
        calls.append(("score", prerank_result, kwargs))
        return "score"

    monkeypatch.setattr(
        "cavachon.tools.enrichment_analysis.InteractiveVisualization.prerank_ringplot",
        fake_ringplot,
    )
    monkeypatch.setattr(
        "cavachon.tools.enrichment_analysis.InteractiveVisualization.prerank_enrichment_score",
        fake_score,
    )

    figures = EnrichmentAnalysis(gene_sets={}).visualize(
        prerank_result=cast(Prerank, cast(object, result)),
        outdir=tmp_path / "figures",
        terms=["Pathway / A"],
    )

    assert figures == {"ringplot": "ringplot", "enrichment_score:Pathway / A": "score"}
    assert calls[0][2]["filename"].endswith("enrichment_ringplot.html")
    assert calls[1][2]["filename"].endswith("enrichment_score_Pathway_A.html")


def test_visualize_uses_term_column_for_automatic_selection(tmp_path, monkeypatch):
    result = SimpleNamespace(
        res2d=pd.DataFrame(
            {
                "Term": ["Pathway / A", "Pathway B"],
                "FDR q-val": [0.01, 0.2],
            },
            index=pd.Index([0, 1]),
        ),
        results={"Pathway / A": object(), "Pathway B": object()},
    )
    terms = []

    monkeypatch.setattr(
        "cavachon.tools.enrichment_analysis.InteractiveVisualization.prerank_ringplot",
        lambda *args, **kwargs: "ringplot",
    )
    monkeypatch.setattr(
        "cavachon.tools.enrichment_analysis.InteractiveVisualization.prerank_enrichment_score",
        lambda prerank_result, **kwargs: terms.append(kwargs["term"]) or "score",
    )

    EnrichmentAnalysis(gene_sets={}).visualize(
        cast(Prerank, cast(object, result)), tmp_path
    )

    assert terms == ["Pathway / A"]


def test_visualize_rejects_unknown_term(tmp_path, monkeypatch):
    result = SimpleNamespace(
        res2d=pd.DataFrame(
            {"Term": ["Pathway"], "FDR q-val": [0.01]}, index=pd.Index([0])
        ),
        results={"Pathway": object()},
    )
    monkeypatch.setattr(
        "cavachon.tools.enrichment_analysis.InteractiveVisualization.prerank_ringplot",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match="not present"):
        EnrichmentAnalysis(gene_sets={}).visualize(
            cast(Prerank, cast(object, result)), tmp_path, terms=["Missing"]
        )


def test_find_deg_tables_selects_deg_and_skips_other_reports(tmp_path):
    deg_dir = tmp_path / "differential_analysis"
    deg_dir.mkdir()
    pd.DataFrame(
        {"K(A>B|Z)": [2.0, -1.0]}, index=pd.Index(["GeneA", "GeneB"])
    ).to_csv(deg_dir / "clusters.tsv", sep="\t", index_label="genes")
    pd.DataFrame({"Term": ["Pathway"]}).to_csv(
        deg_dir / "gseapy.report.csv", index=False
    )

    tables = list(EnrichmentAnalysis.find_deg_tables(tmp_path))

    assert [filename.name for filename, _ in tables] == ["clusters.tsv"]
    assert tables[0][1]["K(A>B|Z)"].tolist() == [2.0, -1.0]


def test_run_directory_processes_nested_tables_separately(tmp_path, monkeypatch):
    input_dir = tmp_path / "analysis"
    nested_dir = input_dir / "hierarchical_differential_analysis"
    nested_dir.mkdir(parents=True)
    first = input_dir / "deg.tsv"
    second = nested_dir / "hdeg.tsv"
    for filename in (first, second):
        pd.DataFrame({"K(A>B|Z)": [1.0]}, index=pd.Index(["GeneA"])).to_csv(
            filename, sep="\t", index_label="genes"
        )

    calls = []

    def fake_run(self, **kwargs):
        calls.append(kwargs)
        return cast(Prerank, cast(object, object()))

    monkeypatch.setattr(EnrichmentAnalysis, "run", fake_run)

    results = EnrichmentAnalysis(gene_sets={}).run_directory(
        input_dir=input_dir, outdir=tmp_path / "enrichment", permutation_num=10
    )

    assert set(results) == {first, second}
    assert {call["outdir"].name for call in calls} == {
        "deg",
        "hierarchical_differential_analysis__hdeg",
    }
    assert all(call["permutation_num"] == 10 for call in calls)


def test_run_directory_rejects_directory_without_usable_tables(tmp_path):
    output_dir = tmp_path / "enrichment"

    with pytest.raises(ValueError, match="No DEG/HDEG tables"):
        EnrichmentAnalysis(gene_sets={}).run_directory(tmp_path, outdir=output_dir)

    assert not output_dir.exists()


def test_run_skips_output_when_no_pathways_pass_threshold(tmp_path, monkeypatch, capsys):
    result = SimpleNamespace(
        res2d=pd.DataFrame({"FDR q-val": [0.2]}), results={"Pathway": object()}
    )

    monkeypatch.setattr(
        "cavachon.tools.enrichment_analysis.gseapy.prerank",
        lambda **kwargs: result,
    )
    monkeypatch.setattr(
        "cavachon.tools.enrichment_analysis.EnrichmentAnalysis._visualize",
        lambda *args, **kwargs: pytest.fail("visualization should be skipped"),
    )

    output_dir = tmp_path / "enrichment"
    returned = EnrichmentAnalysis(gene_sets={}).run(
        deg_table=pd.DataFrame(
            {"K(A>B|Z)": [1.0]}, index=pd.Index(["GeneA"])
        ),
        column="K(A>B|Z)",
        outdir=output_dir,
    )

    assert returned is result
    assert not output_dir.exists()
    assert "No enrichment pathways passed FDR q-val <= 0.05" in capsys.readouterr().out


def test_find_deg_tables_rejects_missing_directory(tmp_path):
    with pytest.raises(ValueError, match="Input directory does not exist"):
        list(EnrichmentAnalysis.find_deg_tables(tmp_path / "missing"))


def test_enrichment_config_requires_valid_analysis_type():
    config = AnalysisEnrichmentConfig(analysis_type="deg")

    assert config.analysis_type == "deg"
    with pytest.raises(ValueError):
        AnalysisEnrichmentConfig.model_validate({"analysis_type": "invalid"})


def test_analysis_config_exposes_enrichment_entries():
    config = AnalysisConfig(
        enrichment_analysis=[AnalysisEnrichmentConfig(analysis_type="hdeg")]
    )

    assert config.enrichment_analysis[0].analysis_type == "hdeg"


def test_enrichment_config_accepts_existing_deg_directory(tmp_path):
    (tmp_path / "differential_analysis").mkdir()
    analysis = AnalysisConfig(
        enrichment_analysis=[AnalysisEnrichmentConfig(analysis_type="deg")]
    )

    ApplicationConfig._validate_enrichment_sources(analysis, str(tmp_path))


def test_enrichment_config_accepts_existing_hdeg_directory(tmp_path):
    (tmp_path / "hierarchical_differential_analysis").mkdir()
    analysis = AnalysisConfig(
        enrichment_analysis=[AnalysisEnrichmentConfig(analysis_type="hdeg")]
    )

    ApplicationConfig._validate_enrichment_sources(analysis, str(tmp_path))


def test_enrichment_config_rejects_missing_source_directory(tmp_path):
    analysis = AnalysisConfig(
        enrichment_analysis=[AnalysisEnrichmentConfig(analysis_type="deg")]
    )

    with pytest.raises(ValueError, match="existing directory"):
        ApplicationConfig._validate_enrichment_sources(analysis, str(tmp_path))


def test_workflow_enrichment_uses_selected_output_directory(tmp_path, monkeypatch):
    calls = []

    class FakeEnrichmentAnalysis:
        def __init__(self, gene_sets, organism):
            calls.append(("init", gene_sets, organism))

        def run_directory(self, **kwargs):
            calls.append(("run_directory", kwargs))

    monkeypatch.setattr(
        "cavachon.workflow.workflow.EnrichmentAnalysis", FakeEnrichmentAnalysis
    )
    workflow = object.__new__(Workflow)
    workflow.config = cast(
        ApplicationConfig,
        cast(
            object,
            SimpleNamespace(
                io=SimpleNamespace(outdir=str(tmp_path)),
                analysis=SimpleNamespace(
                    enrichment_analysis=[
                        AnalysisEnrichmentConfig(
                            analysis_type="hdeg",
                            gene_sets="KEGG_2019_Mouse",
                            permutation_num=25,
                        )
                    ]
                ),
            ),
        ),
    )

    workflow.perform_enrichment_analysis()

    assert calls[0] == ("init", "KEGG_2019_Mouse", "Mouse")
    assert calls[1][0] == "run_directory"
    assert calls[1][1]["input_dir"] == str(
        tmp_path / "hierarchical_differential_analysis"
    )
    assert calls[1][1]["outdir"] == str(
        tmp_path / "enrichment_analysis" / "kegg_2019_mouse"
    )
    assert calls[1][1]["permutation_num"] == 25
