from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from gseapy.gsea import Prerank

from cavachon.tools import EnrichmentAnalysis


def test_run_uses_deg_column_and_visualizes_result(tmp_path, monkeypatch):
    calls = {}
    expected_result = object()

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
    assert calls["outdir"] == str(output_dir)
    assert calls["permutation_num"] == 10
    assert calls["visualize"]["outdir"] == output_dir


def test_visualize_writes_ringplot_and_requested_term(tmp_path, monkeypatch):
    result = SimpleNamespace(
        res2d=pd.DataFrame(
            {"FDR q-val": [0.01]}, index=pd.Index(["Pathway / A"])
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


def test_visualize_rejects_unknown_term(tmp_path, monkeypatch):
    result = SimpleNamespace(
        res2d=pd.DataFrame({"FDR q-val": [0.01]}, index=pd.Index(["Pathway"])),
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
