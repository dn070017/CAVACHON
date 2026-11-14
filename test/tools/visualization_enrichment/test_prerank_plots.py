"""Unit tests for InteractiveVisualization.prerank_ringplot and
prerank_enrichment_score.

The gseapy Prerank result is a plain stand-in (``make_prerank_result``)
whose ``res2d`` / ``results`` / ``ranking`` follow gseapy 1.1 formats:
``Gene %`` strings like ``"12.35%"`` and per-term dicts with lower-case
``nes``/``pval``/``fdr``/``RES``/``hits`` keys.
"""

import numpy as np
import pandas as pd
import pytest

from cavachon.tools.interactive_visualization import InteractiveVisualization
from test.tools.visualization_enrichment.conftest import make_prerank_result


class TestRingplot:
    def test_filters_reverses_and_colors(self):
        result = make_prerank_result()
        fig = InteractiveVisualization.prerank_ringplot(result)

        main, ring = fig.data
        # T3 (q=0.5) removed; order reversed
        assert list(main.y) == ["T2", "T1"]
        np.testing.assert_allclose(main.x, [-1.5, 2.0])
        expected_color = np.log(1 / (np.array([0.04, 0.01], dtype=np.float32) + 1e-7))
        np.testing.assert_allclose(main.marker.color, expected_color, rtol=1e-6)
        np.testing.assert_allclose(main.marker.size, [25.0, 50.0])
        assert main.marker.sizeref == pytest.approx(50.0 / (21 * 50 / 100) ** 2)
        assert fig.layout.coloraxis.colorbar.title.text == "log(1/FDR q-val)"

        assert list(ring.y) == ["T2", "T1"]
        assert ring.marker.size == 30
        assert ring.marker.color == "rgba(0, 0, 0, 0)"
        assert ring.showlegend is False
        for trace in fig.data:
            assert trace.marker.opacity == 0.7
            assert trace.marker.line.color == "DarkSlateGrey"

    def test_layout(self):
        fig = InteractiveVisualization.prerank_ringplot(
            make_prerank_result(), title="RP"
        )
        assert fig.layout.width == 800
        assert fig.layout.height == 35 * 2 + 150
        assert fig.layout.title.text == "RP"

    def test_metric_and_threshold(self):
        result = make_prerank_result()
        result.res2d["NOM p-val"] = [0.3, 0.001, 0.2]
        fig = InteractiveVisualization.prerank_ringplot(
            result, metric="NOM p-val", threshold=0.25
        )
        assert list(fig.data[0].y) == ["T3", "T2"]
        assert fig.layout.coloraxis.colorbar.title.text == "log(1/NOM p-val)"

    def test_does_not_mutate_result_table(self):
        result = make_prerank_result()
        before = result.res2d.copy()
        InteractiveVisualization.prerank_ringplot(result)
        pd.testing.assert_frame_equal(result.res2d, before)

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: prerank_ringplot parses 'Gene %' with x[:-2], dropping the "
        "last digit of gseapy's '12.35%' format",
    )
    def test_gene_percent_parsed_exactly(self):
        res2d = pd.DataFrame(
            {"Term": ["T1"], "NES": [1.0], "FDR q-val": [0.01], "Gene %": ["12.35%"]}
        )
        fig = InteractiveVisualization.prerank_ringplot(make_prerank_result(res2d))
        np.testing.assert_allclose(fig.data[0].marker.size, [12.35])

    @pytest.mark.parametrize(
        "filename,html,image", [(None, 0, 0), ("r.html", 1, 0), ("r.png", 0, 1)]
    )
    def test_saving(self, figure_io, filename, html, image):
        InteractiveVisualization.prerank_ringplot(
            make_prerank_result(), filename=filename
        )
        assert [c[1] for c in figure_io.write_html.calls] == [(filename,)] * html
        assert [c[1] for c in figure_io.write_image.calls] == [(filename,)] * image
        assert len(figure_io.show.calls) == 1


GENES = ["g0", "g1", "g2", "g3", "g4"]


@pytest.fixture
def es_result():
    ranking = pd.Series([2.0, 1.0, 0.0, -1.0, -2.0], index=GENES)
    results = {
        "T1": {
            "RES": [0.1, 0.3, 0.2, 0.4, 0.0],
            "hits": [0, 3],
            "nes": 1.23456,
            "pval": 0.001234,
            "fdr": 0.04,
        }
    }
    return make_prerank_result(results=results, ranking=ranking)


class TestEnrichmentScore:
    def test_traces(self, es_result):
        fig = InteractiveVisualization.prerank_enrichment_score(es_result, "T1")
        es_pos, es_neg, hits, bar, rank_pos, rank_neg, zero = fig.data

        # zero score index = argmin |ranking| = 2
        np.testing.assert_array_equal(es_pos.x, [0, 1])
        np.testing.assert_allclose(es_pos.y, [0.1, 0.3])
        assert es_pos.marker.color == "salmon"
        np.testing.assert_array_equal(es_neg.x, [2, 3, 4])
        np.testing.assert_allclose(es_neg.y, [0.2, 0.4, 0.0])
        assert es_neg.marker.color == "royalblue"

        np.testing.assert_array_equal(hits.x, [0, 3])
        np.testing.assert_array_equal(hits.y, [0, 0])
        assert list(hits.text) == ["g0", "g3"]
        assert list(hits.marker.color) == ["salmon", "royalblue"]

        np.testing.assert_array_equal(bar.x, [0, 1, 2, 3, 4])
        assert bar.marker.colorscale[1][0] == pytest.approx(2 / 5)

        np.testing.assert_allclose(rank_pos.y, [2.0, 1.0])
        np.testing.assert_allclose(rank_neg.y, [0.0, -1.0, -2.0])
        assert rank_pos.fill == "tozeroy"

        assert list(zero.x) == [2]
        assert list(zero.text) == ["Zero score at 2"]

    def test_layout(self, es_result):
        fig = InteractiveVisualization.prerank_enrichment_score(es_result, "T1")
        assert (
            fig.layout.title.text == "T1<br>NES: 1.235, Pval: 1.234e-03, FDR: 4.000e-02"
        )
        assert fig.layout.yaxis.title.text == "Enrichment Score"
        assert fig.layout.xaxis4.title.text == "Rank in Ordered Dataset"
        assert fig.layout.yaxis4.title.text == "Rank Metric"
        assert tuple(fig.layout.xaxis2.range) == (0, 5)
        assert tuple(fig.layout.xaxis4.range) == (0, 5)
        assert fig.layout.width == 700
        assert fig.layout.height == 700

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: prerank_enrichment_score passes the full Target column as "
        "hover text to the split (index >= zero score) traces, so hover labels "
        "are offset",
    )
    def test_hover_text_matches_split_traces(self, es_result):
        fig = InteractiveVisualization.prerank_enrichment_score(es_result, "T1")
        es_neg = fig.data[1]
        assert list(es_neg.text) == ["g2", "g3", "g4"]

    def test_unknown_term_raises(self, es_result):
        with pytest.raises(KeyError):
            InteractiveVisualization.prerank_enrichment_score(es_result, "missing")

    @pytest.mark.parametrize(
        "filename,html,image", [(None, 0, 0), ("e.html", 1, 0), ("e.png", 0, 1)]
    )
    def test_saving(self, es_result, figure_io, filename, html, image):
        InteractiveVisualization.prerank_enrichment_score(
            es_result, "T1", filename=filename
        )
        assert [c[1] for c in figure_io.write_html.calls] == [(filename,)] * html
        assert [c[1] for c in figure_io.write_image.calls] == [(filename,)] * image
        assert len(figure_io.show.calls) == 1
