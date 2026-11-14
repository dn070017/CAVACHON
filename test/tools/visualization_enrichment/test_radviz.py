"""Unit tests for InteractiveVisualization.differential_attribution_radviz."""

import numpy as np
import pandas as pd
import pytest

from cavachon.tools.interactive_visualization import InteractiveVisualization

WRT = ["c0", "c1", "c2", "c3"]


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "abs_contrib_c0": [1.0, 0.0, 1.0, 0.0, -2.0],
            "abs_contrib_c1": [0.0, 1.0, 0.0, 0.0, 0.0],
            "abs_contrib_c2": [0.0, 0.0, 1.0, 0.0, 0.0],
            "abs_contrib_c3": [0.0, 1.0, 0.0, 0.0, 1.0],
            "LogFC": [2.0, -1.0, 0.0, 0.001, -0.5],
        },
        index=["g0", "g1", "g2", "g3", "g4"],
    )


def points(fig):
    """Map gene -> (x, y, direction, marker size) over the px traces."""
    out = {}
    for trace in fig.data:
        if trace.hovertext is None:
            continue
        for i, gene in enumerate(trace.hovertext):
            out[gene] = (trace.x[i], trace.y[i], trace.name, trace.marker.size[i])
    return out


class TestValidation:
    def test_empty_raises(self, data):
        with pytest.raises(ValueError, match="empty"):
            InteractiveVisualization.differential_attribution_radviz(data.iloc[:0], WRT)

    def test_missing_columns_raise(self, data):
        with pytest.raises(ValueError, match="abs_contrib_zz"):
            InteractiveVisualization.differential_attribution_radviz(data, ["c0", "zz"])


class TestProjection:
    def test_positions(self, data):
        fig = InteractiveVisualization.differential_attribution_radviz(data, WRT)
        pts = points(fig)
        # vertices at angles 0, 90, 180, 270 degrees
        assert pts["g0"][:2] == pytest.approx((1.0, 0.0))
        assert pts["g1"][:2] == pytest.approx((0.0, 0.0), abs=1e-12)  # c1 + c3
        assert pts["g2"][:2] == pytest.approx((0.0, 0.0), abs=1e-12)  # c0 + c2
        # all-zero weights -> origin
        assert pts["g3"][:2] == pytest.approx((0.0, 0.0))
        # negative weights are clipped to 0 -> only c3 counts
        assert pts["g4"][:2] == pytest.approx((0.0, -1.0), abs=1e-12)

    def test_direction_and_size(self, data):
        fig = InteractiveVisualization.differential_attribution_radviz(data, WRT)
        pts = points(fig)
        assert {g: p[2] for g, p in pts.items()} == {
            "g0": "Up in A",
            "g1": "Down in A",
            "g2": "Up in A",
            "g3": "Up in A",
            "g4": "Down in A",
        }
        sizes = {g: p[3] for g, p in pts.items()}
        assert sizes == pytest.approx(
            {"g0": 2.0, "g1": 1.0, "g2": 0.01, "g3": 0.01, "g4": 0.5}
        )

    def test_without_logfc_column(self, data):
        fig = InteractiveVisualization.differential_attribution_radviz(
            data.drop(columns="LogFC"), WRT
        )
        pts = points(fig)
        assert {p[2] for p in pts.values()} == {"N/A"}
        assert {p[3] for p in pts.values()} == {1.0}

    def test_custom_columns(self, data):
        renamed = data.rename(
            columns=lambda c: c.replace("abs_contrib_", "w_").replace("LogFC", "lfc")
        )
        fig = InteractiveVisualization.differential_attribution_radviz(
            renamed, WRT, logfc_column="lfc", abs_contrib_prefix="w_"
        )
        assert points(fig)["g1"][2] == "Down in A"

    def test_does_not_mutate_input(self, data):
        before = data.copy()
        InteractiveVisualization.differential_attribution_radviz(data, WRT)
        pd.testing.assert_frame_equal(data, before)


class TestDecorations:
    def test_annotations_circle_and_layout(self, data):
        fig = InteractiveVisualization.differential_attribution_radviz(
            data, WRT, title="R", width=500
        )
        assert [a.text for a in fig.layout.annotations] == WRT
        assert fig.layout.annotations[1].x == pytest.approx(0.0, abs=1e-12)
        assert fig.layout.annotations[1].y == pytest.approx(1.0)
        circle = fig.data[-1]
        assert circle.mode == "lines"
        assert circle.showlegend is False
        assert len(circle.x) == 200
        np.testing.assert_allclose(np.hypot(circle.x, circle.y), 1.0)
        assert fig.layout.title.text == "R"
        assert fig.layout.width == 500
        assert fig.layout.xaxis.scaleanchor == "y"
        assert fig.layout.xaxis.showticklabels is False

    @pytest.mark.parametrize(
        "filename,html,image", [(None, 0, 0), ("r.html", 1, 0), ("r.png", 0, 1)]
    )
    def test_saving(self, data, figure_io, filename, html, image):
        InteractiveVisualization.differential_attribution_radviz(
            data, WRT, filename=filename
        )
        assert [c[1] for c in figure_io.write_html.calls] == [(filename,)] * html
        assert [c[1] for c in figure_io.write_image.calls] == [(filename,)] * image
        assert len(figure_io.show.calls) == 1
