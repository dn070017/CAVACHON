"""Unit tests for InteractiveVisualization.bar and .scatter (plotly real)."""

import numpy as np
import pandas as pd
import pytest

from cavachon.tools.interactive_visualization import InteractiveVisualization


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "x": [1, 1, 2, 2, 1, 1, 2, 2],
            "y": [1.0, 3.0, 2.0, 4.0, 10.0, 20.0, 5.0, 7.0],
            "g": ["b", "b", "b", "b", "a", "a", "a", "a"],
        }
    )


class TestBarWithoutGroup:
    def test_single_control_trace_with_mean_and_sem(self, data):
        fig = InteractiveVisualization.bar(data, x="x", y="y")

        assert len(fig.data) == 1
        trace = fig.data[0]
        assert trace.type == "bar"
        assert trace.name == "Control"
        np.testing.assert_array_equal(trace.x, [1, 2])
        expected = data.groupby("x")["y"]
        np.testing.assert_allclose(trace.y, expected.mean().to_numpy())
        np.testing.assert_allclose(trace.error_y.array, expected.sem().to_numpy())
        assert trace.error_y.type == "data"
        assert trace.marker.opacity == 0.7
        assert trace.marker.line.color == "DarkSlateGrey"
        assert trace.marker.line.width == 1

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: bar() ignores **kwargs (layout args) when group is None",
    )
    def test_layout_kwargs_applied(self, data):
        fig = InteractiveVisualization.bar(
            data, x="x", y="y", title="T", xaxis_title="X axis"
        )
        assert fig.layout.title.text == "T"
        assert fig.layout.xaxis.title.text == "X axis"


class TestBarWithGroup:
    def test_one_trace_per_sorted_group(self, data):
        fig = InteractiveVisualization.bar(data, x="x", y="y", group="g")

        assert [t.name for t in fig.data] == ["a", "b"]
        sub = data[data.g == "a"].groupby("x")["y"]
        np.testing.assert_allclose(fig.data[0].y, sub.mean().to_numpy())
        np.testing.assert_allclose(fig.data[0].error_y.array, sub.sem().to_numpy())
        assert fig.layout.barmode == "group"

    def test_color_map_used_only_for_mapped_groups(self, data):
        fig = InteractiveVisualization.bar(
            data, x="x", y="y", group="g", color_discrete_map={"b": "red"}
        )
        colors = {t.name: t.marker.color for t in fig.data}
        assert colors == {"a": None, "b": "red"}

    def test_layout_kwargs_applied(self, data):
        fig = InteractiveVisualization.bar(
            data, x="x", y="y", group="g", title="T", yaxis_title="Y"
        )
        assert fig.layout.title.text == "T"
        assert fig.layout.yaxis.title.text == "Y"

    def test_does_not_mutate_input(self, data):
        before = data.copy()
        InteractiveVisualization.bar(data, x="x", y="y", group="g")
        pd.testing.assert_frame_equal(data, before)


class TestScatter:
    def test_styles_markers_and_legend(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1], "c": ["u", "v", "u"]})
        fig = InteractiveVisualization.scatter(df, x="a", y="b", color="c")

        assert [t.name for t in fig.data] == ["u", "v"]
        for trace in fig.data:
            assert trace.marker.opacity == 0.7
            assert trace.marker.line.width == 1
            assert trace.marker.line.color == "DarkSlateGrey"
        assert fig.layout.legend.traceorder == "normal"
        np.testing.assert_array_equal(fig.data[0].x, [1, 3])

    def test_passes_kwargs_to_px_scatter(self):
        fig = InteractiveVisualization.scatter(x=[1, 2], y=[3, 4], title="Hello")
        assert fig.layout.title.text == "Hello"
        np.testing.assert_array_equal(fig.data[0].y, [3, 4])

    def test_does_not_show_or_write(self, figure_io):
        InteractiveVisualization.scatter(x=[1], y=[1])
        assert figure_io.show.calls == []
        assert figure_io.write_html.calls == []
