"""Unit tests for the InteractiveVisualization wrappers around other
cavachon analyses: neighbors_with_same_annotations (ClusterAnalysis),
attribution_score (AttributionAnalysis) and differential_volcano_plot
(DifferentialAnalysis). The analyses are MagicMocks (conftest ``iv_deps``);
plotly, tensorflow, muon and pandas are real.
"""

import anndata
import muon as mu
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from cavachon.tools.interactive_visualization import InteractiveVisualization

CLUSTERS = ["a", "a", "b", "b"]


@pytest.fixture
def mdata():
    obs = pd.DataFrame({"cluster": CLUSTERS}, index=[f"c{i}" for i in range(4)])
    rna = anndata.AnnData(X=np.zeros((4, 2), dtype=np.float32), obs=obs)
    return mu.MuData({"rna": rna})


# ---------------------------------------------------------------------------
# neighbors_with_same_annotations
# ---------------------------------------------------------------------------


@pytest.fixture
def neighbor_result(iv_deps):
    result = pd.DataFrame(
        {
            "Number of Neighbors": [5, 5, 10, 10, 5, 10],
            "Cell Enrichment Score": [0.2, 0.4, 0.6, 0.8, 0.1, 0.3],
            "Cluster": ["x", "x", "x", "x", "y", "y"],
        }
    )
    analysis = iv_deps.ClusterAnalysis.return_value
    analysis.compute_neighbors_with_same_annotations.return_value = result
    return result


class TestNeighborsWithSameAnnotations:
    def test_calls_cluster_analysis(self, iv_deps, neighbor_result):
        InteractiveVisualization.neighbors_with_same_annotations(
            "MDATA", "MODEL", "rna", "cluster", "z", n_neighbors=[5, 10]
        )
        iv_deps.ClusterAnalysis.assert_called_once_with("MDATA", "MODEL")
        compute = (
            iv_deps.ClusterAnalysis.return_value.compute_neighbors_with_same_annotations
        )
        compute.assert_called_once_with(
            modality="rna", use_cluster="cluster", use_rep="z", n_neighbors=[5, 10]
        )

    def test_ungrouped_single_bar(self, neighbor_result):
        fig = InteractiveVisualization.neighbors_with_same_annotations(
            "M", "N", "rna", "cluster", "z"
        )
        assert [t.name for t in fig.data] == ["Control"]
        np.testing.assert_array_equal(fig.data[0].x, [5, 10])
        np.testing.assert_allclose(
            fig.data[0].y,
            neighbor_result.groupby("Number of Neighbors")[
                "Cell Enrichment Score"
            ].mean(),
        )

    def test_grouped_by_cluster(self, neighbor_result):
        fig = InteractiveVisualization.neighbors_with_same_annotations(
            "M",
            "N",
            "rna",
            "cluster",
            "z",
            group_by_cluster=True,
            color_discrete_map={"y": "green"},
            title="NN",
        )
        assert [t.name for t in fig.data] == ["x", "y"]
        assert fig.data[1].marker.color == "green"
        np.testing.assert_allclose(fig.data[0].y, [0.3, 0.7])
        assert fig.layout.title.text == "NN"
        assert fig.layout.xaxis.title.text == "Number of Neighbors"
        assert fig.layout.yaxis.title.text == "Cell Enrichment Score"

    @pytest.mark.parametrize(
        "filename,html,image", [(None, 0, 0), ("n.html", 1, 0), ("n.svg", 0, 1)]
    )
    def test_saving(self, neighbor_result, figure_io, filename, html, image):
        InteractiveVisualization.neighbors_with_same_annotations(
            "M", "N", "rna", "cluster", "z", filename=filename
        )
        assert len(figure_io.show.calls) == 1
        assert [c[1] for c in figure_io.write_html.calls] == [(filename,)] * html
        assert [c[1] for c in figure_io.write_image.calls] == [(filename,)] * image


# ---------------------------------------------------------------------------
# attribution_score
# ---------------------------------------------------------------------------


def ig_tensor(values):
    """(n_obs, 2, 2) tensor whose mean |.| over axes 1, 2 equals values."""
    values = np.asarray(values, dtype=np.float32)
    signs = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
    return tf.constant(values[:, None, None] * signs[None])


@pytest.fixture
def attribution(iv_deps):
    per_wrt = {
        "z1": ig_tensor([1.0, 3.0, 2.0, 2.0]),
        "z2": ig_tensor([3.0, 1.0, 2.0, 6.0]),
    }
    analysis = iv_deps.AttributionAnalysis.return_value
    analysis.compute_integrated_gradient.side_effect = lambda **kw: per_wrt[
        kw["with_respect_to"]
    ]
    return analysis


class TestAttributionScore:
    def test_constructs_analysis_and_calls_per_wrt(self, iv_deps, attribution, mdata):
        encoders = {"rna": {"batch": "enc"}}
        InteractiveVisualization.attribution_score(
            mdata,
            "MODEL",
            component="comp",
            modality="rna",
            with_respect_to=["z1", "z2"],
            use_cluster="cluster",
            steps=4,
            selected_variables=["v1"],
            batch_size=16,
            batch_effect_colnames={"rna": ["batch"]},
            distribution_names={"rna": "D"},
            batch_effect_encoders=encoders,
        )
        iv_deps.AttributionAnalysis.assert_called_once_with(
            mdata, "MODEL", {"rna": ["batch"]}, {"rna": "D"}, encoders
        )
        calls = attribution.compute_integrated_gradient.call_args_list
        assert [c.kwargs for c in calls] == [
            dict(
                component="comp",
                modality="rna",
                with_respect_to=w,
                steps=4,
                selected_variables=["v1"],
                batch_size=16,
            )
            for w in ["z1", "z2"]
        ]

    def test_scores_are_mean_abs_and_normalised_across_wrt(self, attribution, mdata):
        figs, scores = InteractiveVisualization.attribution_score(
            mdata, "MODEL", "comp", "rna", ["z1", "z2"], "cluster"
        )
        np.testing.assert_allclose(scores[0].numpy(), [1.0, 3.0, 2.0, 2.0])
        np.testing.assert_allclose(scores[1].numpy(), [3.0, 1.0, 2.0, 6.0])

        assert len(figs) == 2
        # z1 fractions: [.25, .75, .5, .25] -> cluster a: .5, b: .375
        fig = figs[0]
        assert [t.name for t in fig.data] == ["a", "b"]
        np.testing.assert_allclose([t.y[0] for t in fig.data], [0.5, 0.375])
        np.testing.assert_allclose([t.y[0] for t in figs[1].data], [0.5, 0.625])

    def test_single_wrt_is_not_normalised(self, attribution, mdata):
        figs, scores = InteractiveVisualization.attribution_score(
            mdata, "MODEL", "comp", "rna", ["z2"], "cluster"
        )
        assert len(scores) == 1
        np.testing.assert_allclose([t.y[0] for t in figs[0].data], [2.0, 4.0])

    def test_figure_layout(self, attribution, mdata):
        figs, _ = InteractiveVisualization.attribution_score(
            mdata,
            "MODEL",
            "comp",
            "rna",
            ["z1", "z2"],
            "cluster",
            title="Attr",
            color_discrete_map={"a": "black"},
        )
        assert [f.layout.title.text for f in figs] == ["Attr (z1)", "Attr (z2)"]
        for fig in figs:
            assert fig.layout.xaxis.showticklabels is False
            assert fig.layout.xaxis.title.text == "Cluster"
            assert fig.layout.yaxis.title.text == "Attribution Score"
            assert fig.data[0].marker.color == "black"

    def test_single_wrt_saved_to_filename(self, attribution, mdata, figure_io):
        figs, _ = InteractiveVisualization.attribution_score(
            mdata, "MODEL", "comp", "rna", ["z1"], "cluster", filename="a.html"
        )
        assert figure_io.write_html.calls == [(figs[0], ("a.html",), {})]

    def test_no_filename_no_write(self, attribution, mdata, figure_io):
        InteractiveVisualization.attribution_score(
            mdata, "MODEL", "comp", "rna", ["z1", "z2"], "cluster"
        )
        assert figure_io.write_html.calls == []
        assert figure_io.write_image.calls == []
        assert len(figure_io.show.calls) == 2

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: attribution_score writes every with_respect_to figure to "
        "the same filename, so all but the last are overwritten",
    )
    def test_multiple_wrt_saved_to_distinct_files(self, attribution, mdata, figure_io):
        InteractiveVisualization.attribution_score(
            mdata, "MODEL", "comp", "rna", ["z1", "z2"], "cluster", filename="a.html"
        )
        filenames = [c[1][0] for c in figure_io.write_html.calls]
        assert len(filenames) == 2
        assert len(set(filenames)) == 2


# ---------------------------------------------------------------------------
# differential_volcano_plot
# ---------------------------------------------------------------------------


@pytest.fixture
def degs():
    return pd.DataFrame(
        {
            "Mean(A)": [4.0, 1.0, 2.0, 1.0],
            "Mean(B)": [1.0, 4.0, 2.0, 1.0],
            "K(A>B|Z)": [5.0, -6.0, 1.0, -1.0],
        },
        index=["g1", "g2", "g3", "g4"],
    )


class TestDifferentialVolcanoPlot:
    def test_given_results_derived_columns(self, degs, iv_deps):
        fig = InteractiveVisualization.differential_volcano_plot(degs.copy())

        iv_deps.DifferentialAnalysis.assert_not_called()
        names = {t.name: t for t in fig.data}
        assert set(names) == {"Significant", "Non-significant"}
        sig = names["Significant"]
        # g1 up (K=5), g2 down (K=-6 -> 6): both significant at 3.2
        np.testing.assert_allclose(sig.x, np.log([4.0, 0.25]))
        np.testing.assert_allclose(sig.y, [5.0, 6.0])
        assert list(sig.customdata[:, 0]) == ["g1", "g2"]
        non = names["Non-significant"]
        # g3: logfc 0 -> +K = 1; g4: logfc 0 -> +K = -1
        np.testing.assert_allclose(non.y, [1.0, -1.0])
        assert fig.layout.xaxis.title.text == "LogFC(A/B)"
        assert fig.layout.yaxis.title.text == "K(A!=B|Z)"
        assert fig.layout.title.text == "Volcano Plot for Differential Analysis"

    def test_threshold(self, degs):
        fig = InteractiveVisualization.differential_volcano_plot(
            degs.copy(), significant_threshold=5.5
        )
        sig = [t for t in fig.data if t.name == "Significant"][0]
        np.testing.assert_allclose(sig.y, [6.0])

    def test_threshold_is_strict(self, degs):
        fig = InteractiveVisualization.differential_volcano_plot(
            degs.copy(), significant_threshold=5.0
        )
        sig = [t for t in fig.data if t.name == "Significant"][0]
        np.testing.assert_allclose(sig.y, [6.0])

    def test_runs_differential_analysis_when_no_results(self, degs, iv_deps):
        analysis = iv_deps.DifferentialAnalysis.return_value
        analysis.between_two_groups.return_value = degs
        InteractiveVisualization.differential_volcano_plot(
            mdata="MDATA",
            model="MODEL",
            group_a_index=["c0"],
            group_b_index=["c1"],
            component="comp",
            modality="rna",
            title="V",
        )
        iv_deps.DifferentialAnalysis.assert_called_once_with(
            mdata="MDATA", model="MODEL"
        )
        analysis.between_two_groups.assert_called_once_with(
            group_a_index=["c0"],
            group_b_index=["c1"],
            component="comp",
            modality="rna",
        )

    @pytest.mark.parametrize("missing", ["component", "modality"])
    def test_missing_component_or_modality_raises(self, iv_deps, missing):
        kwargs = dict(mdata="M", model="N", component="comp", modality="rna")
        kwargs[missing] = None
        with pytest.raises(ValueError, match="component and modality are required"):
            InteractiveVisualization.differential_volcano_plot(**kwargs)
        iv_deps.DifferentialAnalysis.assert_not_called()

    @pytest.mark.parametrize(
        "filename,html,image", [(None, 0, 0), ("v.html", 1, 0), ("v.pdf", 0, 1)]
    )
    def test_saving(self, degs, figure_io, filename, html, image):
        InteractiveVisualization.differential_volcano_plot(
            degs.copy(), filename=filename
        )
        assert [c[1] for c in figure_io.write_html.calls] == [(filename,)] * html
        assert [c[1] for c in figure_io.write_image.calls] == [(filename,)] * image
        assert len(figure_io.show.calls) == 1
