"""Unit tests for InteractiveVisualization.embedding.

PCA is real (sklearn); TSNE and umap.UMAP are replaced by FakeReducer
(first two columns * 10). Figure.show / write_html / write_image are
recorders (conftest ``figure_io``).
"""

import anndata
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from cavachon.tools.interactive_visualization import InteractiveVisualization

N_OBS = 6


@pytest.fixture
def adata():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(N_OBS, 3)).astype(np.float32)
    ad = anndata.AnnData(
        X=np.zeros((N_OBS, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"cluster": pd.Categorical(["c1", "c2", "c1", "c2", "c1", "c3"])},
            index=[f"cell{i}" for i in range(N_OBS)],
        ),
    )
    ad.obsm["z"] = z
    ad.uns["cavachon"] = {"name": "rna"}
    return ad


def all_x(fig):
    return np.concatenate([np.asarray(t.x) for t in fig.data])


class TestMethods:
    def test_pca_real_and_cached_in_obsm(self, adata, figure_io):
        fig = InteractiveVisualization.embedding(adata, method="pca")

        expected = PCA(n_components=2, random_state=0).fit_transform(adata.obsm["z"])
        np.testing.assert_allclose(adata.obsm["z_pca"], expected, rtol=1e-5)
        np.testing.assert_allclose(fig.data[0].x, expected[:, 0], rtol=1e-5)
        np.testing.assert_allclose(fig.data[0].y, expected[:, 1], rtol=1e-5)
        assert fig.layout.xaxis.title.text == "PCA 1"
        assert fig.layout.yaxis.title.text == "PCA 2"
        # default title: the obsm key
        assert fig.layout.title.text == "z"
        assert len(figure_io.show.calls) == 1

    def test_tsne_uses_reducer_with_random_state(self, adata, reducers):
        fig = InteractiveVisualization.embedding(adata, method="tsne")

        reducers.TSNE.assert_called_once_with(random_state=0)
        np.testing.assert_array_equal(
            reducers.instances[0].fit_inputs[0], adata.obsm["z"]
        )
        np.testing.assert_allclose(fig.data[0].x, adata.obsm["z"][:, 0] * 10)
        assert fig.layout.xaxis.title.text == "t-SNE 1"
        assert fig.layout.yaxis.title.text == "t-SNE 2"
        assert "z_tsne" in adata.obsm

    def test_umap_arguments(self, adata, reducers):
        fig = InteractiveVisualization.embedding(adata, method="umap")

        reducers.UMAP.assert_called_once_with(
            random_state=0, n_neighbors=30, min_dist=0.0
        )
        assert fig.layout.xaxis.title.text == "Umap 1"
        np.testing.assert_allclose(adata.obsm["z_umap"], adata.obsm["z"][:, :2] * 10)

    def test_invalid_method_warns_and_falls_back_to_tsne(self, adata, reducers):
        with pytest.warns(RuntimeWarning, match="Invalid value for method"):
            fig = InteractiveVisualization.embedding(adata, method="mds")
        reducers.TSNE.assert_called_once()
        assert fig.layout.xaxis.title.text == "t-SNE 1"
        assert "z_tsne" in adata.obsm


class TestCaching:
    def test_existing_embedding_is_reused(self, adata, reducers):
        cached = np.arange(N_OBS * 2, dtype=float).reshape(N_OBS, 2)
        adata.obsm["z_tsne"] = cached
        fig = InteractiveVisualization.embedding(adata, method="tsne")

        reducers.TSNE.assert_not_called()
        np.testing.assert_array_equal(fig.data[0].x, cached[:, 0])

    def test_force_recomputes(self, adata, reducers):
        adata.obsm["z_tsne"] = np.zeros((N_OBS, 2))
        InteractiveVisualization.embedding(adata, method="tsne", force=True)

        reducers.TSNE.assert_called_once()
        np.testing.assert_allclose(adata.obsm["z_tsne"], adata.obsm["z"][:, :2] * 10)

    def test_ndarray_rep_is_not_stored(self, adata, reducers):
        rep = np.arange(N_OBS * 3, dtype=float).reshape(N_OBS, 3)
        keys_before = set(adata.obsm.keys())
        fig = InteractiveVisualization.embedding(adata, method="tsne", use_rep=rep)

        assert set(adata.obsm.keys()) == keys_before
        np.testing.assert_allclose(fig.data[0].x, rep[:, 0] * 10)

    def test_other_obsm_key(self, adata, reducers):
        adata.obsm["z_hat"] = adata.obsm["z"] + 1
        InteractiveVisualization.embedding(adata, method="tsne", use_rep="z_hat")
        assert "z_hat_tsne" in adata.obsm


class TestColor:
    def test_no_color_uses_salmon(self, adata):
        fig = InteractiveVisualization.embedding(adata, method="pca")
        assert len(fig.data) == 1
        assert fig.data[0].marker.color == "salmon"

    def test_obs_column_gives_one_trace_per_category(self, adata):
        fig = InteractiveVisualization.embedding(adata, method="pca", color="cluster")
        assert sorted(t.name for t in fig.data) == ["c1", "c2", "c3"]
        assert sorted(all_x(fig)) == pytest.approx(sorted(adata.obsm["z_pca"][:, 0]))

    def test_obsm_key_index_gives_continuous_color(self, adata):
        fig = InteractiveVisualization.embedding(adata, method="pca", color="z/1")
        np.testing.assert_allclose(fig.data[0].marker.color, adata.obsm["z"][:, 1])

    def test_invalid_index_warns_and_ignores(self, adata):
        with pytest.warns(RuntimeWarning, match="not a valid obsm key/index"):
            fig = InteractiveVisualization.embedding(
                adata, method="pca", color="missing"
            )
        assert fig.data[0].marker.color == "salmon"

    def test_missing_obsm_key_warns_and_ignores(self, adata):
        with pytest.warns(RuntimeWarning, match="nope is not in adata.obsm"):
            fig = InteractiveVisualization.embedding(
                adata, method="pca", color="nope/0"
            )
        assert fig.data[0].marker.color == "salmon"

    def test_explicit_color_discrete_sequence(self, adata):
        fig = InteractiveVisualization.embedding(
            adata, method="pca", color_discrete_sequence=["navy"]
        )
        assert fig.data[0].marker.color == "navy"


class TestTitleAndSaving:
    def test_explicit_title(self, adata):
        fig = InteractiveVisualization.embedding(adata, method="pca", title="My title")
        assert fig.layout.title.text == "My title"

    def test_extra_kwargs_reach_px_scatter(self, adata):
        fig = InteractiveVisualization.embedding(adata, method="pca", width=321)
        assert fig.layout.width == 321

    def test_anndata_without_cavachon_uns(self, adata):
        # regression: T8 - an AnnData not produced by cavachon (no
        # uns['cavachon']) raised AttributeError while building the default
        # title; uns is no longer read
        del adata.uns["cavachon"]
        adata.obsm["latent"] = adata.obsm["z"]
        fig = InteractiveVisualization.embedding(adata, method="pca", use_rep="latent")
        assert fig.layout.title.text == "latent"

    def test_array_use_rep_default_title(self, adata):
        fig = InteractiveVisualization.embedding(
            adata, method="pca", use_rep=np.asarray(adata.obsm["z"])
        )
        assert fig.layout.title.text == "Z"

    def test_no_filename_no_write(self, adata, figure_io):
        InteractiveVisualization.embedding(adata, method="pca")
        assert figure_io.write_html.calls == []
        assert figure_io.write_image.calls == []

    def test_html_filename(self, adata, figure_io):
        fig = InteractiveVisualization.embedding(adata, method="pca", filename="e.html")
        assert figure_io.write_html.calls == [(fig, ("e.html",), {})]
        assert figure_io.write_image.calls == []

    def test_image_filename(self, adata, figure_io):
        fig = InteractiveVisualization.embedding(adata, method="pca", filename="e.png")
        assert figure_io.write_image.calls == [(fig, ("e.png",), {})]
        assert figure_io.write_html.calls == []
