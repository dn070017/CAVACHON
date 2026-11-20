import pytest
from pydantic import ValidationError

from cavachon.config import (
    AnalysisAttributionScoreConfig,
    AnalysisClusteringConfig,
    AnalysisConfig,
    AnalysisEnrichmentConfig,
    AnalysisGenericConfig,
    AnalysisVisualizeEmbeddingConfig,
)
from cavachon.config.analysis_config import (
    AnalysisDifferentialAnalysisConfig,
    AnalysisDifferentialAttributionConfig,
    AnalysisHierarchicalDifferentialAnalysisConfig,
)


def test_generic_defaults_and_sanitize():
    g = AnalysisGenericConfig()
    assert g.modality == "" and g.component == ""
    g = AnalysisGenericConfig(modality="my rna", component="c 1")
    assert (g.modality, g.component) == ("my_rna", "c_1")


# ---------------------------------------------------------------- clustering
def test_clustering_defaults():
    c = AnalysisClusteringConfig(modality="RNA", component="RNA")
    assert c.use_rep == "z" and c.min_n_obs == 36


@pytest.mark.parametrize("rep", ["z", "z_hat"])
def test_clustering_use_rep_valid(rep):
    assert AnalysisClusteringConfig(use_rep=rep).use_rep == rep


@pytest.mark.parametrize("rep", ["Z", "zhat", ""])
def test_clustering_use_rep_invalid(rep):
    with pytest.raises(ValidationError, match="use_rep"):
        AnalysisClusteringConfig(use_rep=rep)


@pytest.mark.parametrize("n", [0, -1])
def test_clustering_min_n_obs_positive(n):
    with pytest.raises(
        ValidationError, match=f"min_n_obs should be a positive integer, got {n}"
    ):
        AnalysisClusteringConfig(min_n_obs=n)


def test_clustering_min_n_obs_one_allowed():
    assert AnalysisClusteringConfig(min_n_obs=1).min_n_obs == 1


# ---------------------------------------------------------------- attribution
def test_attribution_defaults_and_sanitize():
    a = AnalysisAttributionScoreConfig(with_respect_to=["a b"])
    assert a.with_respect_to == ["a_b"]
    assert a.use_cluster == ""


def test_differential_attribution_defaults():
    d = AnalysisDifferentialAttributionConfig()
    assert d.analysis_type == "deg"
    assert d.column == "K(A>B|Z)"
    assert d.threshold == 3.2
    assert d.threshold_operator == "geq"
    assert d.use_absolute is True
    assert d.min_num_degs == 10
    assert d.n_steps == 32
    assert d.recursive is True


def test_differential_attribution_validation():
    with pytest.raises(ValidationError):
        AnalysisDifferentialAttributionConfig(analysis_type="xyz")
    with pytest.raises(ValidationError):
        AnalysisDifferentialAttributionConfig(threshold_operator="eq")
    with pytest.raises(ValidationError, match="n_steps"):
        AnalysisDifferentialAttributionConfig(n_steps=0)
    with pytest.raises(
        ValidationError, match="n_steps should be a positive integer, got -1"
    ):
        AnalysisDifferentialAttributionConfig(n_steps=-1)
    assert AnalysisDifferentialAttributionConfig(n_steps=1).n_steps == 1
    assert (
        AnalysisDifferentialAttributionConfig(analysis_type="hdeg").analysis_type
        == "hdeg"
    )


# ---------------------------------------------------------------- visualize
@pytest.mark.parametrize("method", ["pca", "umap", "tsne"])
def test_visualize_methods_valid(method):
    v = AnalysisVisualizeEmbeddingConfig(modality="RNA", embedding_method=method)
    assert v.embedding_method == method
    assert v.interactive is False


@pytest.mark.parametrize("method", ["PCA", "", "mds"])
def test_visualize_methods_invalid(method):
    with pytest.raises(ValidationError, match="embedding_method"):
        AnalysisVisualizeEmbeddingConfig(embedding_method=method)


def test_visualize_missing_embedding_method_rejected():
    # regression: the default '' skipped the validator, so an omitted
    # embedding_method was accepted while an explicit '' was rejected
    with pytest.raises(ValidationError, match="embedding_method"):
        AnalysisVisualizeEmbeddingConfig(modality="RNA")


# ---------------------------------------------------------------- differential
def test_differential_analysis_defaults():
    d = AnalysisDifferentialAnalysisConfig(modality="RNA", component="RNA")
    assert d.use_cluster == "cluster_RNA"
    assert d.group_a == "" and d.group_b == ""
    assert (d.z_sampling_size, d.x_sampling_size, d.batch_size) == (5, 1000, 128)
    assert d.keep_only_significant is False and d.sort_output is True


def test_differential_analysis_explicit_use_cluster_kept():
    d = AnalysisDifferentialAnalysisConfig(component="RNA", use_cluster="ct")
    assert d.use_cluster == "ct"


def test_hierarchical_defaults():
    h = AnalysisHierarchicalDifferentialAnalysisConfig(
        modality="RNA", component="RNA", donor_components=["a b"]
    )
    assert h.donor_components == ["a_b"]
    assert h.n_samples == 10 and h.seed is None and h.batch_size == 128
    assert AnalysisHierarchicalDifferentialAnalysisConfig().donor_components is None


# regression: hierarchical DA use_cluster defaults to cluster_<component>
def test_hierarchical_use_cluster_default_matches_differential():
    h = AnalysisHierarchicalDifferentialAnalysisConfig(modality="RNA", component="RNA")
    assert h.use_cluster == "cluster_RNA"
    explicit = AnalysisHierarchicalDifferentialAnalysisConfig(
        component="RNA", use_cluster="ct"
    )
    assert explicit.use_cluster == "ct"


# ---------------------------------------------------------------- enrichment
def test_enrichment_requires_analysis_type():
    with pytest.raises(ValidationError):
        AnalysisEnrichmentConfig()


def test_enrichment_defaults():
    e = AnalysisEnrichmentConfig(analysis_type="deg")
    assert e.gene_sets == "KEGG_2019_Mouse"
    assert e.organism == "Mouse"
    assert e.terms is None
    assert e.metric == "FDR q-val"
    assert (e.permutation_num, e.min_size, e.max_size) == (1000, 15, 500)


@pytest.mark.parametrize("field", ["permutation_num", "min_size", "max_size"])
def test_enrichment_positive_ints(field):
    with pytest.raises(ValidationError, match="positive integer"):
        AnalysisEnrichmentConfig(analysis_type="deg", **{field: 0})


def test_enrichment_min_greater_than_max():
    with pytest.raises(ValidationError, match="min_size"):
        AnalysisEnrichmentConfig(analysis_type="hdeg", min_size=20, max_size=10)
    e = AnalysisEnrichmentConfig(analysis_type="hdeg", min_size=10, max_size=10)
    assert e.min_size == e.max_size


# ---------------------------------------------------------------- container
def test_analysis_config_defaults_empty_and_not_shared():
    a, b = AnalysisConfig(), AnalysisConfig()
    for field in AnalysisConfig.model_fields:
        assert getattr(a, field) == []
    a.clustering.append(AnalysisClusteringConfig())
    assert b.clustering == []


def test_analysis_config_parses_nested():
    a = AnalysisConfig(
        clustering=[{"modality": "RNA", "component": "RNA"}],
        enrichment_analysis=[{"analysis_type": "deg"}],
    )
    assert isinstance(a.clustering[0], AnalysisClusteringConfig)
    assert isinstance(a.enrichment_analysis[0], AnalysisEnrichmentConfig)


def test_analysis_config_unknown_section_warns():
    with pytest.warns(RuntimeWarning, match="clusterin"):
        AnalysisConfig(clusterin=[])
