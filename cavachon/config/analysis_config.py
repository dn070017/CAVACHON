from typing import List, Optional

from pydantic import field_validator, model_validator

from cavachon.config.base import BaseConfigModel, TensorflowCompatibleStr


class AnalysisGenericConfig(BaseConfigModel):
    """Generic analysis config referencing modality and component."""

    modality: TensorflowCompatibleStr = ""
    component: TensorflowCompatibleStr = ""


class AnalysisClusteringConfig(AnalysisGenericConfig):
    """Clustering analysis configuration.

    Extends ``AnalysisGenericConfig`` with ``use_rep`` and
    ``min_n_obs``.

    """

    use_rep: str = "z"
    min_n_obs: int = 36

    @field_validator("use_rep", mode="after")
    @classmethod
    def _validate_use_rep(cls, v: str) -> str:
        if v not in ("z", "z_hat"):
            raise ValueError("use_rep should be one of 'z' or 'z_hat'")
        return v

    @field_validator("min_n_obs")
    @classmethod
    def _validate_min_n_obs(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"min_n_obs should be a positive integer, got {v}")
        return v


class AnalysisAttributionScoreConfig(AnalysisGenericConfig):
    """Attribution score analysis configuration."""

    with_respect_to: List[TensorflowCompatibleStr] = []
    use_cluster: str = ""


class AnalysisVisualizeEmbeddingConfig(BaseConfigModel):
    """Embedding visualization configuration."""

    modality: TensorflowCompatibleStr = ""
    use_rep: TensorflowCompatibleStr = ""
    embedding_method: str = ""
    color_by: str = ""
    interactive: bool = False

    @field_validator("embedding_method")
    @classmethod
    def _validate_embedding_method(cls, v: str) -> str:
        if v not in ("pca", "umap", "tsne"):
            raise ValueError(
                "embedding_method should be one of 'pca', 'umap' or 'tsne'"
            )
        return v


class AnalysisDifferentialAnalysisConfig(AnalysisGenericConfig):
    """Differential expression analysis configuration.

    Two modes are supported based on whether ``group_a`` and ``group_b``
    are provided:

    - **Pairwise mode** (default): both ``group_a`` and ``group_b`` are
      empty. The workflow calls ``across_clusters_pairwise``, comparing
      every pair of clusters automatically.
    - **Single-pair mode**: both ``group_a`` and ``group_b`` are
      specified. The workflow calls ``between_two_groups`` for that one
      pair only.

    """

    use_cluster: str = ""
    group_a: str = ""
    group_b: str = ""
    z_sampling_size: int = 5
    x_sampling_size: int = 1000
    batch_size: int = 128
    keep_only_significant: bool = False
    sort_output: bool = True

    @model_validator(mode="after")
    def _default_use_cluster(self) -> "AnalysisDifferentialAnalysisConfig":
        if not self.use_cluster:
            self.use_cluster = f"cluster_{self.component}"
        return self


class AnalysisHierarchicalDifferentialAnalysisConfig(AnalysisGenericConfig):
    """Hierarchical differential expression analysis configuration.

    Two modes are supported based on whether ``donor_cluster`` and
    ``recipient_cluster`` are provided:

    - **Pairwise mode** (default): both ``donor_cluster`` and
      ``recipient_cluster`` are empty. The workflow calls
      ``across_clusters_pairwise``, running interventions between every
      pair of clusters automatically.
    - **Single-pair mode**: both ``donor_cluster`` and
      ``recipient_cluster`` are specified. The workflow calls
      ``between_clusters`` for that one pair only.

    """

    use_cluster: str = ""
    donor_cluster: str = ""
    recipient_cluster: str = ""
    donor_components: Optional[List[TensorflowCompatibleStr]] = None
    n_samples: int = 10
    seed: Optional[int] = None
    batch_size: int = 128
    sort_output: bool = True


class AnalysisConfig(BaseConfigModel):
    """Top-level analysis configuration."""

    clustering: List[AnalysisClusteringConfig] = []
    differential_analysis: List[AnalysisDifferentialAnalysisConfig] = []
    visualize_embedding: List[AnalysisVisualizeEmbeddingConfig] = []
    conditional_attribution_scores: List[AnalysisAttributionScoreConfig] = []
    hierarchical_differential_analysis: List[
        AnalysisHierarchicalDifferentialAnalysisConfig
    ] = []
