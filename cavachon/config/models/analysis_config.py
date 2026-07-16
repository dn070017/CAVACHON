from typing import List

from pydantic import field_validator

from cavachon.config.models.base import BaseConfigModel, TensorflowCompatibleStr


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


class AnalysisAttributionScoreConfig(BaseConfigModel):
    """Attribution score analysis configuration."""

    modality: TensorflowCompatibleStr = ""
    component: TensorflowCompatibleStr = ""
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


class AnalysisConfig(BaseConfigModel):
    """Top-level analysis configuration."""

    clustering: List[AnalysisClusteringConfig] = []
    differential_analysis: List[AnalysisGenericConfig] = []
    visualize_embedding: List[AnalysisVisualizeEmbeddingConfig] = []
    annotation_colnames: List[str] = []
    conditional_attribution_scores: List[AnalysisAttributionScoreConfig] = []
