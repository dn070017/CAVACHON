import pytest
from pydantic import ValidationError

from cavachon.config.analysis_visualize_embedding_config import (
    AnalysisVisualizeEmbeddingConfig,
)
from cavachon.utils.general_utils import GeneralUtils


def test_analysis_visualize_embedding_config_valid():
    config_data = {
        "modality": "Modality 1",
        "use_rep": "Component A",
        "embedding_method": "umap",
        "color_by": "cluster_label",
        "interactive": True,
    }
    config = AnalysisVisualizeEmbeddingConfig(**config_data)

    assert config.modality == GeneralUtils.convert_to_tensorflow_compatible_string(
        "Modality 1"
    )
    assert config.use_rep == "Component A"
    assert config.embedding_method == "umap"
    assert config.color_by == "cluster_label"
    assert config.interactive is True


def test_analysis_visualize_embedding_config_defaults():
    config_data = {
        "modality": "Modality 2",
        "use_rep": "Component B",
        "embedding_method": "pca",
        "color_by": "cell_type",
        # interactive is optional, defaults to False
    }
    config = AnalysisVisualizeEmbeddingConfig(**config_data)

    assert config.modality == GeneralUtils.convert_to_tensorflow_compatible_string(
        "Modality 2"
    )
    assert config.use_rep == "Component B"
    assert config.embedding_method == "pca"
    assert config.color_by == "cell_type"
    assert config.interactive is False


def test_analysis_visualize_embedding_config_missing_fields():
    base_data = {
        "modality": "Modality 3",
        "use_rep": "Rep C",
        "embedding_method": "tsne",
        "color_by": "batch",
    }

    # missing modality
    with pytest.raises(ValidationError):
        AnalysisVisualizeEmbeddingConfig(
            **{k: v for k, v in base_data.items() if k != "modality"}
        )

    # missing use_rep
    with pytest.raises(ValidationError):
        AnalysisVisualizeEmbeddingConfig(
            **{k: v for k, v in base_data.items() if k != "use_rep"}
        )

    # missing color_by
    with pytest.raises(ValidationError):
        AnalysisVisualizeEmbeddingConfig(
            **{k: v for k, v in base_data.items() if k != "color_by"}
        )

    # missing all required
    with pytest.raises(ValidationError):
        AnalysisVisualizeEmbeddingConfig()


def test_analysis_visualize_embedding_config_invalid_types():
    base_data = {
        "modality": "Modality 1",
        "use_rep": "Rep A",
        "embedding_method": "umap",
        "color_by": "label",
    }

    # invalid type for modality
    with pytest.raises(ValidationError):
        AnalysisVisualizeEmbeddingConfig(**{**base_data, "modality": 123})

    # invalid type for use_rep
    with pytest.raises(ValidationError):
        AnalysisVisualizeEmbeddingConfig(**{**base_data, "use_rep": ["list"]})

    # invalid type for embedding_method (not a string)
    with pytest.raises(ValidationError):
        AnalysisVisualizeEmbeddingConfig(**{**base_data, "embedding_method": 123})

    # invalid type for color_by
    with pytest.raises(ValidationError):
        AnalysisVisualizeEmbeddingConfig(**{**base_data, "color_by": None})

    # invalid type for interactive
    with pytest.raises(ValidationError):
        AnalysisVisualizeEmbeddingConfig(**{**base_data, "interactive": "not_a_bool"})


def test_analysis_visualize_embedding_config_invalid_embedding_method():
    base_data = {
        "modality": "Modality 1",
        "use_rep": "Rep A",
        "color_by": "label",
    }
    with pytest.raises(ValidationError):
        AnalysisVisualizeEmbeddingConfig(
            **{**base_data, "embedding_method": "invalid_method"}
        )


def test_analysis_visualize_embedding_config_assignment():
    config = AnalysisVisualizeEmbeddingConfig(
        modality="Initial Modality",
        use_rep="Initial Rep",
        embedding_method="pca",
        color_by="initial_color",
    )

    assert config.modality == "initial_modality"
    assert config.use_rep == "Initial Rep"
    assert config.embedding_method == "pca"
    assert config.color_by == "initial_color"
    assert config.interactive is False

    # Assign new values and check transformation/validation
    config.modality = "New Modality"
    assert config.modality == "new_modality"

    config.use_rep = "New Rep"
    assert config.use_rep == "New Rep"

    config.embedding_method = "tsne"
    assert config.embedding_method == "tsne"

    config.color_by = "new_color"
    assert config.color_by == "new_color"

    config.interactive = True
    assert config.interactive is True

    # Test invalid assignment type
    with pytest.raises(ValidationError):
        config.modality = 456

    with pytest.raises(ValidationError):
        config.use_rep = None

    with pytest.raises(ValidationError):
        config.embedding_method = "wrong_method"

    with pytest.raises(ValidationError):
        config.color_by = 123

    with pytest.raises(ValidationError):
        config.interactive = "maybe"
