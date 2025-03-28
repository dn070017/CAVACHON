from typing import Any, Mapping

from cavachon.config.config_mapping.config_mapping import ConfigMapping
from cavachon.utils.general_utils import GeneralUtils


class AnalysisVisualizeEmbedding(ConfigMapping):
    """AnalysisVisualizeEmbedding

    Config mapping for embedding visualization.

    Attributes
    ----------
    modality: str
        which modality of the outputs of the component to used.

    component: str
        the outputs of which component to used.

    embedding_method: str
        the embedding method to use.

    color_by: str
        color by which annotation column.

    interactive: bool
        whether or not to create interactive visualization.

    """

    def __init__(self, **kwargs: Mapping[str, Any]):
        """Constructor for AnalysisAttributionScoreConfigMapping.

        Parameters
        ----------
        modality: str
            which modality to visualize.

        use_rep: str
            which representation of the modality to visualize.

        embedding_method: str
            method used to embed the representation of the modality.
            Should be one of `'pca'`, `'umap'` or `'tsne'`.

        color_by: str
            color by which annotation column.

        interactive: bool
            whether or not to create interactive visualization.

        """
        # change default values here
        self.modality: str = ""
        self.use_rep: str = ""
        self.embedding_method: str = ""
        self.color_by: str = ""
        self.interactive: bool = False

        super().__init__(kwargs)
        self.modality = GeneralUtils.tensorflow_compatible_str(self.modality)
        self.use_rep = GeneralUtils.tensorflow_compatible_str(self.use_rep)
        if self.embedding_method not in ["pca", "umap", "tsne"]:
            raise ValueError(
                "embedding_method should be one of 'pca', 'umap' or 'tsne'"
            )
