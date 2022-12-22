from cavachon.config.config_mapping.AnalysisAttributionScoreConfig import AnalysisAttributionScoreConfig
from cavachon.config.config_mapping.ConfigMapping import ConfigMapping
from cavachon.utils.GeneralUtils import GeneralUtils
from typing import Any, List, Mapping

class AnalysisConfig(ConfigMapping):
  """AnalysisConfig

  Config for analysis.

  Attributes
  ----------
  clustering: Dict[str, str]
        config for clustering. The keys are the modality names, the
        values are the component that is used to identify the clusters
        of modalities. 

  differential_analysis: Dict[str, str]
        config for differential analysis. The keys are the modality 
        to be analyzed, the values are the component of which that 
        generate the modality. 

  embedding_methods: List[str]
      embedding methods used for downstream analysis.

  annotation_colnames: List[str]
      column names for the annotated cluster that needs to be 
      included in the clustering analysis.

  conditional_attribution_scores: List[AnalysisAttributionScoreConfig]
        config for the conditional attribution score analysis. 

  """
  def __init__(self, **kwargs: Mapping[str, Any]):
    """Constructor for AnalysisConfig. 

    Parameters
    ----------
    clustering: Dict[str, str], optional
        config for clustering. The keys are the modality names, the
        values are the component that is used to identify the clusters
        of modalities. Defaults to dict().

    embedding_methods: List[str], optional
        embedding methods used for downstream analysis. Defaults to
        ['tsne'].

    annotation_colnames: List[str], optional
        column names for the annotated cluster that needs to be 
        included in the clustering analysis. Needs to match the column
        names in adata.obs. Defaults to [].
    
    conditional_attribution_scores: List[AnalysisAttributionScoreConfig], optional
        config for the conditional attribution score analysis. 
        Defaults to []. 

    """
    # change default values here
    self.clustering: Mapping[str, str] = dict()
    self.differential_analysis: Mapping[str, str] = dict()
    self.embedding_methods: List[str] = ['tsne']
    self.annotation_colnames: List[str] = []
    self.conditional_attribution_scores: List[AnalysisAttributionScoreConfig] = []

    super().__init__(
        kwargs, 
        ['clustering', 'differential_analysis', 'embedding_methods', 'annotation_colnames', 'conditional_attribution_scores'])
    
    self.clustering = {
        GeneralUtils.tensorflow_compatible_str(x['modality']): GeneralUtils.tensorflow_compatible_str(x['component']) for x in kwargs['clustering']
    }
    
    self.differential_analysis = {
        GeneralUtils.tensorflow_compatible_str(x['modality']): GeneralUtils.tensorflow_compatible_str(x['component']) for x in kwargs['clustering']
    }

    self.conditional_attribution_scores = [
        AnalysisAttributionScoreConfig(**x) for x in self.conditional_attribution_scores
    ]