from cavachon.config.config_mapping.ConfigMapping import ConfigMapping
from cavachon.utils.GeneralUtils import GeneralUtils
from typing import Any, Mapping

class AnalysisAttributionScoreConfig(ConfigMapping):
  """AnalysisAttributionScoreConfig

  Config for attribution score analysis.

  Attributes
  ----------
  modality: str
      which modality of the outputs of the component to used.

  component: str
      the outputs of which component to used.

  with_respect_to: str
      compute integrated gradietn with respect to the latent 
      representation of which component.
    
  """
  def __init__(self, **kwargs: Mapping[str, Any]):
    """Constructor for AnalysisAttributionScoreConfig. 

    Parameters
    ----------
    modality: str
        which modality of the outputs of the component to used.

    component: str
        the outputs of which component to used.

    with_respect_to: str
        compute integrated gradietn with respect to the latent 
        representation of which component.
    
    """
    # change default values here
    self.modality: str = ''
    self.component: str = ''
    self.with_respect_to: str = ''

    super().__init__(kwargs)
    self.modality = GeneralUtils.tensorflow_compatible_str(self.modality)
    self.component = GeneralUtils.tensorflow_compatible_str(self.component)
    self.with_respect_to = GeneralUtils.tensorflow_compatible_str(self.with_respect_to)