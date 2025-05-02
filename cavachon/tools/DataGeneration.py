import muon as mu
import tensorflow as tf

class DataGeneration:
  def __init__(
      self,
      mdata: mu.MuData,
      model: tf.keras.Model):
    """Constructor for DataGeneration

    Parameters
    ----------
    mdata: muon.MuData
        the MuData for analysis.

    model: tf.keras.Model
        the trained generative model.
    
    """
    self.mdata = mdata
    self.model = model
  
  def impute_missing_modality(self)
