from cavachon.dataloader.DataLoader import DataLoader
from cavachon.distributions.Distribution import Distribution
from cavachon.environment.Constants import Constants
from cavachon.layers.parameterizers.MultivariateNormalDiagSampler import MultivariateNormalDiagSampler
from cavachon.model.Model import Model
from cavachon.modules.components.Component import Component
from cavachon.utils.ReflectionHandler import ReflectionHandler
from cavachon.utils.TensorUtils import TensorUtils
from copy import deepcopy
from typing import Set, Sequence, List, Mapping, Optional, Union
from tqdm import tqdm

import muon as mu
import numpy as np
import pandas as pd
import tensorflow as tf

class AttributionAnalysis:
  """AttributionAnalysis

  Attribution analysis of the latent representation of the component to 
  the outputs.

  Attibutes
  ---------
  mdata: muon.MuData
      the MuData for analysis.

  model: tf.keras.Model
      the trained generative model.

  """
  def __init__(
      self,
      mdata: mu.MuData,
      model: tf.keras.Model):
    """Constructor for ContributionAnalysis.

    Parameters
    ----------
    mdata: muon.MuData
        the MuData for analysis.

    model: tf.keras.Model
        the trained generative model.
    
    """
    self.mdata = mdata
    self.model = model

  def compute_delta_x(
      self,
      component: str,
      modality: str,
      exclude_component: str,
      selected_indices: Optional[Sequence[int]] = None,
      batch_size: int = 128
  ) -> np.ndarray:
    """Compute the x - x_baseline in the integrated gradients. The 
    baseline is the mean of the outputs modality from the component
    without using the latent representation z of the exclude component.

    Parameters
    ----------
    component : str
        the outputs of which component to used.
    
    modality : str
        which modality of the outputs of the component to used.
    
    exclude_component: str
        which component to exclude (the latent representation z will 
        not be used in the forward pass)
    
    selected_indices: Optional[Sequence[int]], optional
        the integer indices for which variables to used. All variables
        will be used if provided with None. Defaults to None.

    batch_size : int, optional
        batch size used for the forward pass. Defaults to 128

    Returns
    -------
    np.ndarray
        x - x_baseline in the integrated gradients.

    """
    dist_x_z_name = self.model.components.get(component).distribution_names.get(modality)
    dist_x_z_class = ReflectionHandler.get_class_by_name(dist_x_z_name, 'distributions')
    delta_x = []
    dataloader = DataLoader(self.mdata, batch_size=batch_size)
    progress_message = "Computing delta x"
    for batch in tqdm(dataloader.dataset.batch(batch_size), desc=progress_message):
      x_means_null_batch = self.compute_attribution_target_batch(
          batch=batch,
          component=component,
          modality=modality,
          target_component=exclude_component,
          selected_indices=selected_indices,
          alpha=0.0)
      
      x_means_full_batch = self.compute_attribution_target_batch(
          batch=batch,
          component=component,
          modality=modality,
          target_component=exclude_component,
          selected_indices=selected_indices,
          alpha=1.0)

      delta_x.append(
          np.reshape(
              np.mean(
                  np.abs(x_means_full_batch - x_means_null_batch),
                  axis=-1), 
              (-1, 1)))

    return np.vstack(delta_x)
    
  def compute_integrated_gradient(
      self,
      component: str,
      modality: str,
      target_component: str, 
      steps: int = 10,
      selected_variables: Optional[Sequence[str]] = None,
      batch_size: int = 128) -> np.ndarray:
    """Compute the integrated gradients of ∂rho_m/∂z_m.

    Parameters
    ----------
    component: str
        the outputs of which component to used.

    modality: str
        which modality of the outputs of the component to used.

    target_component: str
        the latent representation of which component to used.

    steps: int, optional
        steps in integrated gradients. Defaults to 10.
    
    selected_variables: Optional[Sequence[str]], optional
        the variables to used. The provided variables needs to match
        the indices of mdata[modality].var. All variables will be used 
        if provided with None. Defaults to None.

    batch_size: int, optional
        batch size used for the forward pass. Defaults to 128

    Returns
    -------
    np.ndarray
        integrated gradients of ∂rho_m/∂z_m.

    """    
    if selected_variables is not None:
      selected_indices = [self.mdata[modality].var.index.get_loc(var) for var in selected_variables]
    else:
      selected_indices = None

    delta_x = self.compute_delta_x(
        component = component,
        modality = modality,
        exclude_component = target_component,
        selected_indices = selected_indices,
        batch_size = batch_size)
    
    dataloader = DataLoader(self.mdata, batch_size=batch_size)
    delta_x_split = TensorUtils.split(delta_x, batch_size=batch_size)

    progress_message = "Computing integrated gradient"
    integrated_gradients = []
    for batch, batch_delta_x in tqdm(zip(dataloader, delta_x_split), desc=progress_message):
      outputs = self.model(batch, training=False)
      unintegrated_gradients_batch = [] 
      for alpha in [x / steps for x in range(steps + 1)]:
        with tf.GradientTape() as tape:
          z_variable = tf.Variable(outputs.get(f'{target_component}/{Constants.MODEL_OUTPUTS_Z}'))
          n_latent_dims = z_variable.shape[-1]
          x_means = self.compute_attribution_target_batch(
              batch = batch,
              component = component,
              modality = modality,
              target_component = target_component,
              selected_indices = selected_indices,
              z_variable = z_variable,
              alpha = alpha)
          gradients = tape.gradient(x_means ** 2, z_variable)
          unintegrated_gradients_batch.append(1 / (steps + 1) * batch_delta_x * gradients)
        
      unintegrated_gradients_batch = tf.stack(unintegrated_gradients_batch, axis=-1)
      integrated_gradients_batch = tf.reduce_sum(unintegrated_gradients_batch, axis=-1)
      integrated_gradients.append(tf.reshape(integrated_gradients_batch, (-1, n_latent_dims)))

    integrated_gradients = tf.concat(integrated_gradients, 0)
    
    return integrated_gradients
  
  def compute_attribution_target_batch(
      self,
      batch: Mapping[str, tf.Tensor],
      component: str,
      modality: str,
      target_component: str,
      selected_indices: Optional[Sequence[int]] = None,
      z_variable: tf.Variable = None,
      alpha: float = 1.0) -> np.ndarray:
    """Compute the means of generative data in each batch with selected 
    indices.

    Parameters
    ----------
    batch: Mapping[str, tf.Tensor]
        batch inputs.
    
    component: str
        generative result of `modality` from which component to used.
    
    modality: str
        which modality to used from the generative result of 
        `component`.
    
    dist_x_z_class: Distribution
        the class for the data distribution of `modality`.

    selected_indices: Optional[Sequence[int]], optional
        the integer indices for which variables to used. All variables
        will be used if provided with None. Defaults to None.

    Returns
    -------
    np.ndarray
        the means of the generative data distribution, where index i 
        specify the samples, index j specify the means of the data 
        distribution of the variables. 
    
    """
    z_conditional = dict()
    z_hat_conditional = dict()
    for component_config in self.model.component_configs:
      component_name = component_config.get('name')
      component_network = self.model.components.get(component_name)
      modality_names = component_config.get(Constants.CONFIG_FIELD_COMPONENT_MODALITY_NAMES)
      component_inputs = Model.prepare_component_inputs(
        batch=batch,
        component_config=component_config,
        target_component=component_name,
        components=self.model.components,
        z_conditional=z_conditional,
        z_hat_conditional=z_hat_conditional
      )
            
      preprocessor_inputs = Component.prepare_preprocessor_inputs(component_inputs, modality_names)
      preprocessor_outputs = component_network.preprocessor(preprocessor_inputs, training=False)
      z_parameters = component_network.encoder(
          preprocessor_outputs.get(component_network.preprocessor.matrix_key),
          training=False)
      if component_name == target_component and z_variable is not None:
        z = alpha * z_variable
      else:
        z_sampler = MultivariateNormalDiagSampler()
        z = alpha * z_sampler(z_parameters, training=False)

      hierarchical_encoder_inputs = Component.prepare_hierarchical_encoder_inputs(component_inputs, z)
      z_hat = component_network.hierarchical_encoder(hierarchical_encoder_inputs)
      
      z_conditional.setdefault(component_name, z)
      z_hat_conditional.setdefault(component_name, z_hat)

      if component_name == component:
        decoder_inputs = Component.prepare_decoder_inputs(
            batch=batch,
            modality_name=modality,
            z_hat=z_hat,
            preprocessor_outputs=dict())
        decoder = component_network.decoders.get(modality)
        attribution_target = decoder.compute_attribution_target(decoder_inputs)
        if selected_indices is not None:
          attribution_target = tf.gather(attribution_target, selected_indices, axis=-1)
        
        return attribution_target