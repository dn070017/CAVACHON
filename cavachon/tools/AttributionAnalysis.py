from cavachon.dataloader.DataLoader import DataLoader
from cavachon.distributions.Distribution import Distribution
from cavachon.environment.Constants import Constants
from cavachon.layers.parameterizers.MultivariateNormalDiagSampler import MultivariateNormalDiagSampler
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
      selected_variables: Optional[Sequence[str]] = None,
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
    
    selected_variables: Optional[Sequence[int]], optional
        the variables to used. The provided variables needs to match
        the indices of mdata[modality].var. All variables will be used 
        if provided with None. Defaults to None.

    batch_size : int, optional
        batch size used for the forward pass. Defaults to 128

    Returns
    -------
    np.ndarray
        x - x_baseline in the integrated gradients.

    """
    dist_x_z_name = self.model.components.get(component).distribution_names.get(modality)
    dist_x_z_class = ReflectionHandler.get_class_by_name(dist_x_z_name, 'distributions')
    if selected_variables is not None:
      selected_indices = [self.mdata[modality].var.index.get_loc(var) for var in selected_variables]
    else:
      selected_indices = None
    delta_x = []
    dataloader = DataLoader(self.mdata, batch_size=batch_size)
    progress_message = "Computing delta x"
    for batch in tqdm(dataloader.dataset.batch(batch_size), desc=progress_message):
      x_means_null_batch = self.compute_attribution_target_batch(
          batch_inputs=batch,
          component=component,
          modality=modality,
          target_component=exclude_component,
          selected_indices=selected_indices,
          alpha=0.0)
      
      x_means_full_batch = self.compute_attribution_target_batch(
          batch_inputs=batch,
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

    batch_size: int, optional
        batch size used for the forward pass. Defaults to 128

    Returns
    -------
    np.ndarray
        integrated gradients of ∂rho_m/∂z_m.

    """    
    delta_x = self.compute_delta_x(
        component = component,
        modality = modality,
        exclude_component = target_component,
        selected_variables = None,
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
              batch_inputs=batch,
              component=component,
              modality=modality,
              target_component=target_component,
              z_variable=z_variable,
              alpha=alpha)
          gradients = tape.gradient(x_means ** 2, z_variable)
          unintegrated_gradients_batch.append(1 / (steps + 1) * batch_delta_x * gradients)
        
      unintegrated_gradients_batch = tf.stack(unintegrated_gradients_batch, axis=-1)
      integrated_gradients_batch = tf.reduce_sum(unintegrated_gradients_batch, axis=-1)
      integrated_gradients.append(tf.reshape(integrated_gradients_batch, (-1, n_latent_dims)))

    integrated_gradients = tf.concat(integrated_gradients, 0)
    
    return integrated_gradients
  
  def compute_attribution_target_batch(
      self,
      batch_inputs: Mapping[str, tf.Tensor],
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
    batch_inputs: Mapping[str, tf.Tensor]
        input batched data from Dataloader.
    
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
      component_inputs = dict()
      component_name = component_config.get('name')
      component_network = self.model.components.get(component_name)
      modality_names = component_config.get(Constants.CONFIG_FIELD_COMPONENT_MODALITY_NAMES)
      for modality_name in modality_names:
        modality_matrix_key = f"{modality_name}/{Constants.TENSOR_NAME_X}"
        modality_batch_key = f'{modality_name}/{Constants.TENSOR_NAME_BATCH}'
        component_inputs.setdefault(modality_matrix_key, batch_inputs.get(modality_matrix_key))
        component_inputs.setdefault(modality_batch_key, batch_inputs.get(modality_batch_key))
      
      conditioned_on_keys = [
        Constants.MODULE_INPUTS_CONDITIONED_Z,
        Constants.MODULE_INPUTS_CONDITIONED_Z_HAT
      ]
      conditioned_on_config_keys = [
        Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z,
        Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT
      ]
      conditioned_on_dict = [
        z_conditional,
        z_hat_conditional
      ]
      conditioned_on_keys = zip(
          conditioned_on_keys, 
          conditioned_on_config_keys, 
          conditioned_on_dict)
      for conditioned_on_key, conditioned_on_config_key, conditioned_on_d in conditioned_on_keys:
        conditional = []
        conditioned_on = component_config.get(conditioned_on_config_key, [])
        if len(conditioned_on) != 0:
          for conditioned_on_component_name in conditioned_on:
            conditional.append(conditioned_on_d.get(conditioned_on_component_name))
          conditional = tf.concat(conditional, axis=-1)
          component_inputs.setdefault(
              conditioned_on_key,
              conditional)
      
      component_outputs = dict()
      hierarchical_encoder_inputs = dict()
      preprocessor_inputs = dict()
      preprocessor_inputs.update(component_inputs)
      preprocessor_inputs.pop(Constants.MODULE_INPUTS_CONDITIONED_Z, None)
      preprocessor_inputs.pop(Constants.MODULE_INPUTS_CONDITIONED_Z_HAT, None)

      for modality_name in modality_names:
        modality_batch_key = f'{modality_name}/{Constants.TENSOR_NAME_BATCH}'
        preprocessor_inputs.pop(modality_batch_key, None)

      preprocessor_outputs = component_network.preprocessor(preprocessor_inputs, training=False)
      z_parameters = component_network.encoder(
          preprocessor_outputs.get(component_network.preprocessor.matrix_key),
          training=False)
      
      if component_name == target_component and z_variable is not None:
        z = alpha * z_variable
      else:
        z_sampler = MultivariateNormalDiagSampler()
        z = alpha * z_sampler(z_parameters, training=False)

      hierarchical_encoder_inputs.setdefault(Constants.MODEL_OUTPUTS_Z, z)
      if Constants.MODULE_INPUTS_CONDITIONED_Z in component_inputs:
          hierarchical_encoder_inputs.setdefault(
              Constants.MODULE_INPUTS_CONDITIONED_Z,
              component_inputs.get(Constants.MODULE_INPUTS_CONDITIONED_Z))
      if Constants.MODULE_INPUTS_CONDITIONED_Z_HAT in component_inputs:
          hierarchical_encoder_inputs.setdefault(
              Constants.MODULE_INPUTS_CONDITIONED_Z_HAT,
              component_inputs.get(Constants.MODULE_INPUTS_CONDITIONED_Z_HAT))
      z_hat = component_network.hierarchical_encoder(hierarchical_encoder_inputs)

      component_outputs.setdefault(Constants.MODEL_OUTPUTS_Z, z)
      component_outputs.setdefault(Constants.MODEL_OUTPUTS_Z_HAT, z_hat)
      component_outputs.setdefault(Constants.MODEL_OUTPUTS_Z_PARAMS, z_parameters)
      
      z_conditional.setdefault(
          component_name, 
          component_outputs.get(Constants.MODEL_OUTPUTS_Z))

      z_hat_conditional.setdefault(
          component_name,
          component_outputs.get(Constants.MODEL_OUTPUTS_Z_HAT))

      if component_name == component:
        modality_batch_key = f'{modality}/{Constants.TENSOR_NAME_BATCH}'
        decoder_inputs = dict()
        decoder_inputs.setdefault(
            Constants.TENSOR_NAME_X,
            tf.concat([z_hat, component_inputs.get(modality_batch_key)], axis=-1))
        decoder = component_network.decoders.get(modality)
        attribution_target = decoder.compute_attribution_target(decoder_inputs)
        
        return attribution_target
  
  """def setup_hierarchical_encoder_inputs(
      self,
      batch_outputs: Mapping[str, tf.Tensor],
      component: str,
      z_variable: tf.Variable,
      alpha: float):

    component_network = self.model.components[component]

    hierarchical_encoder_inputs = dict()
    hierarchical_encoder_inputs.setdefault(
        Constants.MODEL_OUTPUTS_Z,
        alpha * z_variable)

    cond_on_keys = [
      Constants.MODEL_OUTPUTS_Z,
      Constants.MODEL_OUTPUTS_Z_HAT
    ]
    cond_on_inputs_keys = [
      Constants.MODULE_INPUTS_CONDITIONED_Z,
      Constants.MODULE_INPUTS_CONDITIONED_Z_HAT
    ]
    cond_on_components = [
      component_network.conditioned_on_z,
      component_network.conditioned_on_z_hat
    ]
    cond_on_iter = zip(cond_on_keys, cond_on_inputs_keys, cond_on_components)
    for cond_on_key, cond_on_inputs_key, cond_on_component_list in cond_on_iter:
      cond_on_tensor = list()
      for cond_on_component in cond_on_component_list:
        cond_on_tensor.append(batch_outputs.get(f'{cond_on_component}/{cond_on_key}'))
      if len(cond_on_tensor) != 0:
        hierarchical_encoder_inputs.setdefault(
            cond_on_inputs_key,
            tf.concat(cond_on_tensor, axis=-1))
    
    return hierarchical_encoder_inputs

  def compute_preprocessor_outputs_batch(
      self,
      batch: Mapping[str, tf.Tensor],
      component: str,
      modality: str) -> tf.Tensor:
    
    component_network = self.model.components[component]
    modality_names = self.mdata.mod.keys()
    
    preprocessor_inputs = dict()
    preprocessor_inputs.update(batch)
    preprocessor_inputs.pop(Constants.MODULE_INPUTS_CONDITIONED_Z, None)
    preprocessor_inputs.pop(Constants.MODULE_INPUTS_CONDITIONED_Z_HAT, None)
    for modality_name in modality_names:
      modality_batch_key = f'{modality_name}/{Constants.TENSOR_NAME_BATCH}'
      modality_matrix_key = f'{modality_name}/{Constants.TENSOR_NAME_X}'
      preprocessor_inputs.pop(modality_batch_key, None)
      if modality_name != modality:
        preprocessor_inputs.pop(modality_matrix_key, None)
    preprocessor_outputs = component_network.preprocessor(preprocessor_inputs)
  
    return preprocessor_outputs"""