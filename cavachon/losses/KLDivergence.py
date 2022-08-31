from cavachon.distributions.Distribution import Distribution
from cavachon.distributions.MultivariateNormalDiag import MultivariateNormalDiag
from cavachon.distributions.MixtureMultivariateNormalDiag import MixtureMultivariateNormalDiag

import tensorflow as tf

class KLDivergence(tf.keras.losses.Loss):
  def __init__(self, name: str = 'KLDivergence',**kwargs):
    super().__init__(name=name, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, **kwargs)

  def call(self, y_true, y_pred):
    # Based on eq (C.48) from Falck et al., 2021. Here, we use y to denote c_j
    # logpx_z + 𝚺_j𝚺_y[py_z(logpz_y + logpy)] - 𝚺_j[logqz_x] - 𝚺_j𝚺_y[py_z(logpc_z)] 
    # logpx_z + 𝚺_j𝚺_y[py_z(logpz_y)] + 𝚺_j𝚺_y[py_z(logpy)] - 𝚺_j[logqz_x] - 𝚺_j𝚺_y[py_z(logpy_z)]
    # can be written as:
    #   (a)   +          (b)          +          (c)        +      (d)     +          (e)
    # or
    # LogDataLikelihood - NegativeKLDivergence (maximizing the ELBO)
    # or 
    # NegativeLogDataLikelihood + KLDivergence (minimizing the loss)

    event_dims = y_pred.shape[1] // 3
    z = y_pred[..., 0:event_dims]
    dist_z_x_params = y_pred[..., event_dims:]
    logits_prior = y_true[..., 0]
    
    # batch_shape: (batch, ), event_shape: (event_dims, )
    dist_z_x = MultivariateNormalDiag.from_parameterizer_output(dist_z_x_params)
    # the 1 here for dist_z_y and dist_z depends on the dimensionality of input tensor to the
    # parameterizers of MixtureMultivariateNormalDiag. In MFCVAE, this value should be one (one 
    # set of priors with n_componetns for each layer) 
    # batch_shape: (1, n_components), event_shape: (event_dims, )
    dist_z_y = MultivariateNormalDiag.from_parameterizer_output(y_true[..., 1:])
    # batch_shape: (1, ), event_shape: (event_dims, )
    dist_z = MixtureMultivariateNormalDiag.from_parameterizer_output(y_true)

    # change the shape of z from (batch, event_dims) to (batch, 1, event_dims) to make the 
    # operation broadcastable with batch shape (1, n_components) of dist_z_y
    # shape: (batch, n_components)
    logpz_y = dist_z_y.log_prob(tf.expand_dims(z, -2))
    # shape: (1, n_components)
    logpy = tf.math.log(tf.math.softmax(logits_prior) + 1e-7)
    # shape: (batch, 1)
    logpz = tf.expand_dims(dist_z.log_prob(z), -1)

    # shape: (batch, n_components)
    py_z = tf.math.softmax(logpz_y + logpy)
    logpy_z = tf.math.log(py_z + 1e-7)      
    #logpy_z = logpz_y + logpy - logpz
    #py_z = tf.exp(logpy_z)
    
    # term (b): 𝚺_j𝚺_y[py_z(logpz_y)]
    py_z_logpz_y = tf.reduce_sum(py_z * logpz_y, axis=-1)

    # term (c): 𝚺_j𝚺_y[py_z(logpy)]
    py_z_logpy = tf.reduce_sum(py_z * logpy, axis=-1)

    # term (d): 𝚺_j[logqz_x]
    logqz_x = dist_z_x.log_prob(z)
    
    # term (e): 𝚺_j𝚺_y[py_z(logpy_z)]
    py_z_logpy_z = tf.reduce_sum(py_z * logpy_z, axis=-1)

    kl_divergence = -py_z_logpz_y - py_z_logpy + py_z_logpy_z + logqz_x
    kl_divergence = tf.where(kl_divergence < 0, tf.zeros_like(kl_divergence), kl_divergence)

    return kl_divergence