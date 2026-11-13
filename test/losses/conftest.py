"""Shared fakes for cavachon.losses unit tests.

Everything imported by cavachon.losses from OTHER cavachon packages
(cavachon.layers.ProgressiveScaler, cavachon.distributions.*,
cavachon.utils.ReflectionHandler) is replaced here. The fakes build
plain tensorflow_probability distributions, which are third-party and
serve as numeric references.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

LOSS_MODULES = (
    "cavachon.losses.gmm_density_loss",
    "cavachon.losses.gmm_kl_divergence",
    "cavachon.losses.negative_log_data_likelihood",
    "cavachon.losses.standard_kl_divergence",
)


class FakeProgressiveScaler:
    """Callable stand-in for cavachon.layers.ProgressiveScaler.

    ``fake(x)`` returns ``value * x`` where ``value`` starts at
    ``scale``. Deliberately has no arithmetic dunders, so it is (like a
    Keras Layer) not usable as a tensor operand.
    """

    instances = []

    def __init__(self, total_iterations=5000, scale=1.0, name="progressive_scaler"):
        self.total_iterations = total_iterations
        self.scale = scale
        self.value = float(scale)
        self.call_args = []
        FakeProgressiveScaler.instances.append(self)

    def __call__(self, inputs, training=False, **kwargs):
        self.call_args.append(inputs)
        return self.value * inputs

    def pin_to(self, value):
        self.value = float(value)

    def numpy(self):
        return self.value


def _fake_mvn_from_params(params, **kwargs):
    loc, scale = tf.split(params, 2, axis=-1)
    return tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale)


def _fake_mixture_from_params(params, **kwargs):
    logits = params[..., 0]
    loc, scale = tf.split(params[..., 1:], 2, axis=-1)
    return tfp.distributions.MixtureSameFamily(
        tfp.distributions.Categorical(logits=logits),
        tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale),
    )


def _fake_bernoulli_from_params(params, **kwargs):
    return tfp.distributions.Bernoulli(logits=params)


def make_fake_distribution_class(side_effect):
    """A MagicMock 'class' exposing from_parameterizer_output."""
    cls = MagicMock(name="FakeDistributionClass")
    cls.from_parameterizer_output.side_effect = side_effect
    return cls


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    """Patch every cross-package dependency of cavachon.losses where it is
    looked up, and hand the fakes to the test."""
    FakeProgressiveScaler.instances = []
    mvn = make_fake_distribution_class(_fake_mvn_from_params)
    mixture = make_fake_distribution_class(_fake_mixture_from_params)
    bernoulli = make_fake_distribution_class(_fake_bernoulli_from_params)
    get_class_by_name = MagicMock(name="get_class_by_name", return_value=bernoulli)

    for mod in LOSS_MODULES:
        monkeypatch.setattr(f"{mod}.ProgressiveScaler", FakeProgressiveScaler)
    monkeypatch.setattr(
        "cavachon.losses.gmm_density_loss.MixtureMultivariateNormalDiagDistribution",
        mixture,
    )
    monkeypatch.setattr(
        "cavachon.losses.gmm_kl_divergence.MixtureMultivariateNormalDiagDistribution",
        mixture,
    )
    monkeypatch.setattr(
        "cavachon.losses.gmm_kl_divergence.MultivariateNormalDiagDistribution",
        mvn,
    )
    monkeypatch.setattr(
        "cavachon.losses.negative_log_data_likelihood.ReflectionHandler.get_class_by_name",
        get_class_by_name,
    )
    return SimpleNamespace(
        scaler_cls=FakeProgressiveScaler,
        mvn=mvn,
        mixture=mixture,
        bernoulli=bernoulli,
        get_class_by_name=get_class_by_name,
    )
