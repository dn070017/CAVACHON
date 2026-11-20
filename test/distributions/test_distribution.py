import pytest
import tensorflow_probability as tfp

import cavachon.distributions as dists
from cavachon.distributions.distribution import Distribution


def test_distribution_is_abstract():
    with pytest.raises(TypeError):
        Distribution()


def test_subclass_without_implementation_cannot_be_instantiated():
    class Incomplete(Distribution):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_subclass_with_implementation_can_be_instantiated():
    class Complete(Distribution):
        @classmethod
        def from_parameterizer_output(cls, params, **kwargs):
            return params

    assert Complete.from_parameterizer_output(3) == 3
    Complete()


@pytest.mark.parametrize(
    "name",
    [
        "IndependentBernoulliDistribution",
        "IndependentZeroInflatedNegativeBinomialDistribution",
        "MultivariateNormalDiagDistribution",
        "StudenttDistribution",
    ],
)
def test_package_exports_concrete_distributions(name):
    cls = getattr(dists, name)
    assert issubclass(cls, Distribution)
    assert issubclass(cls, tfp.distributions.Distribution)


def test_package_exports_mixture_distribution():
    # regression: fixed bug, see test_report/UNIT_TEST_REPORT.md
    assert hasattr(dists, "MixtureMultivariateNormalDiagDistribution")
