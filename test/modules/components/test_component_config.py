"""Keras 3 serialization surface of Component (get_config / from_config)
and handling of standard Keras constructor kwargs.

Component defines neither get_config nor from_config; these tests pin
down what the inherited Keras 3 implementations do. Mocked: the same
fakes as the other files (see conftest.py).
"""

import pytest

from cavachon.modules.components.component import Component

from .conftest import make_component


def test_get_config_only_has_base_keras_fields(fakes):
    c = make_component(name="c", z_dims=2)
    config = c.get_config()
    assert set(config) == {"name", "trainable", "dtype"}
    assert config["name"] == "c"
    assert config["trainable"] is True
    # nothing needed to rebuild the component is serialized
    for key in (
        "modality_names",
        "distribution_names",
        "conditioned_on_z",
        "learn_z_hat_priors",
    ):
        assert key not in config


def test_from_config_cannot_rebuild_component(fakes):
    """Round trip is impossible: the required constructor arguments are
    absent from get_config(). (.keras saving is unsupported; the project
    only uses save_weights.)"""
    c = make_component(name="c")
    with pytest.raises(TypeError, match="inputs"):
        Component.from_config(c.get_config())


def test_make_swallows_unknown_kwargs(fakes):
    c = make_component(name="c", save_x={"rna": True}, save_z={"c": True})
    assert c.name == "c"
    assert not hasattr(c, "save_x") and not hasattr(c, "save_z")


@pytest.mark.xfail(
    strict=True,
    reason="BUG: Component.__init__ drops **kwargs, so Keras kwargs such as "
    "trainable=False (as emitted by get_config) are silently ignored",
)
def test_keras_trainable_kwarg_is_honoured(fakes):
    c = make_component(name="c", trainable=False)
    assert c.trainable is False
