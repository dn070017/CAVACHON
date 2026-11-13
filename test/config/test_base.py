import warnings

import pytest
from pydantic import BaseModel

import cavachon.config as config_pkg
from cavachon.config import BaseConfigModel, TensorflowCompatibleStr


class _Model(BaseConfigModel):
    name: TensorflowCompatibleStr = "default"
    value: int = 0


def test_package_exports_all_names():
    for name in config_pkg.__all__:
        assert hasattr(config_pkg, name), name


def test_base_is_pydantic_model():
    assert issubclass(BaseConfigModel, BaseModel)


def test_known_fields_emit_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = _Model(name="abc", value=1)
    assert m.value == 1


def test_unknown_field_warns_and_is_kept():
    with pytest.warns(RuntimeWarning, match="Unexpected field typo_field in _Model"):
        m = _Model(typo_field=3)
    assert m.model_extra == {"typo_field": 3}


def test_each_unknown_field_warns_once():
    with pytest.warns(RuntimeWarning) as record:
        _Model(a=1, b=2)
    messages = [str(r.message) for r in record if r.category is RuntimeWarning]
    assert sum("field a " in m for m in messages) == 1
    assert sum("field b " in m for m in messages) == 1


def test_tensorflow_compatible_str_delegates_to_general_utils(general_utils):
    general_utils.tensorflow_compatible_str.side_effect = lambda s: f"tf[{s}]"
    m = _Model(name="raw name")
    general_utils.tensorflow_compatible_str.assert_called_once_with("raw name")
    assert m.name == "tf[raw name]"


def test_tensorflow_compatible_str_runs_after_str_validation(general_utils):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _Model(name=123)
    general_utils.tensorflow_compatible_str.assert_not_called()


def test_tensorflow_compatible_str_default_not_sanitized(general_utils):
    class _M(BaseConfigModel):
        name: TensorflowCompatibleStr = "bad name"

    # Pydantic does not validate defaults, so defaults bypass the sanitizer.
    assert _M().name == "bad name"
    general_utils.tensorflow_compatible_str.assert_not_called()
