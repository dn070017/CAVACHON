import pytest

from cavachon.environment.constants import Constants
from cavachon.utils.general_utils import GeneralUtils


@pytest.fixture
def component_configs_simple_linear():
    return {
        "component_a": {"name": "component_a"},
        "component_b": {
            "name": "component_b",
            Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z: ["component_a"],
        },
        "component_c": {
            "name": "component_c",
            Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT: ["component_b"],
        },
    }


@pytest.fixture
def component_configs_dag():
    return {
        "component_a": {"name": "component_a"},
        "component_b": {"name": "component_b"},
        "component_c": {
            "name": "component_c",
            Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z: [
                "component_a",
                "component_b",
            ],
        },
        "component_d": {
            "name": "component_d",
            Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT: ["component_c"],
        },
        "component_e": {
            "name": "component_e",
            Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z: ["component_b"],
        },
    }


@pytest.fixture
def component_configs_cyclic():
    return {
        "component_a": {
            "name": "component_a",
            Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z: ["component_c"],
        },
        "component_b": {
            "name": "component_b",
            Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z: ["component_a"],
        },
        "component_c": {
            "name": "component_c",
            Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT: ["component_b"],
        },
    }


@pytest.fixture
def component_configs_no_deps():
    return {
        "component_a": {"name": "component_a"},
        "component_b": {"name": "component_b"},
        "component_c": {"name": "component_c"},
    }


def test_order_components_simple_linear(component_configs_simple_linear):
    ordered_components = GeneralUtils.order_components(component_configs_simple_linear)
    ordered_names = [comp["name"] for comp in ordered_components]
    # Expected order: a -> b -> c (reverse topological sort)
    assert ordered_names == ["component_a", "component_b", "component_c"]


def test_order_components_dag(component_configs_dag):
    ordered_components = GeneralUtils.order_components(component_configs_dag)
    ordered_names = [comp["name"] for comp in ordered_components]
    # Possible valid orders (reverse topological sort):
    # [a, b, e, c, d] or [b, a, e, c, d] or [a, b, c, e, d] etc.
    # Check dependencies: c depends on a, b; d depends on c; e depends on b
    assert ordered_names.index("component_a") < ordered_names.index("component_c")
    assert ordered_names.index("component_b") < ordered_names.index("component_c")
    assert ordered_names.index("component_c") < ordered_names.index("component_d")
    assert ordered_names.index("component_b") < ordered_names.index("component_e")
    assert len(ordered_names) == 5


def test_order_components_cyclic(component_configs_cyclic):
    with pytest.raises(AttributeError, match="directed cyclic graph"):
        GeneralUtils.order_components(component_configs_cyclic)


def test_order_components_empty():
    ordered_components = GeneralUtils.order_components({})
    assert ordered_components == []


def test_order_components_no_deps(component_configs_no_deps):
    ordered_components = GeneralUtils.order_components(component_configs_no_deps)
    ordered_names = [comp["name"] for comp in ordered_components]
    # Order doesn't matter, but all components should be present
    assert set(ordered_names) == {"component_a", "component_b", "component_c"}
    assert len(ordered_names) == 3


@pytest.mark.parametrize(
    "input_str, lower, expected_str",
    [
        ("valid_name", True, "valid_name"),
        ("ValidName123", True, "validname123"),
        ("name.with/dots-and_hyphens", True, "name.with/dots-and_hyphens"),
        ("invalid name", True, "invalid_name"),
        ("name_with_!@#$", True, "name_with_____"),
        ("1_starts_with_number", True, "t_1_starts_with_number"),
        ("-starts_with_hyphen", True, "t_-starts_with_hyphen"),
        ("_starts_with_underscore", True, "_starts_with_underscore"),
        ("", True, "t_"),
        ("a", True, "a"),
        ("a-b/c_d.e>f", True, "a-b/c_d.e>f"),
        ("a b c", True, "a_b_c"),
        ("UPPER_CASE", True, "upper_case"),
        ("valid_name", False, "valid_name"),
        ("ValidName123", False, "ValidName123"),
        ("name.with/dots-and_hyphens", False, "name.with/dots-and_hyphens"),
        ("invalid name", False, "invalid_name"),
        ("name_with_!@#$", False, "name_with_____"),
        ("1_starts_with_number", False, "t_1_starts_with_number"),
        ("-starts_with_hyphen", False, "t_-starts_with_hyphen"),
        ("_starts_with_underscore", False, "_starts_with_underscore"),
        ("", False, "t_"),
        ("a", False, "a"),
        ("a-b/c_d.e>f", False, "a-b/c_d.e>f"),
        ("a b c", False, "a_b_c"),
        ("UPPER_CASE", False, "UPPER_CASE"),
    ],
)
def test_tensorflow_compatible_str(input_str, lower, expected_str):
    assert (
        GeneralUtils.convert_to_tensorflow_compatible_string(input_str, lower=lower)
        == expected_str
    )


@pytest.mark.parametrize(
    "obj, n_objs, expected_type, check_identity",
    [
        (5, 3, int, True),
        ("test", 2, str, True),
        (
            [1, 2],
            4,
            list,
            False,
        ),
        ({"a": 1}, 1, dict, False),
        (None, 5, type(None), True),
        (True, 0, bool, True),
    ],
)
def test_duplicate_obj_to_list(obj, n_objs, expected_type, check_identity):
    result_list = GeneralUtils.duplicate_obj_to_list(obj, n_objs)
    assert len(result_list) == n_objs
    if n_objs > 0:
        assert all(isinstance(item, expected_type) for item in result_list)
        assert all(item == obj for item in result_list)  # Check value equality
        if n_objs > 1 and not check_identity:
            # For mutable types, check they are different objects (deep copies)
            assert result_list[0] is not result_list[1]
    else:
        assert result_list == []


def test_duplicate_obj_to_list_zero_duplicates():
    assert GeneralUtils.duplicate_obj_to_list("anything", 0) == []


def test_duplicate_obj_to_list_duplicate():
    original_obj = {"key": [1, 2]}
    result_list = GeneralUtils.duplicate_obj_to_list(original_obj, 3)
    assert len(result_list) == 3
    for item in result_list:
        assert item == original_obj
        assert item is not original_obj
