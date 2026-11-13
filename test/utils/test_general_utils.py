import re
import signal
from types import SimpleNamespace

import pytest

from cavachon.environment.constants import Constants
from cavachon.utils.general_utils import GeneralUtils

Z = Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z
ZH = Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT


def names(components):
    return [c["name"] if isinstance(c, dict) else c.name for c in components]


def assert_topological(ordered):
    pos = {n: i for i, n in enumerate(names(ordered))}
    for c in ordered:
        get = c.get if isinstance(c, dict) else (lambda k, d=None, c=c: getattr(c, k, d))
        for parent in list(get(Z, [])) + list(get(ZH, [])):
            assert pos[parent] < pos[get("name")]


@pytest.fixture
def timeout():
    """Guard against infinite loops in order_components."""

    def handler(signum, frame):
        raise TimeoutError("order_components did not terminate")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(10)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old)


class TestOrderComponents:
    def test_independent_components_keep_order(self, timeout):
        cfg = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        assert names(GeneralUtils.order_components(cfg)) == ["a", "b", "c"]

    def test_parent_before_child(self, timeout):
        cfg = [{"name": "child", Z: ["parent"]}, {"name": "parent"}]
        out = GeneralUtils.order_components(cfg)
        assert names(out) == ["parent", "child"]

    def test_chain_and_z_hat(self, timeout):
        cfg = [
            {"name": "c", ZH: ["b"]},
            {"name": "b", Z: ["a"]},
            {"name": "a"},
        ]
        out = GeneralUtils.order_components(cfg)
        assert names(out) == ["a", "b", "c"]

    def test_diamond(self, timeout):
        cfg = [
            {"name": "d", Z: ["b"], ZH: ["c"]},
            {"name": "c", Z: ["a"]},
            {"name": "b", Z: ["a"]},
            {"name": "a"},
        ]
        out = GeneralUtils.order_components(cfg)
        assert sorted(names(out)) == ["a", "b", "c", "d"]
        assert_topological(out)

    def test_mapping_input_returns_values(self, timeout):
        cfg = {"x": {"tag": 1, Z: ["y"]}, "y": {"tag": 2}}
        out = GeneralUtils.order_components(cfg)
        assert [c["tag"] for c in out] == [2, 1]

    def test_object_configs(self, timeout):
        a = SimpleNamespace(name="a", conditioned_on_z=[], conditioned_on_z_hat=[])
        b = SimpleNamespace(name="b", conditioned_on_z=["a"], conditioned_on_z_hat=[])
        out = GeneralUtils.order_components([b, a])
        assert out[0] is a and out[1] is b

    def test_object_configs_missing_condition_attrs(self, timeout):
        a = SimpleNamespace(name="a")
        assert GeneralUtils.order_components([a]) == [a]

    def test_returns_same_objects_not_copies(self, timeout):
        cfg = [{"name": "a"}]
        assert GeneralUtils.order_components(cfg)[0] is cfg[0]

    def test_does_not_mutate_input(self, timeout):
        cfg = [{"name": "b", Z: ["a"]}, {"name": "a"}]
        snapshot = [dict(c) for c in cfg]
        GeneralUtils.order_components(cfg)
        assert cfg == snapshot

    def test_empty(self, timeout):
        assert GeneralUtils.order_components([]) == []

    def test_cycle_raises(self, timeout):
        cfg = [{"name": "a", Z: ["b"]}, {"name": "b", Z: ["a"]}]
        with pytest.raises(AttributeError, match="cyclic"):
            GeneralUtils.order_components(cfg)

    def test_self_loop_raises(self, timeout):
        with pytest.raises(AttributeError):
            GeneralUtils.order_components([{"name": "a", Z: ["a"]}])

    def test_unknown_parent_reports_name(self, timeout):
        # regression: failed in networkx with 'None cannot be a node'
        with pytest.raises(ValueError, match="'a' is conditioned on unknown component 'missing_parent'"):
            GeneralUtils.order_components([{"name": "a", Z: ["missing_parent"]}])

    def test_unknown_z_hat_parent_reports_name(self, timeout):
        cfg = [{"name": "p"}, {"name": "c", Z: ["p"], ZH: ["ghost"]}]
        with pytest.raises(ValueError, match="'c' is conditioned on unknown component 'ghost'"):
            GeneralUtils.order_components(cfg)

    def test_unknown_parent_in_mapping_input(self, timeout):
        with pytest.raises(ValueError, match="'x' is conditioned on unknown component 'nope'"):
            GeneralUtils.order_components({"x": {Z: ["nope"]}})

    def test_unknown_parent_on_object_config(self, timeout):
        a = SimpleNamespace(name="a", conditioned_on_z=["zz"], conditioned_on_z_hat=[])
        with pytest.raises(ValueError, match="unknown component 'zz'"):
            GeneralUtils.order_components([a])

    def test_duplicate_names_rejected(self, timeout):
        # regression: duplicates were accepted and edges hit the first one only
        cfg = [{"name": "a"}, {"name": "a"}, {"name": "b", Z: ["a"]}]
        with pytest.raises(ValueError, match="Duplicate component name 'a'"):
            GeneralUtils.order_components(cfg)

    def test_duplicate_names_on_object_configs(self, timeout):
        cfg = [SimpleNamespace(name="dup"), SimpleNamespace(name="dup")]
        with pytest.raises(ValueError, match="'dup'"):
            GeneralUtils.order_components(cfg)

    def test_error_paths_do_not_mutate_input(self, timeout):
        cfg = [{"name": "a", Z: ["ghost"]}, {"name": "b"}]
        import copy

        snapshot = copy.deepcopy(cfg)
        with pytest.raises(ValueError):
            GeneralUtils.order_components(cfg)
        assert cfg == snapshot

    def test_explicit_none_conditioned_on_raises_typeerror(self, timeout):
        # key present with value None (e.g. YAML 'conditioned_on_z:' left blank)
        with pytest.raises(TypeError):
            GeneralUtils.order_components([{"name": "a", Z: None}])


class TestTensorflowCompatibleStr:
    @pytest.mark.parametrize("s", ["abc", "a_b", "A.b-c/d\\e>f", "1abc", ".x"])
    def test_valid_unchanged(self, s):
        assert GeneralUtils.tensorflow_compatible_str(s) == s

    @pytest.mark.parametrize(
        "s,expected",
        [
            ("a b", "a_b"),
            ("a:b", "a_b"),
            ("a(b)", "a_b_"),
            ("géne", "g_ne"),
        ],
    )
    def test_invalid_chars_replaced(self, s, expected):
        assert GeneralUtils.tensorflow_compatible_str(s) == expected

    @pytest.mark.parametrize(
        "s,expected", [("_a", "t__a"), ("-a", "t_-a"), (" a", "t__a"), ("", "t_")]
    )
    def test_invalid_start_prefixed(self, s, expected):
        assert GeneralUtils.tensorflow_compatible_str(s) == expected

    @pytest.mark.parametrize("s", ["a b", "_x", "", "x:y", "géne", "1 2", "a\tb"])
    def test_output_always_valid(self, s):
        out = GeneralUtils.tensorflow_compatible_str(s)
        assert re.fullmatch(Constants.TENSORFLOW_NAME_REGEX, out)

    def test_idempotent(self):
        once = GeneralUtils.tensorflow_compatible_str("_a b")
        assert GeneralUtils.tensorflow_compatible_str(once) == once

    @pytest.mark.parametrize("s,expected", [("abc\n", "abc_"), ("_a\n", "t__a_"), ("a\nb", "a_b")])
    def test_trailing_newline_sanitized(self, s, expected):
        # regression: match() with '$' accepted 'abc\n' unchanged
        out = GeneralUtils.tensorflow_compatible_str(s)
        assert out == expected
        assert re.fullmatch(Constants.TENSORFLOW_NAME_REGEX, out)


class TestDuplicateObjToList:
    def test_deep_copies(self):
        obj = {"a": [1, 2]}
        out = GeneralUtils.duplicate_obj_to_list(obj, 3)
        assert out == [obj] * 3
        assert all(o is not obj for o in out)
        out[0]["a"].append(3)
        assert obj["a"] == [1, 2] and out[1]["a"] == [1, 2]

    @pytest.mark.parametrize("n", [0, -1])
    def test_zero_or_negative(self, n):
        assert GeneralUtils.duplicate_obj_to_list(1, n) == []

    def test_immutable(self):
        assert GeneralUtils.duplicate_obj_to_list("x", 2) == ["x", "x"]
