from pathlib import Path
from types import SimpleNamespace

import pytest

from cavachon.utils.reflection_handler import ReflectionHandler


class TestCamelToSnake:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("IndependentBernoulli", "independent_bernoulli"),
            (
                "IndependentZeroInflatedNegativeBinomial",
                "independent_zero_inflated_negative_binomial",
            ),
            ("Distribution", "distribution"),
            ("lowercase", "lowercase"),
            ("", ""),
            ("ZINB", "zinb"),
            ("IndependentZINB", "independent_zinb"),
            ("already_snake", "already_snake"),
            ("Log1p", "log1p"),
        ],
    )
    def test_conversion(self, name, expected):
        assert ReflectionHandler._camel_to_snake(name) == expected

    def test_digit_followed_by_upper_not_split(self):
        # documents current behaviour: only [a-z][A-Z] boundaries split
        assert ReflectionHandler._camel_to_snake("Layer2Norm") == "layer2norm"

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: acronym followed by a word (e.g. 'ZINBDistribution') is not "
        "split, producing 'zinbdistribution' instead of 'zinb_distribution'",
    )
    def test_acronym_followed_by_word(self):
        assert (
            ReflectionHandler._camel_to_snake("ZINBDistribution") == "zinb_distribution"
        )


SRC = "/fake/root/cavachon"


class TestGetFilenames:
    def test_first_depth_pattern_and_hit(self, reflection_env):
        hit = f"{SRC}/distributions/foo_bar_distribution.py"
        reflection_env.glob.hits[f"{SRC}/distributions**/foo_bar_distribution.py"] = [
            hit
        ]
        out = ReflectionHandler._get_filenames(
            "foo_bar_distribution.py", "distributions"
        )
        assert out == [Path(hit)]
        assert reflection_env.glob.calls == [
            f"{SRC}/distributions**/foo_bar_distribution.py"
        ]

    def test_falls_through_to_second_depth(self, reflection_env):
        hit = f"{SRC}/layers/parameterizers/foo_layer.py"
        reflection_env.glob.hits[f"{SRC}/**/**/foo_layer.py"] = [hit]
        out = ReflectionHandler._get_filenames("foo_layer.py")
        assert out == [Path(hit)]
        assert reflection_env.glob.calls == [
            f"{SRC}/**/foo_layer.py",
            f"{SRC}/**/**/foo_layer.py",
        ]

    def test_nested_subdirectory_pattern(self, reflection_env):
        ReflectionHandler._get_filenames("x.py", "layers/parameterizers")
        assert reflection_env.glob.calls[0] == f"{SRC}/layers/parameterizers**/x.py"

    def test_not_found_returns_empty(self, reflection_env):
        assert ReflectionHandler._get_filenames("nope.py") == []
        assert len(reflection_env.glob.calls) == 2  # default max_depth=2

    def test_max_depth_controls_number_of_patterns(self, reflection_env):
        ReflectionHandler._get_filenames("x.py", max_depth=3)
        assert reflection_env.glob.calls[-1] == f"{SRC}/**/**/**/x.py"
        assert ReflectionHandler._get_filenames("x.py", max_depth=0) == []

    def test_pattern_never_targets_src_root_itself(self, reflection_env):
        # at least one '**/' level is always inserted, so a file directly in
        # src_path is never globbed for
        ReflectionHandler._get_filenames("top.py")
        assert f"{SRC}/top.py" not in reflection_env.glob.calls

    def test_subdirectory_glued_to_wildcard(self, reflection_env):
        # documents current behaviour: subdirectory is concatenated with '**'
        # without a separator, so 'dist' prefix-matches 'distributions/'
        ReflectionHandler._get_filenames("x.py", "dist")
        assert reflection_env.glob.calls[0] == f"{SRC}/dist**/x.py"

    def test_partial_prefixes_wildcard(self, reflection_env):
        ReflectionHandler._get_filenames("bar.py", partial=True)
        assert reflection_env.glob.calls[0] == f"{SRC}/**/*bar.py"

    def test_returns_all_hits_at_shallowest_depth_sorted(self, reflection_env):
        pattern = f"{SRC}/**/dup.py"
        reflection_env.glob.hits[pattern] = [f"{SRC}/bbb/dup.py", f"{SRC}/aaa/dup.py"]
        reflection_env.glob.hits[f"{SRC}/**/**/dup.py"] = [f"{SRC}/c/d/dup.py"]
        assert ReflectionHandler._get_filenames("dup.py") == [
            Path(f"{SRC}/aaa/dup.py"),
            Path(f"{SRC}/bbb/dup.py"),
        ]
        assert reflection_env.glob.calls == [pattern]

    def test_deeper_duplicate_shadowed_by_shallower_hit(self, reflection_env):
        # only the shallowest depth with a hit is considered
        reflection_env.glob.hits[f"{SRC}/**/x.py"] = [f"{SRC}/a/x.py"]
        reflection_env.glob.hits[f"{SRC}/**/**/x.py"] = [f"{SRC}/b/c/x.py"]
        assert ReflectionHandler._get_filenames("x.py") == [Path(f"{SRC}/a/x.py")]

    def test_patterns_built_from_settings_not_cwd(self, reflection_env, monkeypatch):
        import os

        monkeypatch.setattr(os, "getcwd", lambda: "/somewhere/else")
        ReflectionHandler._get_filenames("x.py")
        assert all(c.startswith(SRC + "/") for c in reflection_env.glob.calls)

    def test_follows_patched_settings(self, reflection_env, fake_settings):
        fake_settings.src_path = Path("/other/cavachon")
        ReflectionHandler._get_filenames("x.py")
        assert reflection_env.glob.calls[0] == "/other/cavachon/**/x.py"


class FakeDistribution:
    pass


class TestImportClass:
    def test_module_path_from_last_cavachon_part(self, fake_import):
        fake_import.modules["cavachon.layers.foo"] = SimpleNamespace(
            Foo=FakeDistribution
        )
        cls = ReflectionHandler._import_class(
            Path("/home/cavachon/cavachon/layers/foo.py"), "Foo"
        )
        assert cls is FakeDistribution
        assert fake_import.calls == ["cavachon.layers.foo"]

    def test_class_missing_in_module(self, fake_import):
        fake_import.modules["cavachon.environment.settings"] = SimpleNamespace()
        with pytest.raises(AttributeError):
            ReflectionHandler._import_class(
                Path("/x/cavachon/environment/settings.py"), "DoesNotExist"
            )

    def test_path_without_cavachon_component(self, fake_import):
        with pytest.raises(ValueError):
            ReflectionHandler._import_class(Path("/x/y/z.py"), "Z")


class TestGetClassByName:
    def _register(self, env, rel_file, module, attr, cls):
        env.glob.hits[f"{SRC}/{rel_file[0]}"] = [f"{SRC}/{rel_file[1]}"]
        env.importer.modules[module] = SimpleNamespace(**{attr: cls})

    def test_lookup_with_postfix(self, reflection_env):
        self._register(
            reflection_env,
            (
                "distributions**/independent_bernoulli_distribution.py",
                "distributions/independent_bernoulli_distribution.py",
            ),
            "cavachon.distributions.independent_bernoulli_distribution",
            "IndependentBernoulliDistribution",
            FakeDistribution,
        )
        cls = ReflectionHandler.get_class_by_name(
            "IndependentBernoulli", "distributions", "Distribution"
        )
        assert cls is FakeDistribution
        assert reflection_env.importer.calls == [
            "cavachon.distributions.independent_bernoulli_distribution"
        ]

    def test_lookup_camel_case_postfix(self, reflection_env):
        self._register(
            reflection_env,
            (
                "dataloader/modifiers**/foo_data_modifier.py",
                "dataloader/modifiers/foo_data_modifier.py",
            ),
            "cavachon.dataloader.modifiers.foo_data_modifier",
            "FooDataModifier",
            FakeDistribution,
        )
        cls = ReflectionHandler.get_class_by_name(
            "Foo", "dataloader/modifiers", "DataModifier"
        )
        assert cls is FakeDistribution

    def test_lookup_without_postfix(self, reflection_env):
        self._register(
            reflection_env,
            ("**/my_thing.py", "pkg/my_thing.py"),
            "cavachon.pkg.my_thing",
            "MyThing",
            FakeDistribution,
        )
        assert ReflectionHandler.get_class_by_name("MyThing") is FakeDistribution

    def test_not_found_raises_assertion_with_filename(self, reflection_env):
        with pytest.raises(AssertionError, match="no_such_thing_distribution.py"):
            ReflectionHandler.get_class_by_name(
                "NoSuchThing", "distributions", "Distribution"
            )
        assert reflection_env.importer.calls == []

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: lookup validation relies on assert (stripped under python -O, "
        "where a miss becomes IndexError); a missing class should raise a real "
        "exception such as FileNotFoundError/ImportError/ValueError",
    )
    def test_not_found_raises_non_assert_exception(self, reflection_env):
        with pytest.raises((FileNotFoundError, ImportError, ValueError)):
            ReflectionHandler.get_class_by_name("NoSuchThing", "environment")

    def test_ambiguous_lookup_rejected(self, reflection_env):
        # regression: previously glob()[0] was taken silently
        reflection_env.glob.hits[f"{SRC}/**/dup.py"] = [
            f"{SRC}/bbb/dup.py",
            f"{SRC}/aaa/dup.py",
        ]
        reflection_env.importer.modules["cavachon.aaa.dup"] = SimpleNamespace(
            Dup=FakeDistribution
        )
        reflection_env.importer.modules["cavachon.bbb.dup"] = SimpleNamespace(
            Dup=object
        )
        with pytest.raises(
            ValueError, match="more than one file named dup.py"
        ) as excinfo:
            ReflectionHandler.get_class_by_name("Dup")
        assert f"{SRC}/aaa/dup.py" in str(excinfo.value)
        assert f"{SRC}/bbb/dup.py" in str(excinfo.value)
        assert reflection_env.importer.calls == []

    def test_ambiguity_resolved_by_subdirectory(self, reflection_env):
        reflection_env.glob.hits[f"{SRC}/aaa**/dup.py"] = [f"{SRC}/aaa/dup.py"]
        reflection_env.importer.modules["cavachon.aaa.dup"] = SimpleNamespace(
            Dup=FakeDistribution
        )
        assert ReflectionHandler.get_class_by_name("Dup", "aaa") is FakeDistribution
