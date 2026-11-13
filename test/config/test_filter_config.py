import warnings

import pytest
from pydantic import ValidationError

from cavachon.config import FilterConfig


def test_filter_requires_step():
    with pytest.raises(ValidationError):
        FilterConfig()


def test_filter_keeps_arbitrary_extra_fields():
    with pytest.warns(RuntimeWarning):
        f = FilterConfig(step="FilterCells", min_genes=10, inplace=True)
    assert f.step == "FilterCells"
    assert f.model_extra == {"min_genes": 10, "inplace": True}
    assert f.min_genes == 10


@pytest.mark.parametrize(
    "params",
    [
        {"min_counts": 10},
        {"filter_threshold": 0.1},
        {"qc_vars": ["mt"]},
    ],
)
def test_every_real_filter_parameter_warns_current_behaviour(params):
    # Pins current behaviour: every filter parameter is an undeclared
    # extra, so BaseConfigModel's "Unexpected field ... typo" warning fires
    # for every valid filter config.
    with pytest.warns(RuntimeWarning, match=f"Unexpected field {next(iter(params))} in FilterConfig"):
        FilterConfig(step="FilterCells", **params)


def test_step_only_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        FilterConfig(step="FilterQC")


@pytest.mark.xfail(
    strict=True,
    reason="BUG: FilterConfig is designed to take arbitrary step parameters "
    "(extra='allow') but inherits the typo RuntimeWarning, so every valid "
    "filter config warns",
)
def test_valid_filter_parameters_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        FilterConfig(step="FilterCells", min_counts=10, qc_vars=["mt"])
