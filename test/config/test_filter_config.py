import warnings

import pytest
from pydantic import ValidationError

from cavachon.config import FilterConfig


def test_filter_requires_step():
    with pytest.raises(ValidationError):
        FilterConfig()


def test_filter_keeps_arbitrary_extra_fields():
    f = FilterConfig(step="FilterCells", min_genes=10, inplace=True)
    assert f.step == "FilterCells"
    assert f.model_extra == {"min_genes": 10, "inplace": True}
    assert f.min_genes == 10


def test_step_only_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        FilterConfig(step="FilterQC")


# regression: filter step kwargs are expected extras and must not trigger
# the inherited "Unexpected field ... typo" RuntimeWarning.
@pytest.mark.parametrize(
    "params",
    [
        {"min_counts": 10},
        {"filter_threshold": 0.1},
        {"qc_vars": ["mt"]},
        {"min_counts": 10, "qc_vars": ["mt"]},
    ],
)
def test_valid_filter_parameters_do_not_warn(params):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        f = FilterConfig(step="FilterCells", **params)
    assert f.model_extra == params
