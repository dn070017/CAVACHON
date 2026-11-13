import pytest
from pydantic import ValidationError

from cavachon.config import FilterConfig, ModalityConfig


@pytest.mark.parametrize("missing", ["name", "type"])
def test_required_fields(missing):
    data = {"name": "RNA", "type": "rna"}
    data.pop(missing)
    with pytest.raises(ValidationError):
        ModalityConfig(**data)


def test_defaults():
    m = ModalityConfig(name="RNA", type="rna")
    assert m.samples == [] and m.filters == [] and m.batch_effect_colnames == []
    assert m.h5ad == ""


@pytest.mark.parametrize(
    "type_, dist",
    [
        ("rna", "IndependentZeroInflatedNegativeBinomial"),
        ("RNA", "IndependentZeroInflatedNegativeBinomial"),
        ("Atac", "IndependentBernoulli"),
    ],
)
def test_type_lowercased_and_default_dist(type_, dist):
    m = ModalityConfig(name="m", type=type_)
    assert m.type == type_.lower()
    assert m.dist == dist


def test_explicit_dist_kept():
    m = ModalityConfig(name="m", type="rna", dist="IndependentBernoulli")
    assert m.dist == "IndependentBernoulli"


def test_name_sanitized():
    assert ModalityConfig(name="my rna", type="rna").name == "my_rna"


def test_filters_parsed():
    m = ModalityConfig(
        name="m", type="rna", filters=[{"step": "FilterCells", "min_genes": 5}]
    )
    assert isinstance(m.filters[0], FilterConfig)
    assert m.filters[0].min_genes == 5


def test_samples_not_shared_between_instances():
    a = ModalityConfig(name="a", type="rna")
    b = ModalityConfig(name="b", type="rna")
    a.samples.append("s1")
    assert b.samples == []


# regression: unknown type without dist raises
def test_unknown_type_without_dist_rejected():
    with pytest.raises(ValidationError, match="Unknown type 'protein' of modality prot"):
        ModalityConfig(name="prot", type="protein")


def test_unknown_type_with_explicit_dist_allowed():
    m = ModalityConfig(name="prot", type="Protein", dist="IndependentBernoulli")
    assert (m.type, m.dist) == ("protein", "IndependentBernoulli")
