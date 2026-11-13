from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from cavachon.filter.anndata_filter_handler import AnnDataFilterHandler
from cavachon.filter.filter_cells import FilterCells
from cavachon.filter.filter_genes import FilterGenes
from cavachon.filter.filter_qc import FilterQC
from test.filter.conftest import filter_step_config as fc
from test.filter.conftest import make_adata


def config_of(filters):
    # Stand-in for ApplicationConfig: the handler only reads `.filter`.
    return SimpleNamespace(filter=filters)


def test_from_config_looks_up_each_step_in_filter_subdir(mock_get_class_by_name):
    AnnDataFilterHandler.from_config(
        config_of(
            {
                "RNA": [fc("FilterCells", min_counts=1), fc("FilterGenes")],
                "ATAC": [fc("FilterQC", filter_threshold=[])],
            }
        )
    )
    assert mock_get_class_by_name.call_args_list == [
        call("FilterCells", "filter"),
        call("FilterGenes", "filter"),
        call("FilterQC", "filter"),
    ]


def test_from_config_builds_steps(mock_get_class_by_name):
    config = config_of(
        {
            "RNA": [
                fc("FilterCells", min_counts=1),
                fc("FilterGenes", min_cells=3),
            ],
            "ATAC": [],
        }
    )
    handler = AnnDataFilterHandler.from_config(config)
    assert set(handler.steps) == {"RNA", "ATAC"}
    rna = handler.steps["RNA"]
    assert [type(s) for s in rna] == [FilterCells, FilterGenes]
    assert rna[0].name == "FilterCells"
    assert rna[0].kwargs == {"min_counts": 1, "inplace": True}
    assert rna[1].kwargs == {"min_cells": 3, "inplace": True}
    assert handler.steps["ATAC"] == []


def test_from_config_passes_name_and_dumped_params_to_step_class(
    mock_get_class_by_name,
):
    fake_cls = MagicMock(name="FakeStep")
    mock_get_class_by_name.side_effect = None
    mock_get_class_by_name.return_value = fake_cls
    handler = AnnDataFilterHandler.from_config(
        config_of({"RNA": [fc("Anything", a=1, b=[2])]})
    )
    fake_cls.assert_called_once_with(name="Anything", step="Anything", a=1, b=[2])
    assert handler.steps["RNA"] == [fake_cls.return_value]


def test_from_config_qc_kwargs_pass_through(mock_get_class_by_name):
    criteria = [{"field": "total_counts", "operator": "gt", "threshold": 6}]
    handler = AnnDataFilterHandler.from_config(
        config_of(
            {
                "RNA": [
                    fc(
                        "FilterQC",
                        filter_threshold=criteria,
                        qc_vars=["MT-"],
                        percent_top=None,
                    )
                ]
            }
        )
    )
    step = handler.steps["RNA"][0]
    assert isinstance(step, FilterQC)
    assert step.kwargs["filter_threshold"] == criteria
    out = handler({"RNA": make_adata()})
    assert list(out["RNA"].obs_names) == ["c3"]


def test_from_config_misspelled_step_raises(mock_get_class_by_name):
    with pytest.raises(AssertionError):
        AnnDataFilterHandler.from_config(config_of({"RNA": [fc("FilterCel")]}))
    mock_get_class_by_name.assert_called_once_with("FilterCel", "filter")


def test_from_config_empty_filter(mock_get_class_by_name):
    handler = AnnDataFilterHandler.from_config(config_of({}))
    assert handler.steps == {}
    mock_get_class_by_name.assert_not_called()


def test_call_orchestrates_fake_steps_in_order():
    order = []

    def fake_step(tag, result):
        step = MagicMock(name=tag)

        def run(adata):
            order.append((tag, adata))
            return result

        step.side_effect = run
        return step

    a0, a1, a2 = object(), object(), object()
    handler = AnnDataFilterHandler(
        {"RNA": [fake_step("s1", a1), fake_step("s2", a2)]}
    )
    out = handler({"RNA": a0})
    assert order == [("s1", a0), ("s2", a1)]
    assert out["RNA"] is a2


@pytest.mark.parametrize("is_sparse", [False, True])
def test_call_applies_steps_in_order_per_modality(is_sparse):
    handler = AnnDataFilterHandler(
        {
            "RNA": [FilterCells("a", min_counts=1), FilterGenes("b", min_counts=5)],
            "ATAC": [FilterCells("c", min_genes=4)],
        }
    )
    target = {"RNA": make_adata(is_sparse), "ATAC": make_adata(is_sparse, "ATAC")}
    out = handler(target)
    assert out is target
    assert list(out["RNA"].obs_names) == ["c0", "c2", "c3"]
    assert list(out["RNA"].var_names) == ["GAPDH", "ACTB"]
    assert list(out["ATAC"].obs_names) == ["c3"]
    assert out["ATAC"].n_vars == 4


def test_call_replaces_target_value_with_qc_copy():
    original = make_adata()
    handler = AnnDataFilterHandler(
        {
            "RNA": [
                FilterQC(
                    "q",
                    percent_top=None,
                    qc_vars=["MT-"],
                    filter_threshold=[
                        {"field": "total_counts", "operator": "ge", "threshold": 6}
                    ],
                )
            ]
        }
    )
    target = {"RNA": original}
    handler(target)
    assert target["RNA"] is not original
    assert list(target["RNA"].obs_names) == ["c2", "c3"]


def test_call_with_empty_step_list_is_noop():
    adata = make_adata()
    out = AnnDataFilterHandler({"RNA": []})({"RNA": adata})
    assert out["RNA"] is adata
    assert adata.shape == (4, 4)


def test_call_ignores_steps_for_absent_modality():
    adata = make_adata()
    unused = MagicMock()
    handler = AnnDataFilterHandler({"RNA": [], "ATAC": [unused]})
    out = handler({"RNA": adata})
    assert set(out) == {"RNA"}
    unused.assert_not_called()


def test_call_propagates_step_errors():
    handler = AnnDataFilterHandler({"RNA": [FilterCells("x", min_counts=1000)]})
    with pytest.raises(RuntimeError):
        handler({"RNA": make_adata()})


@pytest.mark.xfail(
    strict=True,
    reason="BUG: modality present in target but without configured steps "
    "crashes with TypeError (iterating None from steps.get)",
)
def test_call_modality_without_steps_is_passed_through():
    adata = make_adata()
    handler = AnnDataFilterHandler({"RNA": []})
    out = handler({"RNA": adata, "ADT": make_adata(modality="ADT")})
    assert out["ADT"].shape == (4, 4)
