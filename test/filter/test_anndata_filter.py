from cavachon.filter.anndata_filter import AnnDataFilter


def test_init_stores_name_and_forces_inplace():
    f = AnnDataFilter("my_step", min_counts=3, inplace=False)
    assert f.name == "my_step"
    assert f.kwargs == {"min_counts": 3, "inplace": True}


def test_init_pops_step_key():
    f = AnnDataFilter("FilterCells", step="FilterCells", min_genes=1)
    assert "step" not in f.kwargs
    assert f.kwargs["min_genes"] == 1


def test_init_does_not_mutate_caller_dict():
    params = {"step": "FilterCells", "min_genes": 1}
    AnnDataFilter("x", **params)
    assert params == {"step": "FilterCells", "min_genes": 1}


def test_instances_do_not_share_kwargs():
    a = AnnDataFilter("a", min_counts=1)
    b = AnnDataFilter("b")
    a.kwargs["extra"] = 1
    assert "extra" not in b.kwargs
    assert b.kwargs == {"inplace": True}


def test_base_call_is_identity(dense_adata):
    f = AnnDataFilter("noop")
    out = f(dense_adata)
    assert out is dense_adata
    assert out.shape == (4, 4)
