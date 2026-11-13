import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from cavachon.io import FileReader

# 2 cells x 3 features
X = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])


# ---------------------------------------------------------------- read_mtx
def test_read_mtx_sparse(fake_fs):
    fake_fs.matrices["/d/m.mtx"] = sp.coo_matrix(X)
    m = FileReader.read_mtx("/d/m.mtx")
    fake_fs.realpath.assert_called_once_with("/d/m.mtx")
    fake_fs.mmread.assert_called_once_with("/d/m.mtx")
    assert sp.isspmatrix_csr(m)
    np.testing.assert_array_equal(m.toarray(), X)


def test_read_mtx_uses_realpath_result(fake_fs):
    fake_fs.realpath.side_effect = lambda p: "/resolved/m.mtx"
    fake_fs.matrices["/resolved/m.mtx"] = sp.coo_matrix(X)
    FileReader.read_mtx("link.mtx")
    fake_fs.mmread.assert_called_once_with("/resolved/m.mtx")


def test_read_mtx_transpose(fake_fs):
    fake_fs.matrices["/d/m.mtx"] = sp.coo_matrix(X.T)  # features x cells
    m = FileReader.read_mtx("/d/m.mtx", transpose=True)
    assert sp.isspmatrix_csr(m)
    assert m.shape == (2, 3)
    np.testing.assert_array_equal(m.toarray(), X)


def test_read_mtx_missing_file_propagates(fake_fs):
    with pytest.raises(FileNotFoundError):
        FileReader.read_mtx("/d/missing.mtx")


@pytest.mark.xfail(
    strict=True,
    reason="BUG: dense (array-format) MatrixMarket returns ndarray; "
    ".tocsr() raises AttributeError",
)
def test_read_mtx_dense_array_format(fake_fs):
    fake_fs.matrices["/d/dense.mtx"] = X.copy()  # mmread returns ndarray
    m = FileReader.read_mtx("/d/dense.mtx")
    np.testing.assert_array_equal(m.toarray(), X)


# ---------------------------------------------------------------- read_table
def test_read_table_call_arguments_default(fake_fs):
    fake_fs.tables["/d/f.tsv"] = "g1\tG1\ng2\tG2\n"
    FileReader.read_table("/d/f.tsv", name="RNA")
    fake_fs.realpath.assert_called_once_with("/d/f.tsv")
    fake_fs.read_csv.assert_called_once_with("/d/f.tsv", sep="\t", header=None)


def test_read_table_call_arguments_headers_and_kwargs(fake_fs):
    fake_fs.tables["/d/h.csv"] = "a,b\n1,2\n"
    FileReader.read_table("/d/h.csv", name="X", has_headers=True, delimiter=",", nrows=1)
    fake_fs.read_csv.assert_called_once_with("/d/h.csv", sep=",", header=0, nrows=1)


def test_read_table_without_headers_or_colnames(fake_fs):
    fake_fs.tables["/d/f.tsv"] = "g1\tG1\ng2\tG2\n"
    df = FileReader.read_table("/d/f.tsv", name="RNA")
    assert df.shape == (2, 2)
    assert df.index.name == "RNA"
    assert list(df.columns) == [0, 1]


def test_read_table_with_colnames_sets_index(fake_fs):
    fake_fs.tables["/d/f.tsv"] = "g1\tG1\ng2\tG2\n"
    df = FileReader.read_table("/d/f.tsv", name="RNA", colnames=["id", "sym"])
    assert list(df.columns) == ["id", "sym"]
    assert df.index.name == "RNA:id"
    assert list(df.index) == ["g1", "g2"]


@pytest.mark.parametrize("index_col", ["sym", 1])
def test_read_table_index_col_by_name_or_position(fake_fs, index_col):
    fake_fs.tables["/d/f.tsv"] = "g1\tG1\ng2\tG2\n"
    df = FileReader.read_table(
        "/d/f.tsv", name="RNA", colnames=["id", "sym"], index_col=index_col
    )
    assert df.index.name == "RNA:sym"
    assert list(df.index) == ["G1", "G2"]


def test_read_table_custom_delimiter(fake_fs):
    fake_fs.tables["/d/f.csv"] = "a,1\nb,2\n"
    df = FileReader.read_table("/d/f.csv", name="X", colnames=["k", "v"], delimiter=",")
    assert list(df.index) == ["a", "b"]
    assert list(df["v"]) == [1, 2]


def test_read_table_colnames_length_mismatch(fake_fs):
    fake_fs.tables["/d/f.tsv"] = "g1\tG1\n"
    with pytest.raises(ValueError):
        FileReader.read_table("/d/f.tsv", name="RNA", colnames=["only"])


def test_read_table_missing_file_propagates(fake_fs):
    with pytest.raises(FileNotFoundError):
        FileReader.read_table("/d/nope.tsv", name="RNA")


def test_read_table_with_headers_keeps_columns(fake_fs):
    fake_fs.tables["/d/h.tsv"] = "barcode\tct\nc1\tT\nc2\tB\n"
    df = FileReader.read_table("/d/h.tsv", name="RNA", has_headers=True)
    assert list(df.columns) == ["barcode", "ct"]
    assert len(df) == 2


@pytest.mark.xfail(
    strict=True,
    reason="BUG: with has_headers=True (no colnames) index_col is ignored "
    "(never passed to read_csv nor applied); index becomes 0..n-1",
)
def test_read_table_with_headers_uses_index_col(fake_fs):
    fake_fs.tables["/d/h.tsv"] = "barcode\tct\nc1\tT\nc2\tB\n"
    df = FileReader.read_table("/d/h.tsv", name="RNA", has_headers=True)
    assert list(df.index) == ["c1", "c2"]


# ---------------------------------------------------------------- read_multiomics_data
def test_read_multiomics_single_sample(fake_fs, helpers):
    fake_fs.add_sample("/data", "s1", X, ["c1", "c2"], ["g1", "g2", "g3"])
    cfg = helpers.make_config(
        {"s1": ("first", [helpers.make_sample_modality("s1")])}, {"RNA": ["s1"]}
    )
    adata = FileReader.read_multiomics_data(cfg, "RNA")
    assert adata.shape == (2, 3)
    assert sp.issparse(adata.X)
    assert adata.X.dtype == np.float32
    np.testing.assert_array_equal(adata.X.toarray(), X)
    assert list(adata.obs_names) == ["c1", "c2"]
    assert list(adata.var_names) == ["g1", "g2", "g3"]
    assert list(adata.obs["Sample"]) == ["s1", "s1"]
    assert list(adata.obs["Description"]) == ["first", "first"]
    assert adata.obs.index.name == "RNA:barcode"
    assert list(adata.var["symbol"]) == ["G1", "G2", "G3"]


def test_read_multiomics_paths_joined_with_datadir(fake_fs, helpers):
    fake_fs.add_sample("/data", "s1", X, ["c1", "c2"], ["g1", "g2", "g3"])
    cfg = helpers.make_config(
        {"s1": ("", [helpers.make_sample_modality("s1")])}, {"RNA": ["s1"]}
    )
    FileReader.read_multiomics_data(cfg, "RNA")
    fake_fs.mmread.assert_called_once_with("/data/s1.mtx")
    read_paths = [c.args[0] for c in fake_fs.read_csv.call_args_list]
    assert read_paths == ["/data/s1_b.tsv", "/data/s1_f.tsv"]
    for call in fake_fs.read_csv.call_args_list:
        assert call.kwargs == {"sep": "\t", "header": None}


def test_read_multiomics_has_headers_drops_colnames(fake_fs, helpers):
    fake_fs.add_sample("/data", "s1", X, ["c1", "c2"], ["g1", "g2", "g3"])
    fake_fs.tables["/data/s1_b.tsv"] = "barcode\nc1\nc2\n"
    mod = helpers.make_sample_modality("s1")
    mod.barcodes = helpers.make_feature("s1_b.tsv", ["ignored"], has_headers=True)
    cfg = helpers.make_config({"s1": ("", [mod])}, {"RNA": ["s1"]})
    adata = FileReader.read_multiomics_data(cfg, "RNA")
    assert fake_fs.read_csv.call_args_list[0].kwargs["header"] == 0
    assert list(adata.obs.columns[:1]) == ["barcode"]


def test_read_multiomics_transposed_matrix(fake_fs, helpers):
    fake_fs.add_sample("/data", "s1", X, ["c1", "c2"], ["g1", "g2", "g3"], transpose=True)
    cfg = helpers.make_config(
        {"s1": ("", [helpers.make_sample_modality("s1", transpose=True)])}, {"RNA": ["s1"]}
    )
    adata = FileReader.read_multiomics_data(cfg, "RNA")
    assert adata.shape == (2, 3)
    np.testing.assert_array_equal(adata.X.toarray(), X)


def test_read_multiomics_picks_requested_modality(fake_fs, helpers):
    fake_fs.add_sample("/data", "r1", X, ["c1", "c2"], ["g1", "g2", "g3"])
    fake_fs.add_sample("/data", "a1", X[:, :2], ["c1", "c2"], ["p1", "p2"])
    mods = [
        helpers.make_sample_modality("r1", modality="RNA"),
        helpers.make_sample_modality("a1", modality="ATAC"),
    ]
    cfg = helpers.make_config({"s": ("", mods)}, {"RNA": ["s"], "ATAC": ["s"]})
    assert list(FileReader.read_multiomics_data(cfg, "RNA").var_names) == ["g1", "g2", "g3"]
    assert list(FileReader.read_multiomics_data(cfg, "ATAC").var_names) == ["p1", "p2"]


def test_read_multiomics_unknown_modality(fake_fs, helpers):
    cfg = helpers.make_config({}, {"RNA": []})
    with pytest.raises(KeyError, match="Protein"):
        FileReader.read_multiomics_data(cfg, "Protein")
    fake_fs.read_csv.assert_not_called()


def test_read_multiomics_missing_matrix_file(fake_fs, helpers):
    fake_fs.add_sample("/data", "s1", X, ["c1", "c2"], ["g1", "g2", "g3"])
    del fake_fs.matrices["/data/s1.mtx"]
    cfg = helpers.make_config(
        {"s1": ("", [helpers.make_sample_modality("s1")])}, {"RNA": ["s1"]}
    )
    with pytest.raises(FileNotFoundError):
        FileReader.read_multiomics_data(cfg, "RNA")


def test_read_multiomics_absolute_filename_overrides_datadir(fake_fs, helpers):
    fake_fs.add_sample("/other", "s1", X, ["c1", "c2"], ["g1", "g2", "g3"])
    cfg = helpers.make_config(
        {"s1": ("", [helpers.make_sample_modality("s1", prefix="/other/")])},
        {"RNA": ["s1"]},
    )
    assert FileReader.read_multiomics_data(cfg, "RNA").shape == (2, 3)
    fake_fs.mmread.assert_called_once_with("/other/s1.mtx")


def test_read_multiomics_barcode_count_mismatch(fake_fs, helpers):
    fake_fs.add_sample("/data", "s1", X, ["c1", "c2", "c3"], ["g1", "g2", "g3"])
    cfg = helpers.make_config(
        {"s1": ("", [helpers.make_sample_modality("s1")])}, {"RNA": ["s1"]}
    )
    with pytest.raises(ValueError):
        FileReader.read_multiomics_data(cfg, "RNA")


def test_read_multiomics_mismatched_features_raises(fake_fs, helpers):
    fake_fs.add_sample("/data", "s1", X, ["a1", "a2"], ["g1", "g2", "g3"])
    fake_fs.add_sample("/data", "s2", X, ["b1", "b2"], ["g1", "g2", "gX"])
    cfg = helpers.make_config(
        {
            "s1": ("", [helpers.make_sample_modality("s1")]),
            "s2": ("", [helpers.make_sample_modality("s2")]),
        },
        {"RNA": ["s1", "s2"]},
    )
    with pytest.raises(ValueError, match="Features of sample 's2'"):
        FileReader.read_multiomics_data(cfg, "RNA")


def test_read_multiomics_two_samples(fake_fs, helpers):
    fake_fs.add_sample("/data", "s1", X, ["a1", "a2"], ["g1", "g2", "g3"])
    fake_fs.add_sample("/data", "s2", X * 2, ["b1", "b2"], ["g1", "g2", "g3"])
    cfg = helpers.make_config(
        {
            "s1": ("", [helpers.make_sample_modality("s1")]),
            "s2": ("", [helpers.make_sample_modality("s2")]),
        },
        {"RNA": ["s1", "s2"]},
    )
    adata = FileReader.read_multiomics_data(cfg, "RNA")
    assert adata.shape == (4, 3)
    assert list(adata.obs["Sample"]) == ["s1", "s1", "s2", "s2"]
    np.testing.assert_array_equal(adata.X.toarray(), np.vstack([X, X * 2]))


def test_read_multiomics_without_colnames(fake_fs, helpers):
    fake_fs.add_sample("/data", "s1", X, ["c1", "c2"], ["g1", "g2", "g3"])
    cfg = helpers.make_config(
        {"s1": ("", [helpers.make_sample_modality("s1", colnames=False)])},
        {"RNA": ["s1"]},
    )
    adata = FileReader.read_multiomics_data(cfg, "RNA")
    assert adata.shape == (2, 3)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: AnnData(dtype=...) is deprecated (FutureWarning) and "
    "scheduled for removal",
)
def test_read_multiomics_no_future_warning(fake_fs, helpers):
    fake_fs.add_sample("/data", "s1", X, ["c1", "c2"], ["g1", "g2", "g3"])
    cfg = helpers.make_config(
        {"s1": ("", [helpers.make_sample_modality("s1")])}, {"RNA": ["s1"]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        FileReader.read_multiomics_data(cfg, "RNA")
