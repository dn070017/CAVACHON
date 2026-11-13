import anndata
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

# 4 cells x 4 genes.
#   cell totals : c0=4, c1=0, c2=6, c3=10
#   cell n_genes: c0=2, c1=0, c2=3, c3=4
#   gene totals : MT-CO1=4, MTOR=4, GAPDH=5, ACTB=7
#   gene n_cells: MT-CO1=3, MTOR=2, GAPDH=2, ACTB=2
COUNTS = np.array(
    [
        [1, 0, 3, 0],
        [0, 0, 0, 0],
        [2, 2, 0, 2],
        [1, 2, 2, 5],
    ],
    dtype=np.float32,
)
GENES = ["MT-CO1", "MTOR", "GAPDH", "ACTB"]
CELLS = ["c0", "c1", "c2", "c3"]


def make_adata(is_sparse: bool = False, modality: str = "RNA") -> anndata.AnnData:
    """Mimic cavachon.io.FileReader: var index named '<modality>:gene'
    and a 'gene' column carrying the gene names."""
    X = sparse.csr_matrix(COUNTS) if is_sparse else COUNTS.copy()
    obs = pd.DataFrame(index=pd.Index(CELLS, name=f"{modality}:barcode"))
    var = pd.DataFrame({"gene": GENES}, index=pd.Index(GENES, name=f"{modality}:gene"))
    return anndata.AnnData(X=X, obs=obs, var=var)


@pytest.fixture(params=[False, True], ids=["dense", "sparse"])
def adata(request):
    return make_adata(is_sparse=request.param)


@pytest.fixture
def dense_adata():
    return make_adata(is_sparse=False)


# ---------------------------------------------------------------- shared mocks
# Boundary rule: anything imported from another cavachon package is mocked
# where it is looked up by the filter package.


@pytest.fixture(autouse=True)
def mock_general_utils(monkeypatch):
    """Mock cavachon.utils.GeneralUtils as seen by filter_qc.

    duplicate_obj_to_list(obj, n) is stubbed with the contract the filter
    relies on: a list of n copies of obj."""
    from unittest.mock import MagicMock

    fake = MagicMock(name="GeneralUtils")
    fake.duplicate_obj_to_list.side_effect = lambda obj, n: [obj] * n
    monkeypatch.setattr("cavachon.filter.filter_qc.GeneralUtils", fake)
    return fake


@pytest.fixture
def mock_get_class_by_name(monkeypatch):
    """Mock ReflectionHandler.get_class_by_name as seen by the handler.
    Resolves to the real filter step classes (same package)."""
    from unittest.mock import MagicMock

    from cavachon.filter.filter_cells import FilterCells
    from cavachon.filter.filter_genes import FilterGenes
    from cavachon.filter.filter_qc import FilterQC

    registry = {
        "FilterCells": FilterCells,
        "FilterGenes": FilterGenes,
        "FilterQC": FilterQC,
    }

    def lookup(class_name, subdirectory="", postfix=""):
        if class_name not in registry:
            raise AssertionError(f"could not find {class_name}")
        return registry[class_name]

    mock = MagicMock(side_effect=lookup)
    monkeypatch.setattr(
        "cavachon.filter.anndata_filter_handler.ReflectionHandler.get_class_by_name",
        mock,
    )
    mock.registry = registry
    return mock


def filter_step_config(step, **params):
    """Stand-in for cavachon.config.FilterConfig: exposes .step and
    .model_dump() (which, like pydantic, includes 'step')."""
    from unittest.mock import MagicMock

    cfg = MagicMock(name=f"FilterConfig[{step}]")
    cfg.step = step
    cfg.model_dump.return_value = {"step": step, **params}
    return cfg
