"""Unit tests for the data side of Workflow: read_modalities,
setup_mdata, filter_adata_mapping, update_config_nvars, setup_dataloader
and update_nvars_batch_effect.

Package boundary: ``cavachon.workflow``. FileReader, Modality,
MultiModality and DataLoader are MagicMocks (conftest ``deps``);
anndata.read_h5ad is a recorder. update_config_nvars runs against a tiny
in-memory MuData.
"""

import inspect
from types import SimpleNamespace
from unittest import mock

from cavachon.workflow.workflow import Workflow
from test.workflow.conftest import make_config, make_mdata, make_workflow


def modality_cfg(type_, dist, h5ad="", batch=()):
    return SimpleNamespace(
        type=type_, dist=dist, h5ad=h5ad, batch_effect_colnames=list(batch)
    )


def two_modality_config():
    config = make_config()
    config.modality = {
        "RNA": modality_cfg("rna", "IndependentZeroInflatedNegativeBinomial", "rna.h5ad", ["batch"]),
        "ATAC": modality_cfg("atac", "IndependentBernoulli", "", ["sample", "lane"]),
    }
    config.modality_names = ["RNA", "ATAC"]
    return config


# --------------------------------------------------------------------------
# read_modalities
# --------------------------------------------------------------------------


def test_read_modalities_h5ad_vs_file_reader(deps, read_h5ad):
    config = two_modality_config()
    result = Workflow.read_modalities(config)

    read_h5ad.assert_called_once_with("/data/rna.h5ad")
    deps.FileReader.read_multiomics_data.assert_called_once_with(config, "ATAC")
    assert list(result) == ["RNA", "ATAC"]


def test_read_modalities_wraps_each_adata_in_modality(deps, read_h5ad):
    config = two_modality_config()
    modalities = [mock.sentinel.rna_modality, mock.sentinel.atac_modality]
    deps.Modality.side_effect = modalities

    result = Workflow.read_modalities(config)

    assert deps.Modality.call_args_list == [
        mock.call(
            read_h5ad.return_value,
            name="RNA",
            modality_type="rna",
            distribution_name="IndependentZeroInflatedNegativeBinomial",
            batch_effect_colnames=["batch"],
        ),
        mock.call(
            deps.FileReader.read_multiomics_data.return_value,
            name="ATAC",
            modality_type="atac",
            distribution_name="IndependentBernoulli",
            batch_effect_colnames=["sample", "lane"],
        ),
    ]
    assert result == {"RNA": modalities[0], "ATAC": modalities[1]}


def test_read_modalities_absolute_h5ad_path_is_kept(deps, read_h5ad):
    config = two_modality_config()
    config.modality["RNA"].h5ad = "/elsewhere/rna.h5ad"
    Workflow.read_modalities(config)
    read_h5ad.assert_called_once_with("/elsewhere/rna.h5ad")


def test_read_modalities_follows_modality_names(deps, read_h5ad):
    config = two_modality_config()
    config.modality_names = ["ATAC"]
    result = Workflow.read_modalities(config)

    read_h5ad.assert_not_called()
    assert list(result) == ["ATAC"]


def test_modality_keyword_names_match_signature():
    from cavachon.modality.modality import Modality

    params = inspect.signature(Modality.__init__).parameters
    for name in ("name", "modality_type", "distribution_name", "batch_effect_colnames"):
        assert name in params


# --------------------------------------------------------------------------
# setup_mdata / filter_adata_mapping
# --------------------------------------------------------------------------


def test_setup_mdata_reads_filters_and_builds_multimodality(deps, monkeypatch):
    config = two_modality_config()
    wf = make_workflow(config, mdata=None)
    manager = mock.Mock()
    manager.read_modalities.return_value = {"RNA": "raw"}
    manager.filters.return_value = {"RNA": "filtered"}
    monkeypatch.setattr(Workflow, "read_modalities", staticmethod(manager.read_modalities))
    wf.anndata_filters = manager.filters
    manager.attach_mock(deps.MultiModality, "MultiModality")
    wf.update_config_nvars = manager.update_config_nvars

    wf.setup_mdata()

    assert [c[0] for c in manager.mock_calls] == [
        "read_modalities",
        "filters",
        "MultiModality",
        "update_config_nvars",
    ]
    manager.read_modalities.assert_called_once_with(config)
    manager.filters.assert_called_once_with({"RNA": "raw"})
    deps.MultiModality.assert_called_once_with({"RNA": "filtered"})
    assert wf.mdata is deps.MultiModality.return_value


def test_filter_adata_mapping_delegates_to_filters():
    wf = make_workflow()
    mapping = {"RNA": "adata"}
    assert wf.filter_adata_mapping(mapping) is wf.anndata_filters.return_value
    wf.anndata_filters.assert_called_once_with(mapping)


# --------------------------------------------------------------------------
# update_config_nvars
# --------------------------------------------------------------------------


def components():
    return [
        SimpleNamespace(name="rna_c", modality_names=["RNA"]),
        SimpleNamespace(name="joint", modality_names=["RNA", "ATAC"]),
    ]


def test_update_config_nvars_per_component():
    config = make_config()
    config.components = components()
    wf = make_workflow(config, mdata=make_mdata(n_vars={"RNA": 3, "ATAC": 5}))

    wf.update_config_nvars()

    assert config.components[0].n_vars == {"RNA": 3}
    assert config.components[1].n_vars == {"RNA": 3, "ATAC": 5}


def test_update_config_nvars_syncs_model_components():
    config = make_config()
    comps = components()
    config.components = comps
    wf = make_workflow(config, mdata=make_mdata(n_vars={"RNA": 3, "ATAC": 5}))

    wf.update_config_nvars()

    assert config.components == comps
    assert config.model.components == comps
    assert config.model.components[1] is comps[1]


# --------------------------------------------------------------------------
# setup_dataloader / update_nvars_batch_effect
# --------------------------------------------------------------------------


def test_setup_dataloader_collects_per_modality_names(deps):
    config = two_modality_config()
    wf = make_workflow(config, dataloader=None)
    del wf.batch_effect_colnames, wf.distribution_names
    deps.DataLoader.return_value.n_vars_batch_effect = {}

    wf.setup_dataloader()

    assert wf.distribution_names == {
        "RNA": "IndependentZeroInflatedNegativeBinomial",
        "ATAC": "IndependentBernoulli",
    }
    assert wf.batch_effect_colnames == {"RNA": ["batch"], "ATAC": ["sample", "lane"]}


def test_setup_dataloader_passes_args_in_signature_order(deps):
    config = two_modality_config()
    config.dataset.batch_size = 17
    wf = make_workflow(config, dataloader=None)
    deps.DataLoader.return_value.n_vars_batch_effect = {}

    wf.setup_dataloader()

    deps.DataLoader.assert_called_once_with(
        wf.mdata, 17, wf.batch_effect_colnames, wf.distribution_names
    )
    assert wf.dataloader is deps.DataLoader.return_value

    # the positional order above must match the real signature
    from cavachon.dataloader.dataloader import DataLoader

    names = list(inspect.signature(DataLoader.__init__).parameters)[1:5]
    assert names == ["mdata", "batch_size", "batch_effect_colnames", "distribution_names"]


def test_setup_dataloader_updates_batch_effect_nvars(deps):
    config = two_modality_config()
    config.components = components()
    wf = make_workflow(config, dataloader=None)
    deps.DataLoader.return_value.n_vars_batch_effect = {"RNA": 2, "ATAC": 4}

    wf.setup_dataloader()

    assert config.components[0].n_vars_batch_effect == {"RNA": 2}
    assert config.components[1].n_vars_batch_effect == {"RNA": 2, "ATAC": 4}


def test_update_nvars_batch_effect_missing_modality_is_none():
    config = make_config()
    comps = components()
    config.components = comps
    wf = make_workflow(config)
    wf.dataloader.n_vars_batch_effect = {"RNA": 2}

    wf.update_nvars_batch_effect()

    assert comps[1].n_vars_batch_effect == {"RNA": 2, "ATAC": None}
    assert config.model.components == comps
