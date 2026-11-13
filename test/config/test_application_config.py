import copy
import warnings

import pytest
from pydantic import ValidationError

from cavachon.config import (
    ApplicationConfig,
    ComponentConfig,
    DatasetConfig,
    TrainingConfig,
)


def _sample(name, modality="RNA", description=None):
    s = {
        "name": name,
        "modalities": [
            {
                "name": modality,
                "matrix": {"filename": f"{name}.mtx"},
                "barcodes": {"filename": f"{name}_b.tsv"},
                "features": {"filename": f"{name}_f.tsv"},
            }
        ],
    }
    if description is not None:
        s["description"] = description
    return s


# ---------------------------------------------------------------- loading
def test_minimal_config_loads(base_config, load_config):
    cfg = load_config(base_config)
    assert cfg.filename == "/cfg/config.yaml"
    assert cfg.modality_names == ["RNA", "ATAC"]
    assert set(cfg.modality) == {"RNA", "ATAC"}
    assert cfg.io.datadir == "/data"
    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.dataset, DatasetConfig)
    assert cfg.analysis.clustering == []
    assert cfg.filter == {"RNA": [], "ATAC": []}
    assert len(cfg.sample) == 0


def test_from_yaml_reads_through_open_and_yaml_load(base_config, load_config, yaml_source):
    cfg = load_config(base_config)
    yaml_source.open.assert_called_once_with("/cfg/config.yaml", "r")
    yaml_source.load.assert_called_once()
    _, kwargs = yaml_source.load.call_args
    assert kwargs["Loader"] is yaml_source.full_loader
    assert set(cfg.yaml_raw) == set(base_config)


def test_relative_filename_is_made_absolute(base_config, load_config, yaml_source, fake_os):
    cfg = load_config(base_config, filename="config.yaml")
    fake_os.realpath.assert_any_call("config.yaml")
    yaml_source.open.assert_called_once_with("/cwd/config.yaml", "r")
    assert cfg.filename == "/cwd/config.yaml"


def test_missing_file_raises(base_config, load_config, yaml_source):
    yaml_source.open.side_effect = FileNotFoundError("nope")
    with pytest.raises(FileNotFoundError):
        load_config(base_config)
    yaml_source.load.assert_not_called()


def test_no_modalities_raises(base_config, load_config):
    base_config.pop("modalities")
    with pytest.raises(KeyError, match="No modality found"):
        load_config(base_config)


def test_modality_names_sanitized_and_keyed(base_config, load_config, general_utils):
    base_config["modalities"] = [{"name": "my rna", "type": "rna"}]
    base_config["model"]["components"] = [
        {"name": "c", "modalities": [{"name": "my rna"}]}
    ]
    cfg = load_config(base_config)
    # the dict key is sanitized by from_yaml itself via GeneralUtils
    general_utils.tensorflow_compatible_str.assert_any_call("my rna")
    assert cfg.modality_names == ["my_rna"]
    assert cfg.modality["my_rna"].name == "my_rna"
    assert cfg.components[0].modality_names == ["my_rna"]


def test_learning_rate_yaml_string_coerced(base_config, load_config):
    base_config["model"]["training"] = {"optimizer": {"learning_rate": "5e-4"}}
    cfg = load_config(base_config)
    assert cfg.training.optimizer.learning_rate == pytest.approx(5e-4)
    assert cfg.model.training.optimizer.learning_rate == pytest.approx(5e-4)


def test_training_and_dataset_extracted_from_model(base_config, load_config):
    base_config["model"]["training"] = {"max_regular_training_epochs": 7, "train": False}
    base_config["model"]["dataset"] = {"batch_size": 16, "shuffle": True}
    cfg = load_config(base_config)
    assert cfg.training.max_regular_training_epochs == 7
    assert cfg.training.train is False
    assert cfg.dataset.batch_size == 16 and cfg.dataset.shuffle is True
    assert cfg.model.dataset.batch_size == 16


def test_filters_collected_per_modality(base_config, load_config):
    base_config["modalities"][0]["filters"] = [{"step": "FilterCells", "min_genes": 1}]
    cfg = load_config(base_config)
    assert [f.step for f in cfg.filter["RNA"]] == ["FilterCells"]
    assert cfg.filter["ATAC"] == []


# ---------------------------------------------------------------- samples
def test_samples_linked_to_modalities(base_config, load_config):
    base_config["samples"] = [_sample("s1", description="first"), _sample("s2")]
    cfg = load_config(base_config)
    assert list(cfg.sample) == ["s1", "s2"]
    assert cfg.sample["s1"].description == "first"
    assert cfg.modality["RNA"].samples == ["s1", "s2"]
    assert cfg.modality["ATAC"].samples == []


def test_sample_with_unknown_modality_raises(base_config, load_config):
    base_config["samples"] = [_sample("s1", modality="Protein")]
    with pytest.raises(KeyError, match="No configuration for modality Protein"):
        load_config(base_config)


# regression: samples already listed under a modality are not appended twice
def test_modality_samples_not_duplicated(base_config, load_config):
    base_config["modalities"][0]["samples"] = ["s1"]
    base_config["samples"] = [_sample("s1")]
    cfg = load_config(base_config)
    assert cfg.modality["RNA"].samples == ["s1"]


# regression: re-validating a dumped config does not duplicate samples
def test_revalidation_round_trip_idempotent(base_config, load_config):
    base_config["samples"] = [_sample("s1")]
    cfg = load_config(base_config)
    again = ApplicationConfig.model_validate(cfg.model_dump())
    assert again.modality["RNA"].samples == ["s1"]


# ---------------------------------------------------------------- components
def test_components_ordered_by_order_components(base_config, load_config, general_utils):
    base_config["model"]["components"].reverse()  # RNA (child) first
    general_utils.order_components.side_effect = lambda d: [d["ATAC"], d["RNA"]]
    cfg = load_config(base_config)
    general_utils.order_components.assert_called_once_with(
        {
            "RNA": {"name": "RNA", "conditioned_on_z": [], "conditioned_on_z_hat": ["ATAC"]},
            "ATAC": {"name": "ATAC", "conditioned_on_z": [], "conditioned_on_z_hat": []},
        }
    )
    assert [c.name for c in cfg.components] == ["ATAC", "RNA"]
    # model.components keeps YAML order
    assert [c.name for c in cfg.model.components] == ["RNA", "ATAC"]
    assert all(isinstance(c, ComponentConfig) for c in cfg.components)


def test_component_dist_injected_from_modality(base_config, load_config):
    base_config["modalities"][0]["dist"] = "IndependentBernoulli"
    cfg = load_config(base_config)
    by_name = {c.name: c for c in cfg.components}
    assert by_name["RNA"].distribution_names == {"RNA": "IndependentBernoulli"}
    assert by_name["ATAC"].distribution_names == {"ATAC": "IndependentBernoulli"}


def test_component_level_distribution_names_overridden(base_config, load_config):
    # Documents that a per-component distribution is parsed but ignored.
    base_config["model"]["components"][1]["modalities"][0]["distribution_names"] = "Foo"
    cfg = load_config(base_config)
    rna = next(c for c in cfg.components if c.name == "RNA")
    assert rna.distribution_names["RNA"] == "IndependentZeroInflatedNegativeBinomial"


def test_component_unknown_modality_raises(base_config, load_config):
    base_config["model"]["components"][0]["modalities"] = [{"name": "Protein"}]
    with pytest.raises(KeyError, match="'Protein' is not in the config of modality"):
        load_config(base_config)


def test_order_components_error_propagates(base_config, load_config, general_utils):
    # Cycle detection lives in cavachon.utils; config must not swallow it.
    general_utils.order_components.side_effect = AttributeError("cyclic graph")
    with pytest.raises(AttributeError, match="cyclic"):
        load_config(base_config)


def test_dependency_lists_passed_to_order_components(base_config, load_config, general_utils):
    base_config["model"]["components"][0]["conditioned_on_z"] = ["RNA"]
    load_config(base_config)
    (passed,), _ = general_utils.order_components.call_args
    assert passed["ATAC"]["conditioned_on_z"] == ["RNA"]
    assert passed["RNA"]["conditioned_on_z_hat"] == ["ATAC"]


# regression: unknown parents are rejected before order_components
@pytest.mark.parametrize("field", ["conditioned_on_z", "conditioned_on_z_hat"])
def test_unknown_parent_component_rejected_by_config(
    base_config, load_config, general_utils, field
):
    base_config["model"]["components"][1][field] = ["Ghost"]
    with pytest.raises(ValueError, match=r"'Ghost' \(conditioned on by component 'RNA'\)"):
        load_config(base_config)
    general_utils.order_components.assert_not_called()


# regression: duplicate component names raise
def test_duplicate_component_names_rejected(base_config, load_config):
    base_config["model"]["components"] = [
        {"name": "C", "modalities": [{"name": "RNA"}]},
        {"name": "C", "modalities": [{"name": "ATAC"}]},
    ]
    with pytest.raises(ValueError, match="Duplicate component name 'C'"):
        load_config(base_config)


# regression: duplicate modality names raise
def test_duplicate_modality_names_rejected(base_config, load_config):
    base_config["modalities"].append({"name": "RNA", "type": "atac"})
    with pytest.raises(ValueError, match="Duplicate modality name RNA"):
        load_config(base_config)


# regression: missing model.components raises
@pytest.mark.parametrize("empty", [None, []])
def test_missing_components_rejected(base_config, load_config, empty):
    if empty is None:
        base_config["model"].pop("components")
    else:
        base_config["model"]["components"] = empty
    with pytest.raises(ValidationError, match="No component found"):
        load_config(base_config)


# regression: component without modalities raises
def test_component_without_modalities_rejected(base_config, load_config):
    base_config["model"]["components"].append({"name": "Empty"})
    with pytest.raises(ValidationError, match="Component 'Empty' has no modalities"):
        load_config(base_config)


def test_component_with_empty_modalities_list_rejected(base_config, load_config):
    base_config["model"]["components"].append({"name": "Empty", "modalities": []})
    with pytest.raises(ValidationError, match="Component 'Empty' has no modalities"):
        load_config(base_config)


# ---------------------------------------------------------------- defaults propagation
def test_training_defaults_propagated(base_config, load_config):
    base_config["model"]["training"] = {
        "n_kl_annealing_epochs": 3,
        "learn_z_hat_priors": True,
        "kl_annealing_ratio": [1, 0, 0],
        "max_regular_training_epochs": 9,
    }
    base_config["model"]["components"][1]["n_kl_annealing_epochs"] = 11
    base_config["model"]["components"][1]["learn_z_hat_priors"] = False
    cfg = load_config(base_config)
    atac, rna = cfg.components
    assert atac.n_kl_annealing_epochs == 3
    assert rna.n_kl_annealing_epochs == 11
    assert atac.learn_z_hat_priors is True
    assert rna.learn_z_hat_priors is False  # explicit False is not overridden
    assert atac.kl_annealing_ratio == (1.0, 0.0, 0.0)
    assert atac.max_regular_training_epochs == 9
    assert atac.n_parent_annealing_epochs == 1
    assert atac.enable_kmeans_init is True
    assert atac.n_z_hat_annealing_epochs == 0
    assert atac.n_z_hat_training_epochs == 0
    assert atac.z_hat_kl_weight == 1.0


def test_components_are_same_objects_as_model_components(base_config, load_config):
    cfg = load_config(base_config)
    ids = {id(c) for c in cfg.model.components}
    assert all(id(c) in ids for c in cfg.components)


# ---------------------------------------------------------------- yaml_raw / typos
# regression: validators no longer mutate yaml_raw
def test_yaml_raw_is_unmodified(base_config, load_config):
    base_config["model"]["training"] = {"n_kl_annealing_epochs": 3}  # no optimizer
    original = copy.deepcopy(base_config)
    cfg = load_config(base_config)
    assert cfg.yaml_raw == original


# regression: unknown top-level keys warn
def test_unknown_top_level_key_warns(base_config, load_config):
    base_config["sample"] = [_sample("s1")]  # typo for 'samples'
    with pytest.warns(RuntimeWarning, match="Unexpected field sample in ApplicationConfig"):
        cfg = load_config(base_config)
    assert len(cfg.sample) == 0


def test_unknown_nested_key_warns(base_config, load_config):
    base_config["model"]["training"] = {"n_kl_anealing_epochs": 3}
    with pytest.warns(RuntimeWarning, match="n_kl_anealing_epochs"):
        load_config(base_config)


def test_valid_config_emits_no_runtime_warning(base_config, load_config):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        load_config(base_config)


# ---------------------------------------------------------------- io paths
def test_relative_io_paths_resolve_against_cwd_not_config_dir(
    base_config, load_config, fake_os
):
    # Documents current (suspicious) behaviour: relative io paths are
    # handed to realpath unchanged (=> process cwd), never joined with
    # the config file's directory.
    base_config["io"] = {"datadir": "data"}
    cfg = load_config(base_config, filename="/configs/config.yaml")
    fake_os.realpath.assert_any_call("data")
    assert all(c.args[0] != "/configs/data" for c in fake_os.realpath.call_args_list)
    assert cfg.io.datadir == "/cwd/data"


# ---------------------------------------------------------------- analysis cross-validation
def test_analysis_valid_entries_pass(base_config, load_config):
    base_config["analysis"] = {
        "clustering": [
            {"modality": "RNA", "component": "RNA", "use_rep": "z_hat"},
            {"modality": "ATAC", "component": "ATAC"},
        ],
        "differential_analysis": [{"modality": "RNA", "component": "RNA"}],
        "visualize_embedding": [{"modality": "RNA", "embedding_method": "pca"}],
        "conditional_attribution_scores": [
            {"modality": "RNA", "component": "RNA", "with_respect_to": ["ATAC"]}
        ],
        "differential_attribution_analysis": [
            {"modality": "RNA", "component": "RNA", "with_respect_to": ["ATAC"]}
        ],
        "enrichment_analysis": [{"analysis_type": "deg"}],
    }
    cfg = load_config(base_config)
    assert cfg.analysis.differential_analysis[0].use_cluster == "cluster_RNA"


@pytest.mark.parametrize(
    "section", ["clustering", "differential_analysis", "conditional_attribution_scores"]
)
def test_analysis_modality_not_in_component(base_config, load_config, section):
    base_config["analysis"] = {section: [{"modality": "ATAC", "component": "RNA"}]}
    with pytest.raises(KeyError, match="'ATAC' is not in the config of component 'RNA'"):
        load_config(base_config)


@pytest.mark.parametrize(
    "section",
    [
        "clustering",
        "differential_analysis",
        "hierarchical_differential_analysis",
        "conditional_attribution_scores",
        "differential_attribution_analysis",
    ],
)
def test_analysis_unknown_component_raises(base_config, load_config, section):
    base_config["analysis"] = {section: [{"modality": "RNA", "component": "Nope"}]}
    with pytest.raises(KeyError, match="'Nope' is not in the config of components"):
        load_config(base_config)


@pytest.mark.parametrize("section", ["differential_attribution_analysis"])
def test_analysis_modality_not_in_component_differential_attribution(
    base_config, load_config, section
):
    base_config["analysis"] = {section: [{"modality": "ATAC", "component": "RNA"}]}
    with pytest.raises(KeyError, match="'ATAC' is not in the config of component 'RNA'"):
        load_config(base_config)


# regression: unknown analysis component is reported as such
def test_analysis_unknown_component_error_message(base_config, load_config):
    base_config["analysis"] = {"clustering": [{"modality": "RNA", "component": "Nope"}]}
    with pytest.raises(KeyError, match="'Nope' is not in the config of components"):
        load_config(base_config)


def test_visualize_unknown_modality_raises(base_config, load_config):
    base_config["analysis"] = {
        "visualize_embedding": [{"modality": "Protein", "embedding_method": "umap"}]
    }
    with pytest.raises(KeyError, match="'Protein' is not in the config of any component"):
        load_config(base_config)


def test_attribution_unknown_with_respect_to(base_config, load_config):
    base_config["analysis"] = {
        "conditional_attribution_scores": [
            {"modality": "RNA", "component": "RNA", "with_respect_to": ["Ghost"]}
        ]
    }
    with pytest.raises(KeyError, match="'Ghost' \\(with_respect_to\\)"):
        load_config(base_config)


def test_z_hat_clustering_requires_parent(base_config, load_config):
    base_config["analysis"] = {
        "clustering": [{"modality": "ATAC", "component": "ATAC", "use_rep": "z_hat"}]
    }
    with pytest.raises(KeyError, match="no parents in conditioned_on_z_hat"):
        load_config(base_config)


def test_z_hat_clustering_requires_reparameterize(base_config, load_config):
    base_config["model"]["components"][1]["reparameterize_z_hat"] = False
    base_config["analysis"] = {
        "clustering": [{"modality": "RNA", "component": "RNA", "use_rep": "z_hat"}]
    }
    with pytest.raises(KeyError, match="reparameterize_z_hat"):
        load_config(base_config)


def test_differential_attribution_unknown_with_respect_to(base_config, load_config):
    base_config["analysis"] = {
        "differential_analysis": [{"modality": "RNA", "component": "RNA"}],
        "differential_attribution_analysis": [
            {"modality": "RNA", "component": "RNA", "with_respect_to": ["Ghost"]}
        ],
    }
    with pytest.raises(KeyError, match="Ghost"):
        load_config(base_config)


@pytest.mark.parametrize(
    "section, analysis_type, source",
    [
        ("differential_attribution_analysis", "deg", "differential_analysis"),
        ("differential_attribution_analysis", "hdeg", "hierarchical_differential_analysis"),
        ("enrichment_analysis", "deg", "differential_analysis"),
        ("enrichment_analysis", "hdeg", "hierarchical_differential_analysis"),
    ],
)
def test_source_gate(base_config, load_config, fake_os, section, analysis_type, source):
    entry = {"analysis_type": analysis_type}
    if section == "differential_attribution_analysis":
        entry.update(modality="RNA", component="RNA")
    base_config["analysis"] = {section: [entry]}
    # neither config nor directory -> error
    with pytest.raises(ValueError, match=f"requires either a {source}"):
        load_config(base_config)
    fake_os.isdir.assert_called_with(f"/out/{source}")
    # existing output directory satisfies the gate
    fake_os.isdir.return_value = True
    load_config(base_config)


def test_source_gate_satisfied_by_config(base_config, load_config, fake_os):
    base_config["analysis"] = {
        "hierarchical_differential_analysis": [{"modality": "RNA", "component": "RNA"}],
        "enrichment_analysis": [{"analysis_type": "hdeg"}],
        "differential_attribution_analysis": [
            {"modality": "RNA", "component": "RNA", "analysis_type": "hdeg"}
        ],
    }
    load_config(base_config)
    fake_os.isdir.assert_not_called()


# regression: hierarchical DA entries are cross-validated
@pytest.mark.parametrize(
    "entry, message",
    [
        ({"modality": "ATAC", "component": "RNA"}, "'ATAC' is not in the config of component 'RNA'"),
        ({"modality": "RNA", "component": "YY"}, "'YY' is not in the config of components"),
        (
            {"modality": "RNA", "component": "RNA", "donor_components": ["QQ"]},
            "'QQ' \\(donor_components\\)",
        ),
    ],
)
def test_hierarchical_entries_cross_validated(base_config, load_config, entry, message):
    base_config["analysis"] = {"hierarchical_differential_analysis": [entry]}
    with pytest.raises(KeyError, match=message):
        load_config(base_config)


def test_hierarchical_valid_entry_passes(base_config, load_config):
    base_config["analysis"] = {
        "hierarchical_differential_analysis": [
            {"modality": "RNA", "component": "RNA", "donor_components": ["ATAC", "RNA"]}
        ]
    }
    cfg = load_config(base_config)
    assert cfg.analysis.hierarchical_differential_analysis[0].use_cluster == "cluster_RNA"


def test_direct_construction_requires_metadata():
    with pytest.raises(ValidationError):
        ApplicationConfig(
            io={}, modality={"RNA": {"name": "RNA", "type": "rna"}}, sample={},
            model={}, training={}, dataset={}, analysis={},
        )


@pytest.mark.parametrize("content", [None, [], "just a string"])
def test_empty_or_non_mapping_yaml_rejected(load_config, content):
    # regression: an empty YAML file raised AttributeError ('NoneType' has no 'get')
    with pytest.raises(ValueError, match="empty or not a mapping"):
        load_config(content)
