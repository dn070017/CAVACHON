import os
import warnings
from collections import OrderedDict
from itertools import chain
from typing import Any, Dict, List

import yaml
from pydantic import model_validator

from cavachon.config.analysis_config import AnalysisConfig
from cavachon.config.base import BaseConfigModel
from cavachon.config.component_config import ComponentConfig
from cavachon.config.dataset_config import DatasetConfig
from cavachon.config.filter_config import FilterConfig
from cavachon.config.io_config import IOConfig
from cavachon.config.modality_config import ModalityConfig
from cavachon.config.model_config import ModelConfig
from cavachon.config.sample_config import SampleConfig
from cavachon.config.training_config import TrainingConfig
from cavachon.environment.constants import Constants
from cavachon.utils.general_utils import GeneralUtils


class ApplicationConfig(BaseConfigModel):
    """Root CAVACHON application configuration.

    Loaded from a YAML file via ``from_yaml()``.  Performs cross-field
    validation equivalent to the old ``setup_*`` methods.

    """

    # -- direct YAML fields ------------------------------------------------
    io: IOConfig
    modality: Dict[str, ModalityConfig]
    sample: OrderedDict[str, SampleConfig]
    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig
    analysis: AnalysisConfig
    # -- computed -----------------------------------------------------------
    filter: Dict[str, List[FilterConfig]] = {}
    components: List[ComponentConfig] = []
    modality_names: List[str] = []
    # -- metadata -----------------------------------------------------------
    filename: str
    yaml_raw: Dict[str, Any]

    @classmethod
    def from_yaml(cls, filename: str) -> "ApplicationConfig":
        """Load configuration from a YAML file.

        Parameters
        ----------
        filename : str
            Absolute or relative path to the YAML config file.

        Returns
        -------
        ApplicationConfig

        """
        filename = os.path.realpath(filename)
        with open(filename, "r") as f:
            yaml_raw: Dict[str, Any] = yaml.load(f, Loader=yaml.FullLoader)

        # --- unknown top-level fields (likely typos) ---
        known_fields = (
            Constants.CONFIG_FIELD_IO,
            Constants.CONFIG_FIELD_MODALITY,
            Constants.CONFIG_FIELD_SAMPLE,
            Constants.CONFIG_FIELD_MODEL,
            Constants.CONFIG_FIELD_ANALYSIS,
        )
        for key in yaml_raw:
            if key not in known_fields:
                warnings.warn(
                    f"Unexpected field {key} in {cls.__name__}. "
                    f"Please check if there is any unintentional typo. "
                    f"Some fields might be set to default unexpectedly.",
                    RuntimeWarning,
                )

        # --- io ---
        io_data = yaml_raw.get(Constants.CONFIG_FIELD_IO, {})

        # --- modality: list → dict keyed by sanitized name ---
        modality_raw = yaml_raw.get(Constants.CONFIG_FIELD_MODALITY, [])
        modality_dict: Dict[str, Any] = {}
        for m in modality_raw:
            m_copy = dict(m)
            mname = GeneralUtils.tensorflow_compatible_str(m_copy.get("name", ""))
            if mname in modality_dict:
                raise ValueError(
                    f"Duplicate modality name {mname} in config file {filename}."
                )
            modality_dict[mname] = m_copy

        # --- sample: list → OrderedDict ---
        sample_raw = yaml_raw.get(Constants.CONFIG_FIELD_SAMPLE, [])
        sample_dict: OrderedDict[str, Any] = OrderedDict()
        for s in sample_raw:
            sample_dict[s.get("name", "")] = s

        # --- model (training / dataset extracted for top-level too) ---
        model_raw = yaml_raw.get(Constants.CONFIG_FIELD_MODEL, {})
        training_data = model_raw.get(Constants.CONFIG_FIELD_MODEL_TRAINING, {})
        dataset_data = model_raw.get(Constants.CONFIG_FIELD_MODEL_DATASET, {})

        # --- analysis ---
        analysis_data = yaml_raw.get(Constants.CONFIG_FIELD_ANALYSIS, {})

        return cls(
            filename=filename,
            yaml_raw=yaml_raw,
            io=io_data,
            modality=modality_dict,
            sample=sample_dict,
            model=model_raw,
            training=training_data,
            dataset=dataset_data,
            analysis=analysis_data,
        )

    # ------------------------------------------------------------------
    # Cross-field validation (replaces old ApplicationConfig.setup_*)
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _cross_validate(self) -> "ApplicationConfig":
        # 1. Populate modality_names
        self.modality_names = list(self.modality.keys())

        # 2. Modality list non-empty
        if not self.modality_names:
            raise KeyError(f"No modality found in config file {self.filename}.")

        # 3. Populate filter from modalities
        self.filter = {name: m_cfg.filters for name, m_cfg in self.modality.items()}

        # 4. Cross-validate samples ↔ modalities
        for sname, s_cfg in self.sample.items():
            for mod_file in s_cfg.modalities:
                mname = mod_file.name
                if mname not in self.modality:
                    raise KeyError(
                        f"No configuration for modality {mname} "
                        f"of sample {sname} in the config file {self.filename}."
                    )
                if sname not in self.modality[mname].samples:
                    self.modality[mname].samples.append(sname)

        # 5. Cross-validate model components ↔ modalities, inject dist
        if not self.model.components:
            raise ValueError(
                f"No component found in the config of model in config file "
                f"{self.filename}."
            )
        component_by_name: Dict[str, ComponentConfig] = {}
        for comp in self.model.components:
            if not comp.modality_names:
                raise ValueError(
                    f"Component '{comp.name}' has no modalities in config "
                    f"file {self.filename}."
                )
            for mname in comp.modality_names:
                if mname not in self.modality:
                    raise KeyError(
                        f"'{mname}' is not in the config of modality."
                    )
                modality_dist = self.modality[mname].dist
                comp.distribution_names[mname] = modality_dist
            if comp.name in component_by_name:
                raise ValueError(
                    f"Duplicate component name '{comp.name}' in the config "
                    f"of components."
                )
            component_by_name[comp.name] = comp

        # 5b. Parents in conditioned_on_z(_hat) must be known components
        for comp in component_by_name.values():
            for parent in chain(comp.conditioned_on_z, comp.conditioned_on_z_hat):
                if parent not in component_by_name:
                    raise ValueError(
                        f"'{parent}' (conditioned on by component "
                        f"'{comp.name}') is not in the config of components."
                    )

        # 6. Topological sort components
        comp_configs_as_dicts = {
            name: {
                "name": c.name,
                Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z: c.conditioned_on_z,
                Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT: c.conditioned_on_z_hat,
            }
            for name, c in component_by_name.items()
        }
        ordered = GeneralUtils.order_components(comp_configs_as_dicts)
        self.components = [component_by_name[c["name"]] for c in ordered]

        # Build a fast component lookup
        comp_map: Dict[str, ComponentConfig] = {
            c.name: c for c in self.components
        }

        # 7. Propagate training-level defaults to components when not set
        for comp in self.components:
            if comp.n_parent_annealing_epochs is None:
                comp.n_parent_annealing_epochs = (
                    self.training.n_parent_annealing_epochs
                )
            if comp.n_kl_annealing_epochs is None:
                comp.n_kl_annealing_epochs = self.training.n_kl_annealing_epochs
            if comp.enable_kmeans_init is None:
                comp.enable_kmeans_init = self.training.enable_kmeans_init
            if comp.learn_z_hat_priors is None:
                comp.learn_z_hat_priors = self.training.learn_z_hat_priors
            if comp.kl_annealing_ratio is None:
                comp.kl_annealing_ratio = self.training.kl_annealing_ratio
            if comp.n_z_hat_annealing_epochs is None:
                comp.n_z_hat_annealing_epochs = (
                    self.training.n_z_hat_annealing_epochs
                )
            if comp.n_z_hat_training_epochs is None:
                comp.n_z_hat_training_epochs = self.training.n_z_hat_training_epochs
            if comp.z_hat_kl_weight is None:
                comp.z_hat_kl_weight = self.training.z_hat_kl_weight
            if comp.max_regular_training_epochs is None:
                comp.max_regular_training_epochs = (
                    self.training.max_regular_training_epochs
                )

        # 8. Validate analysis entries
        # 7a. clustering, differential_analysis,
        # hierarchical_differential_analysis, conditional_attribution_scores
        for entry in chain(
            self.analysis.clustering,
            self.analysis.differential_analysis,
            self.analysis.hierarchical_differential_analysis,
            self.analysis.conditional_attribution_scores,
        ):
            self._check_component_modality(entry, comp_map)

        # 7b. visualize_embedding (modality only)
        for viz in self.analysis.visualize_embedding:
            found = any(
                viz.modality in cfg.modality_names for cfg in self.components
            )
            if not found:
                raise KeyError(
                    f"'{viz.modality}' is not in the config of any component."
                )

        # 7c. conditional_attribution_scores with_respect_to check
        for attr in self.analysis.conditional_attribution_scores:
            if attr.component not in comp_map:
                raise KeyError(
                    f"'{attr.component}' (modality) is not in the config "
                    f"of components."
                )
            for wrt in attr.with_respect_to:
                if wrt not in comp_map:
                    raise KeyError(
                        f"'{wrt}' (with_respect_to) is not in the config "
                        f"of components."
                    )

        # 7c'. hierarchical_differential_analysis donor_components check
        for hier in self.analysis.hierarchical_differential_analysis:
            for donor in hier.donor_components or []:
                if donor not in comp_map:
                    raise KeyError(
                        f"'{donor}' (donor_components) is not in the config "
                        f"of components."
                    )

        # 7d. differential_attribution_analysis gate
        for diff_attr in self.analysis.differential_attribution_analysis:
            self._check_component_modality(diff_attr, comp_map)
            for wrt in diff_attr.with_respect_to:
                if wrt not in comp_map:
                    raise KeyError(
                        f"'{wrt}' (with_respect_to) is not in the config "
                        f"of components."
                    )
            source_name = (
                "differential_analysis"
                if diff_attr.analysis_type == "deg"
                else "hierarchical_differential_analysis"
            )
            source_configs = (
                self.analysis.differential_analysis
                if diff_attr.analysis_type == "deg"
                else self.analysis.hierarchical_differential_analysis
            )
            if not source_configs:
                source_directory = os.path.join(self.io.outdir, source_name)
                if not os.path.isdir(source_directory):
                    raise ValueError(
                        f"differential_attribution_analysis with "
                        f"analysis_type '{diff_attr.analysis_type}' requires "
                        f"either a {source_name} configuration or an existing "
                        f"directory at {source_directory}."
                    )

        # 7e. z_hat clustering gate
        for cluster_cfg in self.analysis.clustering:
            if cluster_cfg.use_rep == "z_hat":
                comp_name = cluster_cfg.component
                if comp_name not in comp_map:
                    raise KeyError(
                        f"'{comp_name}' is not in the config of components."
                    )
                target = comp_map[comp_name]
                if not target.reparameterize_z_hat:
                    raise KeyError(
                        f"'{comp_name}' in the config file {self.filename} "
                        f"does not have reparameterize_z_hat set to True, "
                        f"so z_hat clustering is not allowed."
                    )
                if len(target.conditioned_on_z_hat) == 0:
                    raise KeyError(
                        f"'{comp_name}' in the config file {self.filename} "
                        f"has no parents in conditioned_on_z_hat, so z_hat "
                        f"clustering is not allowed."
                    )

        # 7e. Enrichment source analysis gate
        self._validate_enrichment_sources(self.analysis, self.io.outdir)

        return self

    @staticmethod
    def _check_component_modality(
        entry: Any, comp_map: Dict[str, ComponentConfig]
    ) -> None:
        """Check that the component of an analysis entry exists and
        contains the modality of the entry."""
        comp_cfg = comp_map.get(entry.component)
        if comp_cfg is None:
            raise KeyError(
                f"'{entry.component}' is not in the config of components."
            )
        if entry.modality not in comp_cfg.modality_names:
            raise KeyError(
                f"'{entry.modality}' is not in the config of component "
                f"'{entry.component}'."
            )

    @staticmethod
    def _validate_enrichment_sources(
        analysis: AnalysisConfig, outdir: str
    ) -> None:
        """Validate configured or previously generated enrichment inputs."""
        for enrichment in analysis.enrichment_analysis:
            source_configs = (
                analysis.differential_analysis
                if enrichment.analysis_type == "deg"
                else analysis.hierarchical_differential_analysis
            )
            if source_configs:
                continue

            source_name = (
                "differential_analysis"
                if enrichment.analysis_type == "deg"
                else "hierarchical_differential_analysis"
            )
            source_directory = os.path.join(outdir, source_name)
            if not os.path.isdir(source_directory):
                raise ValueError(
                    f"enrichment_analysis with analysis_type "
                    f"'{enrichment.analysis_type}' requires either a "
                    f"{source_name} configuration or an existing directory "
                    f"at {source_directory}."
                )
