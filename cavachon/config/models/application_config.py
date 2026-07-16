import os
from collections import OrderedDict
from itertools import chain
from typing import Any, Dict, List

import yaml
from pydantic import model_validator

from cavachon.config.models.analysis_config import AnalysisConfig
from cavachon.config.models.base import BaseConfigModel
from cavachon.config.models.component_config import ComponentConfig
from cavachon.config.models.dataset_config import DatasetConfig
from cavachon.config.models.filter_config import FilterConfig
from cavachon.config.models.io_config import IOConfig
from cavachon.config.models.modality_config import ModalityConfig
from cavachon.config.models.model_config import ModelConfig
from cavachon.config.models.sample_config import SampleConfig
from cavachon.config.models.training_config import TrainingConfig
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

        # --- io ---
        io_data = yaml_raw.get(Constants.CONFIG_FIELD_IO, {})

        # --- modality: list → dict keyed by sanitized name ---
        modality_raw = yaml_raw.get(Constants.CONFIG_FIELD_MODALITY, [])
        modality_dict: Dict[str, Any] = {}
        for m in modality_raw:
            m_copy = dict(m)
            mname = GeneralUtils.tensorflow_compatible_str(m_copy.get("name", ""))
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
                self.modality[mname].samples.append(sname)

        # 5. Cross-validate model components ↔ modalities, inject dist
        component_by_name: Dict[str, ComponentConfig] = {}
        for comp in self.model.components:
            for mname in comp.modality_names:
                if mname not in self.modality:
                    raise KeyError(
                        f"'{mname}' is not in the config of modality."
                    )
                modality_dist = self.modality[mname].dist
                comp.distribution_names[mname] = modality_dist
            component_by_name[comp.name] = comp

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

        # 7. Validate analysis entries
        # 7a. clustering, differential_analysis, conditional_attribution_scores
        for entry in chain(
            self.analysis.clustering,
            self.analysis.differential_analysis,
            self.analysis.conditional_attribution_scores,
        ):
            modality = entry.modality
            component_name = entry.component
            found_component = False
            found_modality = False
            for comp_cfg in self.components:
                if component_name == comp_cfg.name:
                    found_component = True
                    if modality in comp_cfg.modality_names:
                        found_modality = True
                        break
                if found_modality and found_component:
                    break
            if not found_modality:
                raise KeyError(
                    f"'{modality}' is not in the config of component "
                    f"'{component_name}'."
                )
            if not found_component:
                raise KeyError(
                    f"'{component_name}' is not in the config of components."
                )

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

        # 7d. z_hat clustering gate
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

        return self
