from typing import Any, Dict, List, Tuple

from pydantic import model_validator

from cavachon.config.models.base import BaseConfigModel, TensorflowCompatibleStr
from cavachon.utils.general_utils import GeneralUtils


class ComponentConfig(BaseConfigModel):
    """Component configuration model.

    The YAML ``modalities`` list is flattened into ``modality_names``,
    ``distribution_names``, ``n_decoder_layers``, ``save_x``, and
    ``save_z`` during validation.

    """

    name: TensorflowCompatibleStr
    conditioned_on_z: List[TensorflowCompatibleStr] = []
    conditioned_on_z_hat: List[TensorflowCompatibleStr] = []
    modality_names: List[str] = []
    distribution_names: Dict[str, str] = {}
    save_x: Dict[str, bool] = {}
    save_z: Dict[str, bool] = {}
    n_vars: Dict[str, int] = {}
    n_vars_batch_effect: Dict[str, int] = {}
    n_latent_dims: int = 5
    n_latent_priors: int = 0
    n_encoder_layers: int = 3
    n_decoder_layers: Dict[str, int] = {}
    n_parent_annealing_epochs: int = 1
    n_kl_annealing_epochs: int = 25
    enable_kmeans_init: bool = True
    kl_annealing_ratio: Tuple[float, float, float] = (0.5, 0.2, 0.3)
    reparameterize_z_hat: bool = True

    @model_validator(mode="before")
    @classmethod
    def _flatten_modalities(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "modalities" not in data:
            return data
        modalities = data.pop("modalities")

        modality_names: List[str] = []
        distribution_names: Dict[str, str] = {}
        n_decoder_layers: Dict[str, int] = {}
        save_x: Dict[str, bool] = {}
        save_z: Dict[str, bool] = {}

        for m in modalities:
            mname = GeneralUtils.tensorflow_compatible_str(m.get("name", ""))
            modality_names.append(mname)
            distribution_names[mname] = m.get("distribution_names", "")
            n_decoder_layers[mname] = m.get("n_decoder_layers", 3)
            save_x[mname] = m.get("save_x", True)
            save_z[mname] = m.get("save_z", True)

        data["modality_names"] = modality_names
        data["distribution_names"] = distribution_names
        data["n_decoder_layers"] = n_decoder_layers
        data["save_x"] = save_x
        data["save_z"] = save_z
        return data

    @model_validator(mode="after")
    def _default_n_latent_priors(self) -> "ComponentConfig":
        if self.n_latent_priors <= 0:
            self.n_latent_priors = 2 * self.n_latent_dims + 1
        return self
