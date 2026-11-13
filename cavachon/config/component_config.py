from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field, field_validator, model_validator
from typing_extensions import Annotated

from cavachon.config.base import BaseConfigModel, TensorflowCompatibleStr
from cavachon.config.training_config import validate_kl_annealing_ratio
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
    # set by Model from the latent dimensions of the parent components
    z_conditional_dims: Optional[int] = None
    z_hat_conditional_dims: Optional[int] = None
    n_latent_dims: int = Field(5, ge=1)
    # 0 means 2 * n_latent_dims + 1 (see _default_n_latent_priors)
    n_latent_priors: int = Field(0, ge=0)
    n_encoder_layers: int = Field(3, ge=1)
    n_decoder_layers: Dict[str, Annotated[int, Field(ge=1)]] = {}
    n_parent_annealing_epochs: Optional[int] = Field(None, ge=0)
    n_kl_annealing_epochs: Optional[int] = Field(None, ge=0)
    enable_kmeans_init: Optional[bool] = None
    learn_z_hat_priors: Optional[bool] = None
    kl_annealing_ratio: Optional[Tuple[float, float, float]] = None
    n_z_hat_annealing_epochs: Optional[int] = Field(None, ge=0)
    n_z_hat_training_epochs: Optional[int] = Field(None, ge=0)
    z_hat_kl_weight: Optional[float] = Field(None, ge=0)
    reparameterize_z_hat: bool = True
    max_regular_training_epochs: Optional[int] = Field(None, ge=0)

    @field_validator("kl_annealing_ratio")
    @classmethod
    def _validate_kl_annealing_ratio(
        cls, v: Optional[Tuple[float, float, float]]
    ) -> Optional[Tuple[float, float, float]]:
        return validate_kl_annealing_ratio(v)

    @model_validator(mode="before")
    @classmethod
    def _flatten_modalities(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "modalities" not in data:
            return data
        # copy so the caller's dict (e.g. ApplicationConfig.yaml_raw) is
        # not mutated.
        data = dict(data)
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
        # n_latent_priors: 0 (the default) means automatic: one prior at
        # the origin plus two per latent dimension. Negative values are
        # rejected by the field constraint.
        if self.n_latent_priors == 0:
            self.n_latent_priors = 2 * self.n_latent_dims + 1
        return self
