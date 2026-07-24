import warnings
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict
from pydantic.functional_validators import AfterValidator

from cavachon.utils.general_utils import GeneralUtils


def _tf_compatible_str(v: str) -> str:
    return GeneralUtils.tensorflow_compatible_str(v)


TensorflowCompatibleStr = Annotated[str, AfterValidator(_tf_compatible_str)]


class BaseConfigModel(BaseModel):
    """Base config model with unknown-field RuntimeWarning.

    Uses ``extra='allow'`` and emits a ``RuntimeWarning`` in
    ``model_post_init`` for any field not declared on the model,
    replicating the old ``ConfigMapping`` behaviour.

    """

    model_config = ConfigDict(extra="allow")

    def model_post_init(self, __context: Any) -> None:
        extras = self.__pydantic_extra__ or {}
        for key in extras:
            warnings.warn(
                f"Unexpected field {key} in {self.__class__.__name__}. "
                f"Please check if there is any unintentional typo. "
                f"Some fields might be set to default unexpectedly.",
                RuntimeWarning,
            )
