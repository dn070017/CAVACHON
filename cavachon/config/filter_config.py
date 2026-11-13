from typing import Any

from pydantic import ConfigDict

from cavachon.config.base import BaseConfigModel


class FilterConfig(BaseConfigModel):
    """Filter step configuration with arbitrary extra fields."""

    model_config = ConfigDict(extra="allow")

    step: str

    def model_post_init(self, __context: Any) -> None:
        # Extra fields are the keyword arguments of the filter step, so
        # they are expected and should not trigger the typo warning.
        pass
