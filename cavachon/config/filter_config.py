from pydantic import ConfigDict

from cavachon.config.base import BaseConfigModel


class FilterConfig(BaseConfigModel):
    """Filter step configuration with arbitrary extra fields."""

    model_config = ConfigDict(extra="allow")

    step: str
