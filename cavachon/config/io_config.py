import os

from pydantic import Field, field_validator

from cavachon.config.base import BaseConfigModel


class IOConfig(BaseConfigModel):
    """IO path configuration model."""

    # validate_default so the default "./" is resolved like explicit paths.
    checkpointdir: str = Field("./", validate_default=True)
    datadir: str = Field("./", validate_default=True)
    outdir: str = Field("./", validate_default=True)

    @field_validator("checkpointdir", "datadir", "outdir", mode="after")
    @classmethod
    def _resolve_paths(cls, v: str) -> str:
        return os.path.realpath(os.path.dirname(f"{v}/"))
