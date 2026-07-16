import os

from pydantic import field_validator

from cavachon.config.base import BaseConfigModel


class IOConfig(BaseConfigModel):
    """IO path configuration model."""

    checkpointdir: str = "./"
    datadir: str = "./"
    outdir: str = "./"

    @field_validator("checkpointdir", "datadir", "outdir", mode="after")
    @classmethod
    def _resolve_paths(cls, v: str) -> str:
        return os.path.realpath(os.path.dirname(f"{v}/"))
