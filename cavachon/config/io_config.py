import os

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IOConfig(BaseModel):
    """Config for inputs and outputs.

    Attributes
    ----------
    datadir: str
        Path to the data directory. Defaults to "./".

    outdir: str
        Path to the output directory. Defaults to "./".

    """

    datadir: str = Field(
        default="./", description="path to the data directory.", validate_default=True
    )
    outdir: str = Field(
        default="./", description="path to the output directory.", validate_default=True
    )
    model_config = ConfigDict(
        extra="forbid", revalidate_instances="always", validate_assignment=True
    )

    @field_validator("datadir", "outdir", mode="after")
    @classmethod
    def convert_to_path(cls, value: str) -> str:
        path = os.path.realpath(os.path.dirname(f"{value}/"))
        if not os.path.exists(f"{path}"):
            raise ValueError(f"Path {value} does not exist.")
        return path


# %%
