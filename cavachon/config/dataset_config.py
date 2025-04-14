from typing import List

from pydantic import BaseModel, ConfigDict, Field

from cavachon.config.dataset_modality_config import DatasetModalityConfig


class DatasetConfig(BaseModel):
    """DatasetConfig

    Config for Dataset.

    Attributes
    ----------
    batch_size: int
        batch size for iterating dataset. Defaults to 128.

    shuffle: bool
        whether or not to shuffle the dataset during training.
        Defaults to False.

    modalities: List[DatasetModalityConfig]
        list of modality configurations.
    """

    batch_size: int = Field(
        default=128, description="batch size for iterating dataset."
    )
    shuffle: bool = Field(
        default=False,
        description="whether or not to shuffle the dataset during training.",
    )
    modalities: List[DatasetModalityConfig] = Field(
        description="list of modality configurations."
    )
    model_config = ConfigDict(
        extra="forbid", revalidate_instances="always", validate_assignment=True
    )
