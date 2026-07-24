from cavachon.config.base import BaseConfigModel


class DatasetConfig(BaseConfigModel):
    """Dataset configuration model."""

    batch_size: int = 128
    shuffle: bool = False
