from dataclasses import dataclass
from typing import Self

from sklearn.preprocessing import LabelEncoder


@dataclass
class BatchEffectProcessConfig:
    colname: str
    categorical: bool
    encoder: LabelEncoder | None
    n_vars: int

    def __eq__(self, other: Self) -> bool:
        if self.colname != other.colname:
            return False
        if self.categorical != other.categorical:
            return False
        if self.n_vars != other.n_vars:
            return False
        if self.encoder is None and other.encoder is None:
            return True
        if self.encoder is not None and other.encoder is None:
            return False
        if self.encoder is None and other.encoder is not None:
            return False
        if any(self.encoder.classes_ != other.encoder.classes_):
            return False

        return True
