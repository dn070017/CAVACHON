from dataclasses import dataclass

from sklearn.preprocessing import LabelEncoder


@dataclass
class BatchEffectProcessConfig:
    colname: str
    categorical: bool
    encoder: LabelEncoder | None
    n_vars: int

    def __eq__(this: object, other: object) -> bool:
        if not isinstance(this, BatchEffectProcessConfig) or not isinstance(
            other, BatchEffectProcessConfig
        ):
            raise NotImplementedError(
                "BatchEffectProcessConfig objects can only compare with BatchEffectProcessConfig"
            )

        if this.colname != other.colname:
            return False
        if this.categorical != other.categorical:
            return False
        if this.n_vars != other.n_vars:
            return False
        if this.encoder is None and other.encoder is None:
            return True
        if this.encoder is not None and other.encoder is None:
            return False
        if this.encoder is None and other.encoder is not None:
            return False
        if isinstance(this.encoder, LabelEncoder) and isinstance(
            other.encoder, LabelEncoder
        ):
            if any(this.encoder.classes_ != other.encoder.classes_):
                return False
        else:
            return False

        return True
