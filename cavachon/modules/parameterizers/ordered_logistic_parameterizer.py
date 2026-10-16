from cavachon.modules.parameterizers.parameterizer import Parameterizer


class OrderedLogisticParameterizer(Parameterizer):
    """Module for ordered logistic parameterization."""

    default_libsize_scaling = False

    @classmethod
    def make(
        cls,
        input_dims: int,
        event_dims: int,
        name: str = "ordered_logistic",
        **kwargs,
    ):
        return super().make(
            input_dims=input_dims,
            event_dims=event_dims,
            name=name,
            libsize_scaling=False,
            exp_transform=False,
            **kwargs,
        )
