from cavachon.modules.parameterizers.parameterizer import Parameterizer


class StudenttParameterizer(Parameterizer):
    """Module for Student-T parameterization."""

    default_libsize_scaling = False

    @classmethod
    def make(cls, input_dims: int, event_dims: int, name: str = "student_t", **kwargs):
        # We call the parent make, which handles the Keras functional setup
        return super().make(
            input_dims=input_dims,
            event_dims=event_dims,
            name=name,
            libsize_scaling=False,
            exp_transform=False,
        )
