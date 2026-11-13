import pytest
from pydantic import ValidationError

from cavachon.config import ComponentConfig


def test_name_required():
    with pytest.raises(ValidationError):
        ComponentConfig()


def test_defaults():
    c = ComponentConfig(name="c")
    assert c.conditioned_on_z == [] and c.conditioned_on_z_hat == []
    assert c.modality_names == []
    assert c.n_latent_dims == 5
    assert c.n_latent_priors == 2 * 5 + 1
    assert c.n_encoder_layers == 3
    assert c.reparameterize_z_hat is True
    # global-overridable fields stay None until ApplicationConfig fills them
    for f in (
        "n_parent_annealing_epochs",
        "n_kl_annealing_epochs",
        "enable_kmeans_init",
        "learn_z_hat_priors",
        "kl_annealing_ratio",
        "n_z_hat_annealing_epochs",
        "n_z_hat_training_epochs",
        "z_hat_kl_weight",
        "max_regular_training_epochs",
    ):
        assert getattr(c, f) is None, f


def test_mutable_defaults_not_shared():
    a, b = ComponentConfig(name="a"), ComponentConfig(name="b")
    a.distribution_names["x"] = "d"
    a.conditioned_on_z.append("p")
    a.save_x["x"] = False
    assert b.distribution_names == {}
    assert b.conditioned_on_z == []
    assert b.save_x == {}


def test_flatten_modalities_with_defaults():
    c = ComponentConfig(name="c", modalities=[{"name": "RNA"}])
    assert c.modality_names == ["RNA"]
    assert c.distribution_names == {"RNA": ""}
    assert c.n_decoder_layers == {"RNA": 3}
    assert c.save_x == {"RNA": True}
    assert c.save_z == {"RNA": True}
    assert "modalities" not in (c.model_extra or {})


def test_flatten_modalities_with_overrides_and_sanitizing():
    c = ComponentConfig(
        name="my comp",
        conditioned_on_z=["a b"],
        conditioned_on_z_hat=["c:d"],
        modalities=[
            {"name": "RNA seq", "n_decoder_layers": 1, "save_x": False},
            {"name": "ATAC", "save_z": False},
        ],
    )
    assert c.name == "my_comp"
    assert c.conditioned_on_z == ["a_b"]
    assert c.conditioned_on_z_hat == ["c_d"]
    assert c.modality_names == ["RNA_seq", "ATAC"]
    assert c.n_decoder_layers == {"RNA_seq": 1, "ATAC": 3}
    assert c.save_x == {"RNA_seq": False, "ATAC": True}
    assert c.save_z == {"RNA_seq": True, "ATAC": False}


def test_flatten_sanitizes_modality_names_via_general_utils(general_utils):
    general_utils.tensorflow_compatible_str.side_effect = lambda s: f"tf_{s}"
    c = ComponentConfig(name="c", modalities=[{"name": "RNA"}, {}])
    general_utils.tensorflow_compatible_str.assert_any_call("RNA")
    general_utils.tensorflow_compatible_str.assert_any_call("")  # missing name
    assert c.modality_names == ["tf_RNA", "tf_"]
    assert set(c.save_x) == {"tf_RNA", "tf_"}
    assert c.name == "tf_c"


def test_n_latent_priors_explicit_positive_kept():
    assert ComponentConfig(name="c", n_latent_priors=7).n_latent_priors == 7


def test_n_latent_priors_zero_means_automatic():
    c = ComponentConfig(name="c", n_latent_dims=4, n_latent_priors=0)
    assert c.n_latent_priors == 9


def test_negative_n_latent_priors_rejected():
    # regression: negative values were silently replaced by 2 * dims + 1
    with pytest.raises(ValidationError):
        ComponentConfig(name="c", n_latent_dims=4, n_latent_priors=-3)


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_latent_dims", 0),
        ("n_latent_dims", -1),
        ("n_encoder_layers", 0),
        ("n_kl_annealing_epochs", -1),
        ("n_parent_annealing_epochs", -1),
        ("n_z_hat_annealing_epochs", -1),
        ("n_z_hat_training_epochs", -1),
        ("max_regular_training_epochs", -1),
        ("z_hat_kl_weight", -0.5),
    ],
)
def test_negative_or_zero_dims_rejected(field, value):
    # regression: negative dims, layers and epochs were accepted
    with pytest.raises(ValidationError):
        ComponentConfig(name="c", **{field: value})


def test_decoder_layers_must_be_positive():
    with pytest.raises(ValidationError):
        ComponentConfig.model_validate(
            {"name": "c", "modalities": [{"name": "RNA", "n_decoder_layers": 0}]}
        )


@pytest.mark.parametrize("ratio", [(-0.1, 0.6, 0.5), (0.5, 0.2, 0.2), (0.5, 0.5, 0.5)])
def test_invalid_kl_annealing_ratio_rejected(ratio):
    # regression: negative ratios (or ratios not summing to 1) were accepted
    with pytest.raises(ValidationError):
        ComponentConfig(name="c", kl_annealing_ratio=ratio)


def test_kl_annealing_ratio_none_and_valid_accepted():
    assert ComponentConfig(name="c").kl_annealing_ratio is None
    assert ComponentConfig(
        name="c", kl_annealing_ratio=(1.0, 0.0, 0.0)
    ).kl_annealing_ratio == (1.0, 0.0, 0.0)


def test_kl_annealing_ratio_parsed_as_tuple():
    c = ComponentConfig(name="c", kl_annealing_ratio=[0.2, 0.3, 0.5])
    assert c.kl_annealing_ratio == (0.2, 0.3, 0.5)
    with pytest.raises(ValidationError):
        ComponentConfig(name="c", kl_annealing_ratio=[0.5, 0.5])


# regression: the before-validator no longer pops ``modalities`` from the
# caller's dict, so validating the same dict twice gives the same result.
def test_model_validate_does_not_mutate_input_dict():
    data = {"name": "c", "modalities": [{"name": "RNA"}]}
    first = ComponentConfig.model_validate(data)
    assert data == {"name": "c", "modalities": [{"name": "RNA"}]}
    second = ComponentConfig.model_validate(data)
    assert first == second
    assert second.modality_names == ["RNA"]
