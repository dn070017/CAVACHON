"""Unit tests for cavachon.modules.base.hierarchical_encoder.

Mocked: ProgressiveScaler (FakeProgressiveScaler, multiplies by 0.5) and
MultivariateNormalDiagParameterizerLayer (FakeMVNLayer). The r_network
Dense layer and the Sequential b_network are defined inside the module and
stay real.
"""

import numpy as np
import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.modules.base.hierarchical_encoder import HierarchicalEncoder
from test.modules.base.conftest import FakeMVNLayer, FakeProgressiveScaler

MODULE = "cavachon.modules.base.hierarchical_encoder"

Z = Constants.MODEL_OUTPUTS_Z
ZC = Constants.MODULE_INPUTS_CONDITIONED_Z
ZHC = Constants.MODULE_INPUTS_CONDITIONED_Z_HAT


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(f"{MODULE}.ProgressiveScaler", FakeProgressiveScaler)
    monkeypatch.setattr(
        f"{MODULE}.MultivariateNormalDiagParameterizerLayer", FakeMVNLayer
    )


def _set_r_identity(enc, n):
    dense = enc.r_network.layers[0]
    dense.kernel.assign(np.eye(n, dtype=np.float32))
    if dense.bias is not None:
        dense.bias.assign(np.zeros(n, dtype=np.float32))


Z_VAL = tf.constant([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
ZC_VAL = tf.constant([[10.0, 11.0], [12.0, 13.0]])
ZHC_VAL = tf.constant([[20.0, 21.0, 22.0, 23.0], [24.0, 25.0, 26.0, 27.0]])


class TestConstruction:
    def test_sub_layers(self, patched):
        enc = HierarchicalEncoder(
            n_latent_dims=3, progressive_iterations=123, name="he"
        )
        assert enc.name == "he"
        assert FakeProgressiveScaler.instances[0].total_iterations_arg == 123
        assert enc.progressive_scaler is FakeProgressiveScaler.instances[0]
        assert enc.z_hat_parameterizer is FakeMVNLayer.instances[0]
        assert FakeMVNLayer.instances[0].init_args == dict(
            event_dims=3, name="he_b_parameterizer"
        )
        assert enc.r_network.name == Constants.MODULE_R_NETWORK
        assert enc.b_network.name == Constants.MODULE_B_NETWORK
        assert enc.b_network.layers == [enc.z_hat_parameterizer]
        dense = enc.r_network.layers[0]
        assert (
            dense.units == 3 and dense.use_bias is False and dense.name == "he_r_dense"
        )

    def test_use_bias(self, patched):
        enc = HierarchicalEncoder(n_latent_dims=3, use_bias=True)
        assert enc.r_network.layers[0].use_bias is True

    def test_flags_stored(self, patched):
        enc = HierarchicalEncoder(
            is_conditioned_on_z=True, is_conditioned_on_z_hat=False
        )
        assert enc.is_conditioned_on_z is True and enc.is_conditioned_on_z_hat is False

    def test_unknown_kwarg_is_rejected(self, patched):
        # regression: fixed bug, see test_report/UNIT_TEST_REPORT.md
        # Keras 3 rejects unrecognized keyword arguments with ValueError
        with pytest.raises((TypeError, ValueError)):
            HierarchicalEncoder(n_latent_dims=3, progressive_step=10)

    def test_trainable_kwarg_forwarded(self, patched):
        # regression: fixed bug, see test_report/UNIT_TEST_REPORT.md
        enc = HierarchicalEncoder(n_latent_dims=3, trainable=False)
        assert enc.trainable is False


class TestCall:
    def test_unconditioned(self, patched):
        enc = HierarchicalEncoder(n_latent_dims=3, name="he")
        enc({Z: Z_VAL})
        _set_r_identity(enc, 3)
        # extra conditioned keys must be ignored when flags are False
        out = enc({Z: Z_VAL, ZC: ZC_VAL, ZHC: ZHC_VAL})
        mvn = FakeMVNLayer.instances[0]
        np.testing.assert_allclose(mvn.calls[-1]["inputs"].numpy(), 0.5 * Z_VAL.numpy())
        scaler = FakeProgressiveScaler.instances[0]
        np.testing.assert_allclose(scaler.calls[-1]["inputs"].numpy(), Z_VAL.numpy())
        assert tuple(out.shape) == (2, 6)
        np.testing.assert_allclose(
            out.numpy()[:, :3],
            np.repeat(0.5 * Z_VAL.numpy().sum(-1, keepdims=True), 3, -1),
            atol=1e-6,
        )
        np.testing.assert_allclose(out.numpy()[:, 3:], np.log(2.0), atol=1e-6)

    def test_conditioned_on_z(self, patched):
        enc = HierarchicalEncoder(n_latent_dims=3, is_conditioned_on_z=True)
        enc({Z: Z_VAL, ZC: ZC_VAL})
        _set_r_identity(enc, 3)
        enc({Z: Z_VAL, ZC: ZC_VAL, ZHC: ZHC_VAL})
        expected = np.concatenate([ZC_VAL.numpy(), 0.5 * Z_VAL.numpy()], axis=-1)
        np.testing.assert_allclose(
            FakeMVNLayer.instances[0].calls[-1]["inputs"].numpy(), expected
        )

    def test_conditioned_on_z_hat(self, patched):
        enc = HierarchicalEncoder(n_latent_dims=3, is_conditioned_on_z_hat=True)
        enc({Z: Z_VAL, ZHC: ZHC_VAL})
        _set_r_identity(enc, 3)
        enc({Z: Z_VAL, ZC: ZC_VAL, ZHC: ZHC_VAL})
        expected = np.concatenate([ZHC_VAL.numpy(), 0.5 * Z_VAL.numpy()], axis=-1)
        np.testing.assert_allclose(
            FakeMVNLayer.instances[0].calls[-1]["inputs"].numpy(), expected
        )

    def test_conditioned_on_both_order(self, patched):
        enc = HierarchicalEncoder(
            n_latent_dims=3, is_conditioned_on_z=True, is_conditioned_on_z_hat=True
        )
        enc({Z: Z_VAL, ZC: ZC_VAL, ZHC: ZHC_VAL})
        _set_r_identity(enc, 3)
        out = enc({Z: Z_VAL, ZC: ZC_VAL, ZHC: ZHC_VAL})
        expected = np.concatenate(
            [ZC_VAL.numpy(), ZHC_VAL.numpy(), 0.5 * Z_VAL.numpy()], axis=-1
        )
        mvn = FakeMVNLayer.instances[0]
        np.testing.assert_allclose(mvn.calls[-1]["inputs"].numpy(), expected)
        assert mvn.loc_weight.shape == (2 + 4 + 3, 3)
        assert tuple(out.shape) == (2, 6)

    def test_missing_conditioned_input_raises(self, patched):
        enc = HierarchicalEncoder(n_latent_dims=3, is_conditioned_on_z=True)
        # the None placeholder for the missing key reaches tf.concat
        with pytest.raises(ValueError, match="None"):
            enc({Z: Z_VAL})

    def test_r_network_linear_in_z(self, patched):
        """r_network is a bias-free linear map: r(2z) == 2 r(z)."""
        enc = HierarchicalEncoder(n_latent_dims=3)
        enc({Z: Z_VAL})
        enc({Z: 2.0 * Z_VAL})
        mvn_in = FakeMVNLayer.instances[0].calls
        np.testing.assert_allclose(
            mvn_in[-1]["inputs"].numpy(),
            2.0 * mvn_in[-2]["inputs"].numpy(),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_train_step_not_implemented(self, patched):
        with pytest.raises(NotImplementedError):
            HierarchicalEncoder().train_step(None)


class TestInitializeScaleDiag:
    def test_sets_weight_zero_and_bias_inverse_softplus(self, patched):
        enc = HierarchicalEncoder(n_latent_dims=3)
        enc({Z: Z_VAL})
        target = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        enc.initialize_scale_diag(target)
        p = FakeMVNLayer.instances[0]
        np.testing.assert_array_equal(p.scale_diag_weight.numpy(), 0.0)
        assert p.scale_diag_bias.shape == (1, 3)
        np.testing.assert_allclose(
            tf.math.softplus(p.scale_diag_bias).numpy(),
            target.reshape(1, -1),
            rtol=1e-5,
        )
        out = enc({Z: Z_VAL})
        np.testing.assert_allclose(
            out.numpy()[:, 3:], np.tile(target, (2, 1)), rtol=1e-5
        )

    def test_clamps_non_positive(self, patched):
        enc = HierarchicalEncoder(n_latent_dims=2)
        enc({Z: Z_VAL})
        enc.initialize_scale_diag([0.0, -3.0])
        bias = FakeMVNLayer.instances[0].scale_diag_bias.numpy()
        assert np.all(np.isfinite(bias))
        np.testing.assert_allclose(tf.math.softplus(bias).numpy(), 1e-6, rtol=1e-3)

    def test_does_not_touch_loc(self, patched):
        enc = HierarchicalEncoder(n_latent_dims=3)
        enc({Z: Z_VAL})
        p = FakeMVNLayer.instances[0]
        before_w, before_b = p.loc_weight.numpy().copy(), p.loc_bias.numpy().copy()
        enc.initialize_scale_diag(np.ones(3))
        np.testing.assert_array_equal(p.loc_weight.numpy(), before_w)
        np.testing.assert_array_equal(p.loc_bias.numpy(), before_b)
        # loc_weight is the first weight of b_network (relied upon by callers)
        np.testing.assert_array_equal(enc.b_network.get_weights()[0], before_w)


class TestConfig:
    def test_get_config_round_trip(self, patched):
        enc = HierarchicalEncoder(
            n_latent_dims=3,
            is_conditioned_on_z=True,
            progressive_iterations=7,
            name="he",
        )
        config = enc.get_config()
        assert config["n_latent_dims"] == 3
        assert config["is_conditioned_on_z"] is True
        assert config["progressive_iterations"] == 7
