# `cavachon/distributions`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| ✅ Clean | [`test/distributions`](../../test/distributions) | 61 | 0 | 99% | Standalone: depends on 0 packages, used by 3 |

- **Depends on:** nothing
- **Used by:** callbacks, losses, tools
- **Mocked in tests:** nothing; tfp and scipy serve as numeric references.

## Fixed

- ✅ **ZINB mixture weights (under-rated at first; a real training problem).** `IndependentZeroInflatedNegativeBinomialDistribution` built its weights as `probs = sigmoid(logits)` and `1 - probs`. In float32, `sigmoid(l)` rounds to 1.0 once the dropout logit reaches about 17, so the NB component got weight 0. Measured with x=5, mean 3, dispersion 2:

  | logit | old `log_prob` | fixed / exact |
  |---|---|---|
  | 15 | −18.172 | −18.146 |
  | 17 | **−78.527** | −20.146 |
  | 20 | **−78.527** | −23.146 |
  | 30 | **−78.527** | −33.146 |

  - The gradient w.r.t. the dropout logit was **NaN** for any non-zero count once the logit reached about 17. `remove_nan_gradients` set it to 0, so such a gene got no gradient to pull it back.
  - **Fix:** `Categorical(logits=stack([logits, 0]))`. Since `softmax([l, 0]) = [sigmoid(l), sigmoid(−l)]`, the weights are mathematically the same, but computed in log space. Parameters and shapes are unchanged.
  - Models trained before the fix were affected only where a dropout logit exceeded about 17. Retraining is recommended anyway.
- ✅ Every `from_parameterizer_output` raised `UnboundLocalError` for a `tf.Variable` or numpy input.
- ✅ `IndependentBernoulliDistribution.from_parameterizer_output` returned a plain tfp `Bernoulli`, not `cls`, and dropped `**kwargs`.
- ✅ `MixtureMultivariateNormalDiagDistribution` was missing from `distributions/__init__`.
- ✅ The Mixture docstring described the wrong `y_pred` layout.

## Worth a look

- **ZINB "zero" component** (documented in the class docstring, not a bug): it is NB(mean = 1e-7), not an exact point mass at 0. So `mean()` includes a `sigmoid(logits) * 1e-7` term, and `log_prob(0)` is very close to, but not exactly, the textbook ZINB value.
- The `StudenttDistribution` comment promises a finite variance, but df can be exactly 2.
