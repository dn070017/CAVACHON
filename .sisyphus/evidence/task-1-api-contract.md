# v1 API Contract: `HierarchicalDifferentialAnalysis`

## Status
Frozen v1 contract. Documentation only. No production code.

## Class
`HierarchicalDifferentialAnalysis` SHALL be a subclass of `DifferentialAnalysis`.

## Constructor
The constructor MUST keep the same core arguments as `DifferentialAnalysis.__init__`:

```python
def __init__(
    self,
    mdata: mu.MuData,
    model: tf.keras.Model,
    batch_effect_colnames: Optional[Dict[str, List[str]]] = None,
    distribution_names: Optional[Dict[str, str]] = None,
    batch_effect_encoders: Dict[str, Dict[str, LabelEncoder]] = dict(),
) -> None
```

## Public Method
### `between_clusters`
v1 public signature:

```python
def between_clusters(
    self,
    donor_cluster: str,
    recipient_cluster: str,
    component: str,
    modality: str,
    n_samples: int = 10,
    seed: Optional[int] = None,
    batch_size: int = 128,
) -> pd.DataFrame
```

### Parameter contract
- `donor_cluster: str` — donor cluster name.
- `recipient_cluster: str` — recipient cluster name.
- `component: str` — component used for staged encode / hierarchical encode / decode.
- `modality: str` — modality decoded from the selected component.
- `n_samples: int = 10` — number of donor-cluster `z` cells sampled per intervention.
- `seed: Optional[int] = None` — RNG seed for deterministic donor-pool sampling.
- `batch_size: int = 128` — forward-pass batch size used for the staged model calls.

## Required behavior
1. Use the staged inference path in this order:
   - `model.encode(...)`
   - `model.hierarchical_encode(...)`
   - donor-pool substitution on `z`
   - `model.decode(...)` for the target component only
2. Always pass the full component graph to `encode` and `hierarchical_encode`.
   - Child-only calls are invalid because they silently drop parent conditioning.
3. Sampling strategy MUST be donor-pool sampling:
   - randomly draw `n_samples` cells from the donor cluster's `z`
   - substitute those latent values into recipient cells
4. Reproducibility guarantee:
   - fixed `seed` MUST produce identical results across runs.

## Return value
`between_clusters(...)` MUST return `pd.DataFrame` with legacy columns preserved **exactly** and intervention columns appended.

### Legacy columns (exact order, exact names)
1. `Mean(A)`
2. `Mean(B)`
3. `P(A>B|Z)`
4. `P(B>A|Z)`
5. `K(A>B|Z)`
6. `K(B>A|Z)`

### Intervention columns (exact names and dtypes)
- `InterventionType` (`str`)
- `DonorCluster` (`str`)
- `RecipientCluster` (`str`)
- `SamplingStrategy` (`str`)
- `RandomSeed` (`int` or `None`)
- `MeanDelta(Substituted-Original)` (`float`)

### Output schema rule
- Legacy columns MUST remain in the same order.
- Intervention columns MUST be additive (appended after legacy columns).

## Failure modes
- Missing donor or recipient cluster MUST raise `ValueError`.
- Empty recipient cluster MUST raise `ValueError`.

## Non-goals for v1
- No production code changes in this task.
- No new output columns beyond the additive intervention columns above.
- No change to the legacy DEG schema.
