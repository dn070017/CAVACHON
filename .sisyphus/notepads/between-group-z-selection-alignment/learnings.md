# between-group-z-selection-alignment Learnings

## 2026-06-05 — Completion summary

1. **API change**: Added `donor_components: list[str] | None = None` to `between_clusters`.
   - `None` defaults to `[component]` (backward-compatible).
   - Duplicates deduped preserving order.
   - Empty/invalid → fail fast with explicit ValueError.

2. **Multi-component substitution**: `_encode_z_pool` now returns `dict[str, np.ndarray]`.
   - Donor pools encoded per batch for all donor components simultaneously.
   - `_compute_substituted_x_means` samples indices once, applies across all pools.

3. **Semantic separation**: `component`/`modality` remain pure DEG target.
   - `model.decode(components=[component])` unchanged.
   - Distribution lookup still uses `component`+`modality`.

4. **QA script**: `test_run_qa_hdeg.py` updated with explicit `donor_components=[component]`.

5. **Verification**: All scenarios passed — legacy, explicit, multi-component, invalid names, empty list.

6. **Files modified**: 
   - `cavachon/tools/hierarchical_differential_analysis.py` (107 insertions, 14 deletions)
   - `test_run_qa_hdeg.py` (3 lines added)
