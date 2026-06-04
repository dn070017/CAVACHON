# Task 6 — Validation Evidence Index

**Feature:** `HierarchicalDifferentialAnalysis`  
**Date:** 2026-06-05  
**Overall Verdict:** ✅ PASS

## Summary Table

| Command | Description | Exit Code | Result |
|---------|-------------|-----------|--------|
| 1 | Import sanity — `HierarchicalDifferentialAnalysis` | 0 | ✅ PASS |
| 2 | DEG regression — `test_differential_analysis.py` (14 checks) | 0 | ✅ PASS |
| 3 | Model encode/decode regression — `test_model_encode_decode.py` (38 checks) | 0 | ✅ PASS |
| 4 | Existing imports regression — `DifferentialAnalysis, ClusterAnalysis, AttributionAnalysis` | 0 | ✅ PASS |

## Evidence Files

| File | Contents |
|------|----------|
| `task-6-import.txt` | Output of `HierarchicalDifferentialAnalysis` import check |
| `task-6-deg-regression.txt` | Full output of `test_differential_analysis.py` |
| `task-6-model-regression.txt` | Full output of `test_model_encode_decode.py` |
| `task-6-import-regression.txt` | Output of existing tools import check |
| `task-6-validation.txt` | Consolidated validation summary |

## Notes

- `test/utils/test_TensorUtils.py` is a **pre-existing failure** (module renamed `TensorUtils` → `tensor_utils`) — unrelated to this work, not run in this suite.
- All warnings in output are pre-existing TensorFlow/Keras/MuData FutureWarnings — not errors.
- `HierarchicalDifferentialAnalysis` is correctly exported from `cavachon/tools/__init__.py`.

## Overall Verdict: ✅ PASS

All 4 targeted suites exited 0. No regressions introduced by the `HierarchicalDifferentialAnalysis` feature.
