# `cavachon/app`

[← Overview](../UNIT_TEST_REPORT.md)

| Status | Tests | Passed | Open bugs | Coverage | Coupling |
|---|---|---|---|---|---|
| ✅ Clean | [`test/app`](../../test/app) | 11 | 0 | 87% | Low: depends on 1 package, used by 0 |

- **Depends on:** workflow
- **Used by:** nothing (entry point)
- **Mocked in tests:** `Workflow`.

## Worth a look

- Importing `app/cavachon_application.py` enables GPU memory growth and globally silences `FutureWarning`.
