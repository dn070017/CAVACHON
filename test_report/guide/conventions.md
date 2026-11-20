# Conventions and commands

[← Overview](../UNIT_TEST_REPORT.md)

## How bugs are recorded

Each confirmed bug has a test that asserts the **correct** behaviour and is marked:

```python
@pytest.mark.xfail(strict=True, reason="BUG: ...")
```

The suite stays green while the bug exists. When a bug is fixed, its test starts passing, which `strict=True` turns into an XPASS failure. That prompts you to remove the marker. The test then stays as a regression test, with a `# regression:` comment explaining what used to go wrong.

List every open bug:

```bash
grep -rn "xfail" test | grep BUG
```

See the actual failures:

```bash
pixi run pytest --runxfail
```

## Bug IDs

| Prefix | Meaning |
|---|---|
| H1–H6 | High severity, first round (all fixed) |
| M1–M7 | Medium, coupled packages |
| C1–C4 | `Component` training path, latent (not used in production) |
| MD1–MD8 | `model` |
| W1–W7 | `workflow` |
| T1–T12 | `tools` |

Severity labels in the package pages:

- **Production impact: yes** — silently wrong results or crashes on a normal configuration.
- **Edge case** — needs an unusual input or configuration.
- **Latent** — the code path is not used by the workflow today.

## Running

```bash
pixi run pytest                       # whole suite, CPU, about 90 s
pixi run pytest test/tools            # one package
pixi run coverage run --source=cavachon -m pytest && pixi run coverage report
pixi run ruff check cavachon test
pixi run ruff format --check cavachon test
pixi run mypy                         # config in pyproject.toml [tool.mypy]
```

`pyproject.toml` sets `testpaths = ["test"]`. Without it, pytest would also collect `test_run_qa_hdeg.py`, a script at the repo root that starts a full workflow.

## Typing policy for `Optional`

1. If `None` can't actually happen, fix the type (initialise it, or use `d[key]` instead of `d.get(key)`).
2. If `None` is possible but invalid and a user can trigger it, narrow with `if x is None: raise ...` and a helpful message.
3. Use `assert x is not None` only for internal invariants.
4. `cast` and `# type: ignore[<code>]` are a last resort, always with a reason. In this codebase every remaining `type: ignore` marks either a known bug or an intentional signature difference.
