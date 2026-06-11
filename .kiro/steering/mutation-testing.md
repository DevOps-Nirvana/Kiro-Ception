---
inclusion: manual
---

# Mutation Testing with mutmut

## Overview

Mutation testing validates test quality by injecting small bugs (mutants) into source code and checking whether the test suite catches them. A surviving mutant means no test failed despite the introduced bug — a gap in coverage.

We use **mutmut v3+** which is configured in `pyproject.toml` and uses a fork-based trampoline execution model.

mutmut is an **on-demand tool**, not a permanent dev dependency. Install it when you need it:

```bash
uv add --dev mutmut
```

Or run it without installing via:

```bash
uvx mutmut run "kiro_ception.tool_summaries*"
```

## Configuration

```toml
# pyproject.toml
[tool.mutmut]
source_paths = ["src/kiro_ception/"]
```

mutmut auto-discovers which tests exercise which functions via its trampoline system. You can optionally narrow the test scope:

```toml
pytest_add_cli_args_test_selection = ["tests/test_tool_summaries.py"]
```

## Running

```bash
# Mutate a single module (recommended — fast, focused)
uv run mutmut run "kiro_ception.tool_summaries*"

# Mutate a single function
uv run mutmut run "kiro_ception.tool_summaries._extract_outcome*"

# Mutate everything (slow — runs all tests per mutant)
uv run mutmut run
```

## Reviewing Results

```bash
# Summary of all checked mutants
uv run mutmut results

# Filter to a specific module
uv run mutmut results 2>&1 | grep "tool_summaries"

# Count killed vs survived
uv run mutmut results 2>&1 | grep "tool_summaries.*survived" | wc -l
uv run mutmut results 2>&1 | grep "tool_summaries.*killed" | wc -l

# Show what a specific mutant changed
uv run mutmut show "kiro_ception.tool_summaries.x_generate_tool_summary__mutmut_3"
```

## Hypothesis Incompatibility

mutmut v3 uses `fork()` to isolate each mutant run. Hypothesis detects the PID change and raises `FailedHealthCheck: differing_executors`, causing the initial "clean test" step to fail before any mutations are tested.

**Workaround**: Exclude hypothesis-based test files from `pytest_add_cli_args_test_selection`:

```toml
# Only run non-hypothesis tests for mutation testing
pytest_add_cli_args_test_selection = ["tests/test_tool_summaries.py"]
```

Do NOT include files like `test_pbt_tool_summaries.py` or any file using `@given` decorators. The hypothesis tests still provide value for fuzz-finding (they found the newline-in-error bug), but they can't participate in mutmut's fork-based execution.

## Interpreting Surviving Mutants

Not all survivors are worth killing:

| Mutation type | Worth testing? |
|---------------|---------------|
| `x[:97]` → `x[:98]` (boundary off-by-one) | Yes — add exact-length assertions |
| `.get("key", "")` → `.get("key", None)` (default change) | Yes if code path differs — test with missing keys |
| `"error"` → `"XXerrorXX"` (keyword mutation) | Yes — test that each keyword triggers detection |
| `replace("\n", " ")` → `replace("XX\nXX", " ")` (pattern mutation) | Yes — test with actual newlines in input |
| `description = "(no input)"` → `"(NO INPUT)"` (string case) | Maybe — only if exact text is part of the contract |
| Logger messages mutated | No — don't test log strings |

## Workflow

1. Run mutmut against a specific module
2. Review survived mutants with `mutmut show`
3. Group them by category (boundary, fallback, keyword, etc.)
4. Write targeted tests that pin down the exact behavior
5. Add tests to the existing test file for that module (don't create separate "mutation test" files)
6. Re-run mutmut to confirm kills

## Clearing State

mutmut caches results in the `mutants/` directory. To start fresh:

```bash
rm -rf mutants/
```

The `mutants/` directory is gitignored.
