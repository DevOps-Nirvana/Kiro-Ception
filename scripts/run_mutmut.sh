#!/usr/bin/env bash
#
# Run mutmut against a source module with hypothesis test files excluded.
#
# Usage:
#   ./scripts/run_mutmut.sh kiro_ception.search_utils
#   ./scripts/run_mutmut.sh "kiro_ception.tool_summaries*"
#   ./scripts/run_mutmut.sh kiro_ception.search_utils --max-children 4
#
# What this does:
#   1. Installs mutmut if not already available
#   2. Temporarily renames test files containing hypothesis (@given) to .bak
#      so pytest/mutmut never sees them (avoids fork-incompatibility crash)
#   3. Clears mutmut state and runs mutation testing
#   4. Restores all original test files (even on failure/interrupt)
#
# Why: mutmut v3 uses fork() for isolation. Hypothesis detects the PID change
# and raises FailedHealthCheck(differing_executors), crashing the test run.
# Excluding hypothesis files lets mutmut use all the regular pytest tests.
#
# The first argument is the mutmut module pattern (required).
# Any additional arguments are passed directly to `mutmut run`.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <module_pattern> [mutmut args...]"
    echo ""
    echo "Examples:"
    echo "  $0 kiro_ception.search_utils"
    echo "  $0 'kiro_ception.tool_summaries*'"
    echo "  $0 kiro_ception.ide_loader --max-children 4"
    exit 1
fi

MODULE_PATTERN="$1"
shift
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TESTS_DIR="$PROJECT_DIR/tests"

echo "=== Mutation testing: $MODULE_PATTERN ==="
echo ""

# --- Step 1: Ensure mutmut is available ---
if ! uv run python -c "import mutmut" 2>/dev/null; then
    echo "Installing mutmut..."
    uv add --dev mutmut 2>&1 | tail -1
fi

# --- Step 2: Temporarily hide hypothesis test files ---
echo "Excluding hypothesis test files..."

EXCLUDED_FILES=()
for test_file in "$TESTS_DIR"/test_*.py; do
    if grep -q "from hypothesis\|import hypothesis" "$test_file" 2>/dev/null; then
        mv "$test_file" "${test_file}.mutmut_bak"
        EXCLUDED_FILES+=("$test_file")
    fi
done

echo "  Excluded ${#EXCLUDED_FILES[@]} files containing hypothesis"

# --- Step 3: Restore on exit (trap) ---
restore_tests() {
    echo ""
    echo "Restoring hypothesis test files..."
    for f in "${EXCLUDED_FILES[@]}"; do
        if [ -f "${f}.mutmut_bak" ]; then
            mv "${f}.mutmut_bak" "$f"
        fi
    done
    echo "  Restored ${#EXCLUDED_FILES[@]} files."
}
trap restore_tests EXIT

# --- Step 4: Verify remaining tests pass ---
echo ""
echo "Verifying remaining tests pass..."
REMAINING=$(find "$TESTS_DIR" -name "test_*.py" | wc -l | tr -d ' ')
echo "  $REMAINING test files available"

if ! uv run pytest tests/ -x -q --tb=line 2>&1 | tail -3; then
    echo ""
    echo "ERROR: Tests fail after excluding hypothesis files. Aborting."
    exit 1
fi

# --- Step 5: Clear mutmut state and run ---
echo ""
echo "Running mutmut against: $MODULE_PATTERN"
echo "---"
rm -rf "$PROJECT_DIR/mutants/"
uv run mutmut run "$MODULE_PATTERN" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 || true

# --- Step 6: Show results ---
echo ""
echo "=== Results for $MODULE_PATTERN ==="
echo ""

RESULTS_OUTPUT=$(uv run mutmut results 2>&1)
SURVIVED=$(echo "$RESULTS_OUTPUT" | grep -c "survived" || true)
KILLED=$(echo "$RESULTS_OUTPUT" | grep -c "killed" || true)
TOTAL=$((SURVIVED + KILLED))

if [ "$TOTAL" -gt 0 ]; then
    KILL_RATE=$(echo "scale=1; $KILLED * 100 / $TOTAL" | bc)
    echo "  Mutants tested: $TOTAL"
    echo "  Killed: $KILLED"
    echo "  Survived: $SURVIVED"
    echo "  Kill rate: ${KILL_RATE}%"
    echo ""
    echo "  Survivors by function:"
    echo "$RESULTS_OUTPUT" | grep "survived" | \
        awk -F'x_' '{print $2}' | awk -F'__mutmut' '{print $1}' | \
        sort | uniq -c | sort -rn | head -10 | \
        while read count func; do echo "    $count  $func"; done
else
    echo "  No mutants were tested (mutmut may not have linked tests to functions)."
    echo ""
    echo "  This usually means mutmut's trampoline couldn't trace which tests"
    echo "  call the mutated functions. Check 'uv run mutmut results' for details."
fi

echo ""
echo "Commands:"
echo "  uv run mutmut results                    # Full results"
echo "  uv run mutmut show <mutant_name>         # Inspect a survivor"
echo "  uv run mutmut results | grep survived    # List all survivors"
