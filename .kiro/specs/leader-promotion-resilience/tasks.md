# Implementation Plan

## Overview

Fix the engine-follower coordination system so that followers can recover from dead/unresponsive engines. The implementation follows the exploratory bugfix workflow: write bug condition tests first (confirm the bug exists), write preservation tests (capture healthy-engine behavior), implement the fix, then validate both sets of tests pass.

## Tasks

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Resilient Promotion Under Dead Engine
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate followers permanently stall when the engine is dead/unresponsive
  - **Scoped PBT Approach**: Generate `PromotionAttempt` inputs where `isBugCondition(X)` holds: `engine_json_exists=True AND (NOT http_health_ok OR NOT pid_alive)`. Test combinations of dead PIDs, unresponsive HTTP, and locked files.
  - Bug condition from design: `X.engine_json_exists AND (NOT X.http_health_ok OR NOT X.pid_alive)`
  - Test that for all bug-condition inputs, `promote_to_engine()` either succeeds, retries more than once, or discovers a new valid engine — never permanently gives up after a single attempt
  - Mock `_is_port_open` to return False (dead engine), mock `Path.unlink` to raise `PermissionError` (Windows lock), mock engine.json to contain stale PID
  - Run test on UNFIXED code - expect FAILURE (confirms the bug: single-shot promotion with no retry)
  - Document counterexamples found (e.g., "promote_to_engine() returns False after single attempt with no retry when lock_path.unlink raises PermissionError")
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Healthy Engine Unaffected
  - **IMPORTANT**: Follow observation-first methodology
  - Observe: When engine is healthy (HTTP `/health` responds ok, PID alive), `is_engine_alive()` returns `True` on unfixed code
  - Observe: When healthy engine exists, `initialize()` returns "follower" and does not attempt promotion on unfixed code
  - Observe: When engine holds lock and is responsive, `promote_to_engine()` is never triggered during heartbeat on unfixed code
  - Write property-based test: for all inputs where `NOT isBugCondition(X)` (healthy engine — port open, process alive), the unfixed `FollowerInstance.is_engine_alive()` returns `True` and `InstanceManager.initialize()` assigns "follower" role without attempting lock removal
  - Generate random healthy-engine configurations: valid ports (1024-65535), alive PIDs, responsive HTTP mocks
  - Verify tests pass on UNFIXED code (confirms baseline healthy-engine behavior is captured correctly)
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 3. Fix for engine promotion resilience under dead/unresponsive engine

  - [x] 3.1 Add `_is_pid_alive(pid: int) -> bool` helper function
    - Use `os.kill(pid, 0)` on Unix (catch `OSError` with `errno.ESRCH` for dead PIDs, `errno.EPERM` means alive but no permission)
    - Use `ctypes.windll.kernel32.OpenProcess()` on Windows with `PROCESS_QUERY_LIMITED_INFORMATION` access
    - Return `False` for PID <= 0 edge cases
    - _Bug_Condition: isBugCondition(X) where NOT X.pid_alive — PID does not exist in OS process table_
    - _Expected_Behavior: detect dead PIDs immediately without relying on socket timeout_
    - _Preservation: when PID is alive, returns True — no change to healthy-engine flow_
    - _Requirements: 2.2, 2.5_

  - [x] 3.2 Add `_is_engine_healthy(port: int, pid: int) -> bool` function
    - First check: `_is_pid_alive(pid)` — fast OS-level check, return `False` immediately if PID dead
    - Second check: HTTP GET `http://127.0.0.1:{port}/health` with 3-second timeout
    - Only return `True` if both PID check passes AND HTTP response contains `{"status": "ok"}`
    - Replace `_is_port_open` usage in liveness-checking contexts with this function
    - _Bug_Condition: isBugCondition(X) where NOT X.http_health_ok — HTTP health check fails even if TCP port is open_
    - _Expected_Behavior: only consider engine alive when both PID exists AND HTTP /health responds correctly_
    - _Preservation: for healthy engines both checks pass, returning True as before_
    - _Requirements: 2.1, 2.2_

  - [x] 3.3 Add `_is_engine_info_stale(info: dict) -> bool` function
    - Read `started_at` from engine.json info dict
    - Compare against `time.time()` — return `True` if older than 2× `heartbeat_interval_seconds` from config
    - Used as additional signal when lock cannot be removed (zombie/Windows scenario)
    - _Bug_Condition: zombie engine where PID exists but process is defunct, lock file held_
    - _Expected_Behavior: detect stale engine.json even when lock removal fails_
    - _Preservation: for healthy engines with fresh heartbeat, returns False_
    - _Requirements: 2.5_

  - [x] 3.4 Modify `FollowerInstance.is_engine_alive()` to use health check
    - Replace `_is_port_open(port)` with `_is_engine_healthy(port, pid)`
    - Read PID from `_read_engine_info()` (already called to get port)
    - If engine_info is None, return False
    - _Bug_Condition: engine_json_exists but engine HTTP is unresponsive or PID dead_
    - _Expected_Behavior: correctly report engine as NOT alive when HTTP /health fails_
    - _Preservation: healthy engine still reported as alive_
    - _Requirements: 2.1, 2.2, 3.1_

  - [x] 3.5 Modify `InstanceManager.initialize()` to use health check and retry
    - Replace `follower.is_engine_alive()` check with `_is_engine_healthy(port, pid)` using info from engine.json
    - Add retry loop for force-acquire path: up to 3 attempts with exponential backoff (0.5s, 1s, 2s)
    - On each retry: re-check if a new valid engine appeared (re-read engine.json, call `_is_engine_healthy`), if so become follower of new engine
    - On final failure: fall back to follower mode (can retry on next heartbeat)
    - _Bug_Condition: single-shot promotion fails due to transient lock contention_
    - _Expected_Behavior: retry with backoff, either self-promote or discover new engine_
    - _Preservation: when healthy engine exists, immediately returns "follower" without retries_
    - _Requirements: 2.3, 2.4, 3.3_

  - [x] 3.6 Modify `InstanceManager.promote_to_engine()` to retry with backoff
    - Add retry loop: up to 3 attempts with exponential backoff (0.5s, 1s, 2s)
    - On each iteration: try `lock_path.unlink()`, then `try_acquire_engineship()`
    - If `unlink` raises `PermissionError` and `_is_engine_info_stale()` is True: force-create new lock file path as fallback
    - Between retries: re-check if engine came back alive or another process promoted (call `_is_engine_healthy`)
    - If engine came back alive during retry, abort promotion and return False (remain follower of recovered engine)
    - _Bug_Condition: lock_path.unlink() fails with PermissionError on Windows, single attempt gives up_
    - _Expected_Behavior: retry with backoff, handle stale locks via staleness detection_
    - _Preservation: when promotion succeeds on first try (no contention), behavior unchanged_
    - _Requirements: 2.3, 2.4, 2.5_

  - [x] 3.7 Modify `_heartbeat_loop()` to use `_is_engine_healthy`
    - Replace `self._follower.is_engine_alive()` with `_is_engine_healthy(port, pid)` reading from engine.json
    - After failed promotion, log clearly that next heartbeat cycle will retry (already structurally the case)
    - _Bug_Condition: heartbeat uses _is_port_open which misses HTTP-level failures_
    - _Expected_Behavior: heartbeat detects truly dead engines via HTTP + PID check_
    - _Preservation: healthy engine heartbeat cycles unchanged_
    - _Requirements: 2.1, 2.2, 3.4_

  - [x] 3.8 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Resilient Promotion Under Dead Engine
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior (retry with backoff, discover new engine, or exhaust retries without permanent stall)
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed — promotion retries and succeeds)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 3.9 Verify preservation tests still pass
    - **Property 2: Preservation** - Healthy Engine Unaffected
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions — healthy engine behavior unchanged)
    - Confirm all tests still pass after fix (no regressions)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 4. Checkpoint - Ensure all tests pass
  - Run full test suite: `pytest tests/` from `Kiro-Ception-upstream` directory
  - Ensure both Property 1 (bug condition) and Property 2 (preservation) tests pass
  - Ensure no existing tests regressed
  - Run `ruff check src/` to verify no lint issues introduced (100 char line length)
  - Ask the user if questions arise

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1", "2"] },
    { "id": 1, "tasks": ["3.1", "3.3"] },
    { "id": 2, "tasks": ["3.2"] },
    { "id": 3, "tasks": ["3.4"] },
    { "id": 4, "tasks": ["3.5", "3.6"] },
    { "id": 5, "tasks": ["3.7"] },
    { "id": 6, "tasks": ["3.8"] },
    { "id": 7, "tasks": ["3.9"] },
    { "id": 8, "tasks": ["4"] }
  ]
}
```

## Notes

- **Test framework**: pytest + Hypothesis for property-based tests
- **Platform**: Windows primary — pay special attention to PID checks (`ctypes.windll`) and file locking (`PermissionError` on locked files)
- **Line length**: 100 characters (ruff configuration)
- **Key insight**: The exploration test (Property 1) is designed to FAIL on unfixed code. This is correct and expected. Do not "fix" the test — implement the code fix in task 3, then re-verify in task 3.8.
- **Mocking strategy**: Use `unittest.mock.patch` to mock `_is_port_open`, `Path.unlink`, `os.kill`, HTTP responses, and `FileLock.acquire` for both exploration and preservation tests
- **Backoff timing**: Use `time.sleep` mocking in tests to avoid actual delays (patch `time.sleep` to no-op)
