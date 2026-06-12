# Engine Promotion Resilience Bugfix Design

## Overview

The engine-follower coordination system in `coordination.py` has multiple failure modes that cause followers to permanently stall when the engine becomes unresponsive. The fix introduces: (1) HTTP-based health checks replacing raw TCP socket probes, (2) PID liveness validation via the OS process table, (3) exponential backoff retries for promotion attempts, and (4) staleness-aware lock recovery. The goal is to make followers self-heal when the engine dies without disrupting healthy engine operation.

## Glossary

- **Bug_Condition (C)**: The condition where `engine.json` exists but the engine process is unreachable — either its HTTP `/health` endpoint fails to respond with `{"status": "ok"}`, or its PID no longer exists in the OS process table
- **Property (P)**: The desired behavior when C holds — followers must retry promotion with backoff and eventually either self-promote or discover a new valid engine, never getting permanently stuck
- **Preservation**: The existing behavior for healthy engines (HTTP responding, PID alive) must remain unchanged — followers route to the existing engine, search works, roles are assigned correctly
- **`_is_port_open(port)`**: Current liveness check that only tests TCP socket connectivity to `127.0.0.1:port` — insufficient because a socket can accept connections without the HTTP server being functional
- **`engine.json`**: JSON file containing `{port, pid, started_at}` that followers read to find the engine
- **`engine.lock`**: filelock-managed lock file that grants exclusive engineship via `FileLock`
- **PID liveness**: Whether a process ID exists in the OS process table (checked via `os.kill(pid, 0)` on Unix or `OpenProcess` on Windows)
- **Staleness**: A engine.json whose `started_at` timestamp is older than 2× the heartbeat interval, indicating the engine stopped updating

## Bug Details

### Bug Condition

The bug manifests when `engine.json` exists and points to a engine that is actually dead, unresponsive, or zombied. The current code uses only `_is_port_open()` which tests raw TCP connectivity — this passes for half-alive processes and fails to detect PID death. When the check does fail, only a single promotion attempt is made with no retry.

**Formal Specification:**
```
FUNCTION isBugCondition(X)
  INPUT: X of type PromotionAttempt {engine_pid, engine_port, engine_json_exists, 
         http_health_ok, pid_alive, lock_removable}
  OUTPUT: boolean

  RETURN X.engine_json_exists
     AND (NOT X.http_health_ok OR NOT X.pid_alive)
END FUNCTION
```

### Examples

- **Dead process, stale json**: Engine PID 12345 was killed. `engine.json` still has `{port: 9111, pid: 12345}`. Follower calls `_is_port_open(9111)` which times out, tries once to unlink lock, fails (Windows file locking), gives up permanently. **Expected**: Follower detects PID 12345 is dead, invalidates engine.json, retries lock acquisition with backoff, succeeds.
- **Unresponsive HTTP**: Engine PID exists, port 9111 accepts TCP but HTTP handler is hung (GIL contention during heavy indexing). `_is_port_open` returns `True` so follower thinks engine is alive, stays in follower mode unable to get search results. **Expected**: Follower issues `GET /health` with 3s timeout, gets no valid JSON response, marks engine as unhealthy and begins promotion.
- **Zombie process (Windows)**: Engine process becomes zombie — PID exists but is defunct. Lock file is held. `_is_port_open` fails but `lock_path.unlink()` raises `PermissionError` on Windows because the zombie still holds the lock. Single attempt fails. **Expected**: Follower detects stale `started_at` timestamp (older than 2× heartbeat interval), uses PID check as secondary signal, force-creates new lock file to bypass stale one.
- **Transient lock contention**: Two followers detect dead engine simultaneously. First grabs lock, second fails its single attempt and gives up permanently. **Expected**: Second follower retries with backoff (0.5s, 1s, 2s), discovers first follower became new engine on retry, connects as follower of the new engine.

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- When the engine is healthy (HTTP `/health` returns `{"status": "ok"}` and PID is alive), followers continue routing to it without disruption
- Follower search forwarding via `POST /search` works identically — same request/response contract
- Initial lock acquisition race: multiple simultaneous starts still correctly assign engine/follower based on who acquires `FileLock` first
- Engine heartbeat continues writing `{port, pid, started_at}` to `engine.json` on each cycle
- Post-promotion startup sequence (background indexer + search index initialization) unchanged
- HTTP server port fallback (try configured port, then port+1) unchanged

**Scope:**
All inputs where `isBugCondition` is false (engine is genuinely healthy) must produce identical behavior to the unfixed code. This includes:
- Healthy engine heartbeat cycles
- Follower connecting to a responsive engine
- Normal search request forwarding
- Config and status API calls
- Multiple instances starting with a healthy engine already running

## Hypothesized Root Cause

Based on the bug analysis and code review, the root causes are:

1. **Insufficient liveness check (`_is_port_open`)**: Uses raw TCP `socket.create_connection()` which only tests if something is listening on the port. It does not verify the HTTP server is functional. A hung GIL, an overloaded server, or even a different process that claimed the port would pass this check.

2. **No PID validation**: `_read_engine_info()` returns the stored `{port, pid}` but neither `FollowerInstance.is_engine_alive()` nor `InstanceManager.initialize()` checks whether `pid` actually exists in the OS process table. A dead engine's `engine.json` is trusted indefinitely.

3. **Single-shot promotion with no retry**: Both `initialize()` (the force-acquire path) and `promote_to_engine()` attempt lock acquisition exactly once. If another process holds the lock transiently (e.g., during its own promotion race), the caller permanently gives up.

4. **`lock_path.unlink()` failure on Windows**: On Windows, `Path.unlink()` raises `PermissionError` if the file is locked by another process (even a zombie). The code catches `OSError` and continues to `try_acquire_engineship()`, which then fails because the stale lock file still exists. There is no fallback strategy.

5. **No staleness detection**: The heartbeat loop writes `started_at` to `engine.json` but the follower never checks how old this timestamp is. Even if the engine hasn't updated in minutes, the follower still trusts it.

## Correctness Properties

Property 1: Bug Condition - Resilient Promotion Under Dead Engine

_For any_ input where the bug condition holds (engine.json exists but the engine's HTTP `/health` is unresponsive OR the engine's PID does not exist in the OS process table), the fixed promotion logic SHALL either (a) successfully self-promote to engine after retrying with backoff, (b) discover that another process became the new valid engine and connect as follower to it, or (c) exhaust all retry attempts and remain a follower capable of retrying on the next heartbeat cycle — never permanently giving up.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

Property 2: Preservation - Healthy Engine Unaffected

_For any_ input where the bug condition does NOT hold (engine's HTTP `/health` responds with `{"status": "ok"}` AND the engine's PID is alive), the fixed code SHALL produce the same result as the original code, preserving all existing healthy-engine routing, search forwarding, role assignment, and heartbeat behavior without disruption.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `src/kiro_ception/coordination.py`

**1. Add `_is_pid_alive(pid)` helper function**:
- Use `os.kill(pid, 0)` on Unix (catches `OSError` for dead PIDs)
- Use `ctypes.windll.kernel32.OpenProcess()` on Windows with appropriate access flags
- Returns `bool` indicating if the process exists in the OS process table

**2. Add `_is_engine_healthy(port, pid)` function** replacing `_is_port_open` usage:
- First check: `_is_pid_alive(pid)` — fast OS-level check, returns `False` immediately if PID is dead
- Second check: HTTP `GET /health` to `http://127.0.0.1:{port}/health` with 3-second timeout
- Only returns `True` if both checks pass and HTTP response is `{"status": "ok"}`
- This replaces all calls to `_is_port_open` in liveness-checking contexts

**3. Add `_is_engine_info_stale(info)` function**:
- Reads `started_at` from engine.json
- Compares against `time.time()` — if older than 2× `heartbeat_interval_seconds`, returns `True`
- Used as additional signal when lock cannot be removed (zombie scenario)

**4. Modify `FollowerInstance.is_engine_alive()`**:
- Replace `_is_port_open(port)` with `_is_engine_healthy(port, pid)`
- Read PID from engine.json (already available via `_read_engine_info()`)

**5. Modify `InstanceManager.initialize()` — force-acquire path**:
- After detecting dead engine, add retry loop: attempt promotion up to 3 times with exponential backoff (0.5s, 1s, 2s)
- On each retry: re-check if a new valid engine appeared (another process may have promoted), if so become follower of new engine
- Use `_is_engine_healthy` instead of `follower.is_engine_alive()` for the initial check

**6. Modify `InstanceManager.promote_to_engine()`**:
- Add retry loop with exponential backoff (3 attempts: 0.5s, 1s, 2s)
- On each iteration: try `lock_path.unlink()`, then `try_acquire_engineship()`
- If `unlink` fails with `PermissionError` and `_is_engine_info_stale()` is true, force-create a new `FileLock` targeting a fresh lock file path (e.g., `engine.lock.new`) and atomically rename
- Between retries: re-check if engine came back alive or another process promoted

**7. Modify `_heartbeat_loop()`**:
- Use `_is_engine_healthy` for follower liveness check instead of `self._follower.is_engine_alive()`
- After failed promotion, don't permanently give up — the next heartbeat cycle will try again (this is already the case structurally, but log clearly)

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that simulate dead/unresponsive engines by mocking `_is_port_open`, file system operations, and `os.kill`. Run these on UNFIXED code to observe single-shot failures.

**Test Cases**:
1. **Dead PID Test**: Set up `engine.json` with a non-existent PID, attempt promotion — observe single failure with no retry (will fail on unfixed code)
2. **Unresponsive HTTP Test**: Mock `_is_port_open` to return `True` but have HTTP `/health` hang — observe follower stuck thinking engine is alive (will fail on unfixed code)
3. **Lock Contention Test**: Have two threads race for promotion after dead engine — observe second thread permanently gives up (will fail on unfixed code)
4. **Windows Lock Test**: Mock `Path.unlink()` to raise `PermissionError` — observe promotion permanently fails (will fail on unfixed code)

**Expected Counterexamples**:
- `promote_to_engine()` returns `False` after single attempt with no retry
- `is_engine_alive()` returns `True` for a process that cannot serve HTTP requests
- `initialize()` falls through to permanent follower with dead engine still referenced

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL X WHERE isBugCondition(X) DO
  result := promote_to_engine_fixed(X)
  ASSERT result.promoted = true
     OR result.retry_count > 0
     OR result.connected_to_new_engine = true
  ASSERT result.permanently_stuck = false
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL X WHERE NOT isBugCondition(X) DO
  ASSERT initialize_original(X) = initialize_fixed(X)
  ASSERT promote_original(X) = promote_fixed(X)
  ASSERT is_engine_alive_original(X) = is_engine_alive_fixed(X)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many combinations of healthy engine states (various ports, PIDs, timestamps)
- It catches regressions where the new health check incorrectly flags a healthy engine
- It verifies that search forwarding, status queries, and role assignment are unchanged

**Test Plan**: Observe behavior on UNFIXED code first for healthy engine scenarios, then write property-based tests capturing that behavior and verify it's unchanged after the fix.

**Test Cases**:
1. **Healthy Engine Routing**: Verify that when `/health` returns ok and PID is alive, `is_engine_alive` returns `True` and no promotion is attempted — same as original
2. **Search Forwarding Preservation**: Verify follower search requests are forwarded identically when engine is healthy
3. **Role Assignment Preservation**: Verify `initialize()` assigns "follower" when a healthy engine exists, without any extra delay from new health checks
4. **Heartbeat Write Preservation**: Verify engine continues writing `{port, pid, started_at}` to `engine.json` on schedule

### Unit Tests

- Test `_is_pid_alive()` with current PID (true), non-existent PID (false), and PID 0 edge case
- Test `_is_engine_healthy()` with mock HTTP responses (healthy, timeout, wrong JSON, connection refused)
- Test `_is_engine_info_stale()` with fresh timestamp vs very old timestamp
- Test retry logic: mock lock acquisition to fail N times then succeed, verify backoff delays
- Test `promote_to_engine()` retry exhaustion: all 3 attempts fail, returns `False` without crash

### Property-Based Tests

- Generate random `engine.json` states (port in 1024-65535, PID from valid/invalid ranges, timestamps from recent to ancient) and verify `_is_engine_healthy` classification matches oracle
- Generate random sequences of lock-acquisition outcomes (success/failure per attempt) and verify retry logic terminates within bounded attempts
- Generate random healthy-engine configurations and verify `initialize()` produces identical role assignment as the original code

### Integration Tests

- Full lifecycle: start engine, start follower, kill engine, verify follower promotes within 2× heartbeat interval
- Race condition: start 3 instances simultaneously with dead engine.json, verify exactly one becomes engine
- Stale lock recovery: set up unremovable lock file + stale engine.json, verify promotion eventually succeeds
