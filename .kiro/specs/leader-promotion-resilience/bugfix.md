# Bugfix Requirements Document

## Introduction

The engine-follower promotion system in `coordination.py` fails without recovery when the engine process becomes unresponsive. Followers encounter "Engine unavailable and could not promote" errors and give up permanently. The root cause is a combination of: (1) liveness checks that only test TCP socket connectivity rather than actual HTTP health, (2) no PID existence validation before trusting engine.json, (3) single-shot promotion attempts with no retry/backoff, and (4) stale engine.json blocking promotion even when the engine is clearly dead. This leaves all follower instances permanently broken until manual intervention.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN the engine process is alive (PID exists, port is in LISTEN state) but its HTTP server is unresponsive (connections refused or hanging) THEN the system reports the engine as "alive" via `_is_port_open` socket check and followers remain stuck unable to communicate

1.2 WHEN engine.json contains a PID that no longer exists (process crashed/was killed) THEN the system does not check PID liveness and instead attempts a socket connection to the stale port, fails, and performs only a single promotion attempt that can fail due to lock contention

1.3 WHEN a follower detects the engine is unresponsive during `initialize()` THEN the system attempts lock removal and re-acquisition exactly once, and if that single attempt fails (e.g., another process holds the lock briefly), it permanently gives up and becomes a non-functional follower

1.4 WHEN a follower's heartbeat detects the engine is dead and `promote_to_engine()` is called THEN the system makes a single promotion attempt with no retry or backoff, failing permanently if the lock file is transiently unavailable

1.5 WHEN the engine process becomes a zombie (PID exists but process is defunct) THEN the system's `_is_port_open` check fails but the lock file remains held, blocking all promotion attempts since `lock_path.unlink()` cannot remove a file locked by another (zombie) process on Windows

### Expected Behavior (Correct)

2.1 WHEN the engine process is alive but its HTTP server is unresponsive THEN the system SHALL perform an HTTP GET to `/health` endpoint with a 2-3 second timeout and only consider the engine alive if it receives a valid JSON response with `{"status": "ok"}`

2.2 WHEN engine.json contains a PID that no longer exists (verified via OS process table check) THEN the system SHALL immediately invalidate engine.json, remove the stale lock file, and proceed with self-promotion without waiting for socket timeout

2.3 WHEN a follower fails to acquire engineship during initialization or heartbeat-driven promotion THEN the system SHALL retry with exponential backoff (e.g., 0.5s, 1s, 2s) for up to 3 attempts before falling back to follower mode

2.4 WHEN a follower's promotion attempt fails due to lock contention THEN the system SHALL retry promotion with short backoff intervals rather than giving up after a single attempt

2.5 WHEN the engine process is a zombie or the lock file cannot be removed THEN the system SHALL detect the stale state via PID liveness check and engine.json staleness (e.g., last heartbeat timestamp older than 2x heartbeat interval), and force-create a new lock file to bypass the stale one

### Unchanged Behavior (Regression Prevention)

3.1 WHEN the engine process is healthy and its HTTP server responds normally THEN the system SHALL CONTINUE TO route followers to the existing engine without disruption

3.2 WHEN a follower successfully connects to a healthy engine for search requests THEN the system SHALL CONTINUE TO forward requests and return results without change to the search API contract

3.3 WHEN multiple processes start simultaneously and one acquires the lock first THEN the system SHALL CONTINUE TO correctly assign engine/follower roles based on lock acquisition order

3.4 WHEN the engine writes heartbeat timestamps to engine.json THEN the system SHALL CONTINUE TO update engine.json with port, PID, and started_at fields on each heartbeat cycle

3.5 WHEN a follower is successfully promoted to engine THEN the system SHALL CONTINUE TO start the background indexer and search index as part of the promotion sequence

3.6 WHEN the engine HTTP server fails to bind to the configured port THEN the system SHALL CONTINUE TO fall back to the next port and update engine.json accordingly

---

## Bug Condition (Formal)

```pascal
FUNCTION isBugCondition(X)
  INPUT: X of type PromotionAttempt {engine_pid, engine_port, engine_json_exists, http_health_ok, pid_alive, lock_held}
  OUTPUT: boolean

  // Bug triggers when engine.json exists but the engine is actually unreachable/dead
  RETURN X.engine_json_exists
     AND (NOT X.http_health_ok OR NOT X.pid_alive)
END FUNCTION
```

### Fix Checking Property

```pascal
// Property: Fix Checking — Resilient Promotion
FOR ALL X WHERE isBugCondition(X) DO
  result ← promote'(X)
  ASSERT result.promoted = true
     OR result.retry_count > 0
     OR result.became_follower_of_new_engine = true
  // The follower must either successfully self-promote, retry multiple times,
  // or discover a new valid engine — never permanently stuck
END FOR
```

### Preservation Checking Property

```pascal
// Property: Preservation Checking — Healthy Engine Unchanged
FOR ALL X WHERE NOT isBugCondition(X) DO
  ASSERT promote(X) = promote'(X)
  // When the engine is genuinely healthy, behavior is identical
END FOR
```
