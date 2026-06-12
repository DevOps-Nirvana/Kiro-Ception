# Tech Debt & Future Work

## Search — workspace decode fallback (low priority)

**Added:** 2026-06-12
**Context:** `search.py` SearchIndex.search and `cache.py` fts_search

The workspace filter in both the vector search path and FTS path includes a
fallback that decodes base64-encoded workspace values at query time. This
handles messages indexed before the `_decode_workspace_dir_name` fix (where
Kiro's `_`-as-padding scheme was misinterpreted, storing raw base64 strings
like `YzpcVXNlcnN...dw__` instead of decoded paths like `c:\Users\...`).

After all users have run `rescan(full=True)` (or enough time passes that all
sessions get re-indexed naturally via mtime changes), these fallback paths
become dead code:

- `search.py` line ~350: the `_decode_workspace_dir_name(stored_ws)` fallback
- `cache.py` fts_search: the `OR m.workspace LIKE ? || '%'` clause matching
  base64-encoded workspace

**Removal criteria:** Safe to remove once no messages in any user's DB have
base64-encoded workspace values. In practice, a couple release cycles after
the fix ships.

## Engine spawn — Windows venv shim creates intermediate PID (FIXED)

**Added:** 2026-06-12
**Fixed:** 2026-06-12

Fix: `_find_engine_executable()` now uses `sys._base_executable` on Windows to
skip the venv launcher shim. `spawn_engine()` no longer tracks the Popen PID —
it only polls `engine.json` for the real engine to appear and become healthy.

## Process separation — steering/docs still reference old terms

**Added:** 2026-06-12

Some steering files and documentation still use "leader" terminology from the
old architecture. The codebase uses "engine" consistently now but grep may
find stale references in:

- `.kiro/steering/architecture.md`
- `.kiro/steering/development.md`
- `.kiro/steering/operations.md`

Low priority — these are internal development docs, not user-facing.
