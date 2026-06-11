"""Leader-follower pattern for multi-instance coordination.

Only one process (the leader) holds embeddings in RAM and runs the indexer.
All other processes (followers) forward requests to the leader via localhost HTTP.
Automatic failover: if the leader dies, a follower promotes itself.

Zombie handling: if the leader process is alive but its HTTP server is dead,
the successor will kill it by PID to release the file lock, then take over.
On Windows, file locks cannot be removed by another process — the holder must die.
"""

import ctypes
import errno
import json
import logging
import os
import random
import signal
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from filelock import FileLock, Timeout

from .config import get_config, expand_path

logger = logging.getLogger(__name__)

# Track process start time for uptime reporting
_process_start_time = time.time()

# Lock file location
_LOCK_FILE_NAME = "leader.lock"
_LEADER_INFO_NAME = "leader.json"

# Simple HTML dashboard served at GET /
_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kiro Ception</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #1a1a2e; color: #e0e0e0; padding: 2rem; line-height: 1.5; }
  h1 { color: #64ffda; margin-bottom: 0.5rem; }
  h2 { color: #80cbc4; margin: 1.5rem 0 0.5rem; font-size: 1.1rem; }
  .subtitle { color: #888; margin-bottom: 2rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
  @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
  .card { background: #16213e; border-radius: 8px; padding: 1.2rem; }
  .card h2 { margin-top: 0; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  td { padding: 0.3rem 0.5rem; border-bottom: 1px solid #1a1a2e; }
  td:first-child { color: #80cbc4; white-space: nowrap; width: 40%; }
  .status-badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
                  font-size: 0.8rem; font-weight: 600; }
  .status-idle { background: #2e7d32; color: #c8e6c9; }
  .status-indexing { background: #e65100; color: #ffe0b2; }
  .status-starting { background: #1565c0; color: #bbdefb; }
  .status-error { background: #b71c1c; color: #ffcdd2; }
  .loading { color: #888; font-style: italic; }
  #error { color: #ef5350; margin: 1rem 0; }
  .refresh { color: #64ffda; cursor: pointer; font-size: 0.85rem; float: right; }
  .refresh:hover { text-decoration: underline; }
  .countdown-ring { display: inline-block; vertical-align: middle; margin-left: 0.5rem; }
  .countdown-ring svg { width: 18px; height: 18px; transform: rotate(-90deg); }
  .countdown-ring circle { fill: none; stroke: #64ffda; stroke-width: 2.5;
    stroke-dasharray: 44; stroke-dashoffset: 0;
    transition: none; }
  .countdown-ring.active circle {
    animation: countdown 10s linear forwards; }
  @keyframes countdown {
    from { stroke-dashoffset: 0; }
    to { stroke-dashoffset: 44; }
  }
</style>
</head>
<body>
<h1>Kiro Ception</h1>
<p class="subtitle">Status Dashboard</p>
<div id="error"></div>
<div class="grid">
  <div class="card">
    <span class="refresh" onclick="load()">refresh<span class="countdown-ring" id="ring"><svg viewBox="0 0 18 18"><circle cx="9" cy="9" r="7"/></svg></span></span>
    <h2>Indexing Status</h2>
    <div id="status"><p class="loading">Loading...</p></div>
  </div>
  <div class="card">
    <h2>Configuration</h2>
    <div id="config"><p class="loading">Loading...</p></div>
  </div>
</div>
<script>
function esc(s) { return String(s).replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function badge(state) {
  const cls = state === 'idle' ? 'status-idle' : state === 'indexing' ? 'status-indexing'
    : state === 'starting' ? 'status-starting' : 'status-error';
  return `<span class="status-badge ${cls}">${esc(state)}</span>`;
}
function row(k, v) { return `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`; }
function renderStatus(d) {
  let h = '<table>';
  h += `<tr><td>State</td><td>${badge(d.state)}</td></tr>`;
  h += row('Progress', d.progress_percent + '%');
  h += row('Sessions (total)', d.sessions_total);
  h += row('Sessions (processed)', d.sessions_processed);
  h += row('Sessions (unchanged)', d.sessions_unchanged);
  h += row('Messages embedded', d.messages_embedded);
  h += row('Messages cached', d.messages_cached);
  h += row('Errors', d.errors);
  h += row('Rate', d.rate_msg_per_sec + ' msg/s');
  h += row('Elapsed', Math.round(d.elapsed_seconds) + 's');
  h += row('Uptime', Math.round(d.uptime_seconds) + 's');
  h += row('Memory used', (d.memory_used_mb || 0) + ' MB' + (d.memory_used_percent ? ' (' + d.memory_used_percent + '%)' : ''));
  if (d.memory_limit_mb) h += row('Memory limit', d.memory_limit_mb + ' MB');
  h += row('Embedding count', d.embedding_count);
  h += row('DB size', (d.db_size_mb || 0) + ' MB');
  h += row('Schema version', d.schema_version || '?');
  h += row('FTS enabled', d.fts_enabled ? 'Yes' : 'No');
  h += row('Search ready', d.search_ready ? 'Yes' : 'No');
  h += row('Search messages', d.search_message_count);
  if (d.last_error) h += row('Last error', d.last_error);
  if (d.last_completed_at) h += row('Last completed', d.last_completed_at);
  h += '</table>';
  document.getElementById('status').innerHTML = h;
}
function renderConfig(d) {
  let h = '<table>';
  h += row('Version', d.version?.kiro_ception || '?');
  h += row('Python', d.version?.python || '?');
  h += row('Platform', d.version?.platform || '?');
  h += row('Role', d.instance?.role || '?');
  h += row('PID', d.instance?.pid || '?');
  h += row('Port', d.instance?.port || d.server?.leader_port || '?');
  h += row('Backend', d.embedding?.backend || '?');
  h += row('Model', d.embedding?.model || '?');
  h += row('Dimensions', d.embedding?.dimensions || 'auto');
  h += row('Embeddings', d.cache?.embedding_count || 0);
  h += row('Messages', d.cache?.message_count || 0);
  h += row('Sessions indexed', d.cache?.indexed_sessions || 0);
  h += row('Memory limit', d.memory?.effective_limit_mb + ' MB' || '?');
  h += row('Rescan interval', d.indexing?.rescan_interval_minutes + ' min');
  h += row('Heartbeat interval', (d.server?.heartbeat_interval_seconds || 30) + 's');
  h += row('Peers enabled', d.peers?.enabled ? 'Yes' : 'No');
  if (d.peers?.enabled) h += row('Peer nodes', d.peers.nodes?.join(', ') || 'none');
  h += '</table>';
  document.getElementById('config').innerHTML = h;
}
async function load() {
  try {
    const [statusRes, configRes] = await Promise.all([
      fetch('/status'), fetch('/config')
    ]);
    renderStatus(await statusRes.json());
    renderConfig(await configRes.json());
    document.getElementById('error').textContent = '';
  } catch(e) {
    document.getElementById('error').textContent = 'Failed to load: ' + e.message;
  }
  // Restart countdown animation
  const ring = document.getElementById('ring');
  ring.classList.remove('active');
  void ring.offsetWidth; // Force reflow to restart animation
  ring.classList.add('active');
}
load();
setInterval(load, 10000);
</script>
</body>
</html>"""


def _get_lock_path() -> Path:
    """Get the leader lock file path."""
    config = get_config()
    cache_path = expand_path(config.embedding.cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path / _LOCK_FILE_NAME


def _get_leader_info_path() -> Path:
    """Get the leader info file path (stores port + pid)."""
    config = get_config()
    cache_path = expand_path(config.embedding.cache_dir)
    return cache_path / _LEADER_INFO_NAME


def _write_leader_info(port: int, pid: int):
    """Write leader info to disk atomically so followers can find it.

    Uses write-to-temp + rename to avoid partial-write corruption if the
    process crashes mid-write or the disk fills up.
    """
    info_path = _get_leader_info_path()
    info = {"port": port, "pid": pid, "started_at": time.time()}
    dir_path = info_path.parent
    fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(info, f)
            f.flush()
            os.fsync(f.fileno())
        Path(tmp_path).replace(info_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _read_leader_info() -> dict | None:
    """Read leader info from disk."""
    info_path = _get_leader_info_path()
    try:
        with open(info_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _is_port_open(port: int) -> bool:
    """Check if the leader's HTTP port is responding."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            return True
    except (ConnectionRefusedError, OSError, TimeoutError):
        return False


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID exists.

    Uses os.kill(pid, 0) on Unix, OpenProcess on Windows.
    """
    if pid <= 0:
        return False

    if sys.platform == "win32":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError as e:
            if e.errno == errno.EPERM:
                return True  # exists but no permission
            return False


def _is_leader_healthy(port: int, pid: int, retries: int = 2) -> bool:
    """Check if the leader is both alive AND serving HTTP.

    A process that holds the lock but doesn't respond to HTTP /health
    is a zombie and must be killed.

    Retries the health check to avoid false-positive kills when the leader's
    HTTP thread is momentarily busy (e.g., serving a large search request).
    """
    if not _is_pid_alive(pid):
        return False

    import urllib.request

    for attempt in range(retries):
        try:
            url = f"http://127.0.0.1:{port}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                if body.get("status") == "ok":
                    return True
        except Exception as e:
            logger.debug(
                f"Health check attempt {attempt + 1}/{retries} failed for "
                f"port={port} pid={pid}: {type(e).__name__}: {e}"
            )
        if attempt < retries - 1:
            time.sleep(1)

    return False


def _kill_process(pid: int) -> bool:
    """Forcefully kill a process by PID. Returns True if killed or already dead."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        # Never kill ourselves
        return False

    logger.warning(f"Killing zombie leader process (pid={pid})")

    if sys.platform == "win32":
        PROCESS_TERMINATE = 0x0001
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
        if handle:
            result = ctypes.windll.kernel32.TerminateProcess(handle, 1)
            ctypes.windll.kernel32.CloseHandle(handle)
            if result:
                logger.info(f"Successfully terminated zombie pid={pid}")
                return True
            else:
                logger.error(f"TerminateProcess failed for pid={pid}")
                return False
        else:
            # Can't open — might already be dead
            return not _is_pid_alive(pid)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
            logger.info(f"Sent SIGKILL to zombie pid={pid}")
            return True
        except ProcessLookupError:
            return True  # already dead
        except PermissionError:
            logger.error(f"Permission denied killing pid={pid}")
            return False


def _wait_for_pid_death(pid: int, timeout: float = 5.0) -> bool:
    """Wait for a process to die after being killed. Returns True if dead."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _is_pid_alive(pid):
            return True
        time.sleep(0.2)
    return not _is_pid_alive(pid)


class LeaderInstance:
    """Manages the leader role: holds embeddings in RAM, serves HTTP API, runs indexer."""

    def __init__(self):
        self._lock: FileLock | None = None
        self._http_server: HTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._is_leader = False
        self._port = get_config().server.leader_port
        self._search_handler = None  # Set by server.py

    @property
    def is_leader(self) -> bool:
        return self._is_leader

    @property
    def port(self) -> int:
        return self._port

    def try_acquire_leadership(self) -> bool:
        """Attempt to become the leader. Non-blocking.

        Returns True if this process is now the leader.
        """
        lock_path = _get_lock_path()
        self._lock = FileLock(lock_path, timeout=0)

        try:
            self._lock.acquire(timeout=0)
            self._is_leader = True
            _write_leader_info(self._port, os.getpid())
            logger.info(f"Acquired leadership (pid={os.getpid()}, port={self._port})")
            return True
        except Timeout:
            self._is_leader = False
            return False

    def start_http_server(self, search_handler) -> bool:
        """Start the localhost HTTP API server in a background thread.

        Args:
            search_handler: Callable that handles search requests.
                           Signature: search_handler(request_dict) -> response_dict

        Returns:
            True if the HTTP server started and is responding to /health.
        """
        self._search_handler = search_handler
        leader_instance = self

        class RequestHandler(BaseHTTPRequestHandler):
            """HTTP handler for leader API."""

            def log_message(self, format, *args):
                """Suppress default HTTP logging."""
                pass

            def _send_json(self, data: dict, status: int = 200):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def _send_response_maybe_encrypted(self, data: dict, status: int = 200):
                """Send response, encrypting if peer secret is configured."""
                try:
                    from .peers import encrypt_response_body
                    body_bytes, content_type = encrypt_response_body(data)
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.end_headers()
                    self.wfile.write(body_bytes)
                except ImportError:
                    self._send_json(data, status)

            def _read_body(self) -> dict:
                length = int(self.headers.get("Content-Length", 0))
                if length == 0:
                    return {}
                body = self.rfile.read(length)
                return json.loads(body)

            def do_POST(self):
                try:
                    content_type = self.headers.get("Content-Type", "application/json")
                    length = int(self.headers.get("Content-Length", 0))
                    raw_body = self.rfile.read(length) if length > 0 else b"{}"

                    # Decrypt if encrypted
                    try:
                        from .peers import decrypt_request_body, encrypt_response_body
                        body = decrypt_request_body(raw_body, content_type)
                    except PermissionError as e:
                        self.send_response(401)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": str(e)}).encode())
                        return
                    except ImportError:
                        # peers module not available, parse as JSON
                        body = json.loads(raw_body) if raw_body else {}

                    if self.path == "/search":
                        result = leader_instance._search_handler(body)
                        self._send_response_maybe_encrypted(result)

                    elif self.path == "/reindex":
                        from .background_indexer import get_background_indexer
                        indexer = get_background_indexer()
                        indexer.trigger_reindex()
                        self._send_json({"status": "reindex_triggered"})

                    elif self.path == "/rescan":
                        from .background_indexer import get_background_indexer
                        indexer = get_background_indexer()
                        indexer.trigger_rescan()
                        self._send_json({"status": "rescan_triggered"})

                    elif self.path == "/reload-config":
                        from .config import reload_config as _reload, diff_configs
                        old_config, new_config = _reload()
                        changes = diff_configs(old_config, new_config)
                        safe_changes = [c for c in changes if c["impact"] == "safe"]
                        breaking_changes = [c for c in changes if c["impact"] == "requires_reindex"]
                        self._send_json({
                            "status": "config_reloaded" if changes else "no_changes",
                            "changes": changes,
                            "applied": [c["key"] for c in safe_changes],
                            "deferred": [f"{c['key']} — requires rescan(full=True)" for c in breaking_changes],
                            "warnings": [
                                f"{c['key']} changed: {c['old']} → {c['new']}. Requires rescan(full=True)."
                                for c in breaking_changes
                            ],
                        })

                    else:
                        self._send_json({"error": "not found"}, 404)

                except Exception as e:
                    self._send_json({"error": str(e)}, 500)

            def do_GET(self):
                try:
                    if self.path == "/":
                        self._send_dashboard()

                    elif self.path == "/status":
                        from .background_indexer import get_background_indexer
                        from .search import get_search_index
                        from .migrations import get_schema_version
                        indexer = get_background_indexer()
                        status = indexer.status.to_dict()
                        # Add fields that the MCP tool also adds
                        search_index = get_search_index()
                        status["search_ready"] = search_index.message_count > 0
                        status["search_message_count"] = search_index.message_count
                        if indexer.cache:
                            try:
                                db_path = indexer.cache.db_path
                                from pathlib import Path
                                db_path = Path(db_path) if not isinstance(db_path, Path) else db_path
                                status["db_size_mb"] = round(db_path.stat().st_size / (1024 * 1024), 2) if db_path.exists() else 0
                            except (OSError, TypeError):
                                status["db_size_mb"] = None
                            status["schema_version"] = get_schema_version(indexer.cache.conn)
                            try:
                                indexer.cache.conn.execute("SELECT 1 FROM messages_fts LIMIT 0")
                                status["fts_enabled"] = True
                            except Exception:
                                status["fts_enabled"] = False
                        else:
                            status["db_size_mb"] = None
                            status["schema_version"] = None
                            status["fts_enabled"] = False
                        status["uptime_seconds"] = round(time.time() - _process_start_time, 1)
                        # Process memory usage
                        try:
                            import resource
                            import platform as _platform
                            mem_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                            if _platform.system() == "Darwin":
                                mem_mb = mem_bytes / (1024 * 1024)
                            else:
                                mem_mb = mem_bytes / 1024
                        except ImportError:
                            # Windows: use psutil or fallback
                            try:
                                import psutil
                                mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                            except ImportError:
                                mem_mb = 0.0
                        status["memory_used_mb"] = round(mem_mb, 1)
                        from .memory import get_memory_limit
                        memory_limit = get_memory_limit()
                        if memory_limit > 0:
                            status["memory_limit_mb"] = round(memory_limit / (1024 * 1024))
                            status["memory_used_percent"] = round(mem_mb / (memory_limit / (1024 * 1024)) * 100, 1)
                        else:
                            status["memory_limit_mb"] = None
                            status["memory_used_percent"] = None
                        self._send_json(status)

                    elif self.path == "/config":
                        # Import here to avoid circular
                        from .server import get_config
                        self._send_json(get_config())

                    elif self.path == "/health":
                        self._send_json({"status": "ok", "role": "leader", "pid": os.getpid()})

                    elif self.path == "/role":
                        self._send_json({
                            "role": "leader",
                            "port": leader_instance._port,
                            "pid": os.getpid(),
                        })

                    else:
                        self._send_json({"error": "not found"}, 404)

                except Exception as e:
                    self._send_json({"error": str(e)}, 500)

            def _send_dashboard(self):
                """Serve a simple status dashboard HTML page."""
                html = _DASHBOARD_HTML
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))

        try:
            self._http_server = ThreadingHTTPServer(("127.0.0.1", self._port), RequestHandler)
            self._http_server.timeout = 1
            self._http_thread = threading.Thread(
                target=self._http_server.serve_forever,
                daemon=True,
                name="leader-http",
            )
            self._http_thread.start()
            logger.info(f"Leader HTTP server started on 127.0.0.1:{self._port}")
        except OSError as e:
            logger.error(f"Failed to start leader HTTP server on port {self._port}: {e}")
            # Try next port
            self._port += 1
            try:
                self._http_server = ThreadingHTTPServer(("127.0.0.1", self._port), RequestHandler)
                self._http_server.timeout = 1
                self._http_thread = threading.Thread(
                    target=self._http_server.serve_forever,
                    daemon=True,
                    name="leader-http",
                )
                self._http_thread.start()
                _write_leader_info(self._port, os.getpid())
                logger.info(f"Leader HTTP server started on 127.0.0.1:{self._port} (fallback port)")
            except OSError as e2:
                logger.error(f"Failed to start leader HTTP on fallback port {self._port}: {e2}")

        # Verify the server is actually responding.
        # Use a simple socket connect first (faster than HTTP) to detect when
        # the thread has bound and is accepting, then do the full /health check.
        started = False
        for _wait in range(20):  # up to ~2 seconds
            time.sleep(0.1)
            if _is_port_open(self._port):
                started = True
                break

        if not started:
            logger.error(
                f"Leader HTTP server not accepting connections on port "
                f"{self._port} after 2s — server failed to start"
            )
            self.release_leadership()
            return False

        # Port is open — now verify full /health response
        if not _is_leader_healthy(self._port, os.getpid()):
            logger.error(
                f"Leader HTTP server on port {self._port} accepts connections "
                f"but /health check failed — handler may be broken"
            )
            self.release_leadership()
            return False

        return True

    def release_leadership(self):
        """Release the leader lock and stop HTTP server."""
        if self._http_server:
            self._http_server.shutdown()
            self._http_server = None
        if self._lock:
            try:
                self._lock.release()
            except Exception:
                pass
        self._is_leader = False


class FollowerInstance:
    """Manages the follower role: forwards requests to leader via HTTP."""

    def __init__(self):
        self._leader_port: int | None = None
        self._session = None

    @property
    def leader_port(self) -> int | None:
        if self._leader_port is None:
            self._refresh_leader_port()
        return self._leader_port

    def _refresh_leader_port(self):
        """Re-read leader port from disk (handles leader restarts on new port)."""
        info = _read_leader_info()
        if info:
            self._leader_port = info.get("port")
        else:
            self._leader_port = None

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers["Content-Type"] = "application/json"
        return self._session

    def _base_url(self) -> str:
        port = self.leader_port
        if port is None:
            raise ConnectionError("No leader info available")
        return f"http://127.0.0.1:{port}"

    def is_leader_alive(self) -> bool:
        """Check if the leader is responding AND its PID is alive."""
        info = _read_leader_info()
        if not info:
            return False
        port = info.get("port")
        pid = info.get("pid")
        if port is None or pid is None:
            return False
        return _is_leader_healthy(port, pid)

    def search(self, request: dict) -> dict:
        """Forward a search request to the leader."""
        session = self._get_session()
        try:
            resp = session.post(f"{self._base_url()}/search", json=request, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (ConnectionError, OSError):
            # Leader may have restarted on a different port — refresh and retry
            self._refresh_leader_port()
            resp = session.post(f"{self._base_url()}/search", json=request, timeout=30)
            resp.raise_for_status()
            return resp.json()

    def get_status(self) -> dict:
        """Get indexing status from leader."""
        session = self._get_session()
        resp = session.get(f"{self._base_url()}/status", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def get_config(self) -> dict:
        """Get config from leader."""
        session = self._get_session()
        resp = session.get(f"{self._base_url()}/config", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def trigger_reindex(self) -> dict:
        """Tell leader to reindex."""
        session = self._get_session()
        resp = session.post(f"{self._base_url()}/reindex", json={}, timeout=5)
        resp.raise_for_status()
        return resp.json()

    def reload_config(self) -> dict:
        """Tell leader to reload config."""
        session = self._get_session()
        resp = session.post(f"{self._base_url()}/reload-config", json={}, timeout=5)
        resp.raise_for_status()
        return resp.json()

    def trigger_rescan(self) -> dict:
        """Tell leader to rescan now."""
        session = self._get_session()
        resp = session.post(f"{self._base_url()}/rescan", json={}, timeout=5)
        resp.raise_for_status()
        return resp.json()


class InstanceManager:
    """Manages the lifecycle of this process as either leader or follower.

    Handles initial role assignment, automatic failover, and periodic
    heartbeat checks for leader liveness.

    Zombie handling: if the current leader holds the file lock but its HTTP
    server is dead, this manager will kill it by PID to release the lock,
    then attempt to take over leadership. If 3 consecutive leader spawns
    fail health checks, gives up and stays in follower mode permanently.
    """

    MAX_SPAWN_ATTEMPTS = 3
    SPAWN_DEBOUNCE_SECONDS = 2.0
    DEGRADED_RETRY_INTERVAL = 300  # 5 minutes before retrying from degraded state

    def __init__(self):
        self._leader: LeaderInstance | None = None
        self._follower: FollowerInstance | None = None
        self._role: str = "undecided"  # "leader" | "follower" | "undecided" | "degraded"
        self._lock = threading.Lock()
        self._search_handler = None  # Stored for use during promotion
        self._heartbeat_thread: threading.Thread | None = None
        self._spawn_failures: int = 0  # Consecutive failed leader spawns
        self._permanently_degraded: bool = False  # True = stop trying to become leader
        self._degraded_since: float | None = None  # When we entered degraded state

    @property
    def role(self) -> str:
        return self._role

    @property
    def is_leader(self) -> bool:
        return self._role == "leader"

    @property
    def leader_instance(self) -> LeaderInstance | None:
        return self._leader

    @property
    def follower_instance(self) -> FollowerInstance | None:
        return self._follower

    def _kill_zombie_leader(self) -> bool:
        """Kill the zombie leader process identified in leader.json.

        Returns True if the zombie was killed (or was already dead) and
        the lock file is now available for acquisition.
        """
        leader_info = _read_leader_info()
        if not leader_info:
            return True  # No leader info — nothing to kill

        pid = leader_info.get("pid")
        if not pid or pid == os.getpid():
            return True  # No PID or it's us

        if not _is_pid_alive(pid):
            logger.info(f"Zombie leader pid={pid} already dead")
            # Try to clean up the lock file
            try:
                _get_lock_path().unlink(missing_ok=True)
            except OSError:
                pass
            return True

        # PID is alive but HTTP is dead — it's a zombie. Kill it.
        killed = _kill_process(pid)
        if not killed:
            logger.error(f"Failed to kill zombie leader pid={pid}")
            return False

        # Wait for it to actually die so the OS releases the file lock
        if not _wait_for_pid_death(pid, timeout=5.0):
            logger.error(
                f"Zombie leader pid={pid} did not die within timeout "
                f"after kill signal"
            )
            return False

        # Now try to remove the stale lock file
        try:
            _get_lock_path().unlink(missing_ok=True)
        except OSError:
            pass  # May already be gone

        return True

    def _try_become_leader(self, search_handler) -> bool:
        """Single attempt to acquire leadership and start HTTP server.

        Returns True only if leadership is acquired AND the HTTP server
        passes health check. On failure, releases any acquired lock.
        """
        leader = LeaderInstance()
        if not leader.try_acquire_leadership():
            return False

        # Acquired the lock — now start the HTTP server and verify health
        server_ok = leader.start_http_server(search_handler)
        if not server_ok:
            logger.error(
                "Acquired leadership but HTTP server failed health check — "
                "releasing lock"
            )
            leader.release_leadership()
            return False

        # Success
        self._leader = leader
        self._follower = None
        self._role = "leader"
        self._spawn_failures = 0
        return True

    def initialize(self, search_handler) -> str:
        """Determine role and set up accordingly.

        Attempts to become leader. If an existing leader is healthy, becomes
        follower. If the existing leader is a zombie, kills it and takes over.
        If our own HTTP server fails to start 3 times, enters degraded follower
        mode and logs clearly that it cannot serve with the current config.

        Args:
            search_handler: The search function to serve if this becomes leader.

        Returns:
            "leader", "follower", or "degraded"
        """
        self._search_handler = search_handler

        with self._lock:
            # Fast path: try to grab leadership directly
            if self._try_become_leader(search_handler):
                return "leader"

            # Couldn't get the lock. Check if existing leader is healthy.
            leader_info = _read_leader_info()
            if leader_info:
                port = leader_info.get("port")
                pid = leader_info.get("pid")
                if port and pid and _is_leader_healthy(port, pid):
                    # Healthy leader exists — become follower
                    self._follower = FollowerInstance()
                    self._role = "follower"
                    return "follower"

            # Leader is unhealthy (zombie or dead). Kill it and retry.
            logger.warning(
                "Existing leader is unresponsive — killing zombie and "
                "attempting takeover"
            )

            for attempt in range(self.MAX_SPAWN_ATTEMPTS):
                if self._kill_zombie_leader():
                    time.sleep(0.5)  # Brief pause for OS to release handles

                    if self._try_become_leader(search_handler):
                        return "leader"

                # Check if another process took over while we waited
                leader_info = _read_leader_info()
                if leader_info:
                    port = leader_info.get("port")
                    pid = leader_info.get("pid")
                    if port and pid and _is_leader_healthy(port, pid):
                        logger.info(
                            "Another process became healthy leader during our "
                            "takeover attempt — becoming follower"
                        )
                        self._follower = FollowerInstance()
                        self._role = "follower"
                        return "follower"

                self._spawn_failures += 1
                logger.warning(
                    f"Leader spawn attempt {attempt + 1}/{self.MAX_SPAWN_ATTEMPTS} "
                    f"failed (total failures: {self._spawn_failures})"
                )

                if attempt < self.MAX_SPAWN_ATTEMPTS - 1:
                    time.sleep(self.SPAWN_DEBOUNCE_SECONDS)

            # All attempts exhausted
            self._permanently_degraded = True
            self._degraded_since = time.time()
            self._role = "degraded"
            logger.error(
                "UNABLE TO START LEADER SERVICE: all %d spawn attempts failed. "
                "Falling back to degraded follower mode. The HTTP server cannot "
                "start with the current configuration (port=%s). Check for port "
                "conflicts or other processes holding resources.",
                self.MAX_SPAWN_ATTEMPTS,
                get_config().server.leader_port,
            )
            # Still set up as follower in case another instance manages to lead
            self._follower = FollowerInstance()
            return "degraded"

    def promote_to_leader(self, search_handler) -> bool:
        """Attempt to promote this follower to leader (failover).

        If the current leader is a zombie (alive but not serving HTTP),
        kills it first. If our own server fails to start after killing
        the zombie, counts toward the spawn failure limit.

        Returns True if promotion succeeded.
        """
        if self._permanently_degraded:
            logger.debug(
                "Skipping promotion attempt — permanently degraded"
            )
            return False

        with self._lock:
            # Try direct acquisition (leader might have exited cleanly and
            # the OS released the file lock). Do NOT unlink the lock file —
            # that creates a TOCTOU race with other followers also promoting.
            # FileLock handles stale files correctly on its own.
            if self._try_become_leader(search_handler):
                logger.info("Promoted to leader after failover")
                return True

            # Direct acquisition failed — zombie is holding the lock.
            # Kill it.
            if not self._kill_zombie_leader():
                self._spawn_failures += 1
                if self._spawn_failures >= self.MAX_SPAWN_ATTEMPTS:
                    self._permanently_degraded = True
                    self._degraded_since = time.time()
                    self._role = "degraded"
                    logger.error(
                        "PROMOTION PERMANENTLY FAILED: unable to kill zombie "
                        "leader after %d attempts. Entering degraded mode. "
                        "Manual intervention required.",
                        self._spawn_failures,
                    )
                return False

            # Zombie killed — wait briefly for OS cleanup, then try again
            time.sleep(0.5)

            if self._try_become_leader(search_handler):
                logger.info(
                    "Promoted to leader after killing zombie (pid from leader.json)"
                )
                return True

            # Even after killing zombie, we couldn't start. Count it.
            self._spawn_failures += 1
            if self._spawn_failures >= self.MAX_SPAWN_ATTEMPTS:
                self._permanently_degraded = True
                self._degraded_since = time.time()
                self._role = "degraded"
                logger.error(
                    "PROMOTION PERMANENTLY FAILED: acquired lock but HTTP server "
                    "will not start after %d consecutive failures. Entering "
                    "degraded mode. Check port %s and current config.",
                    self._spawn_failures,
                    get_config().server.leader_port,
                )
            else:
                logger.warning(
                    f"Promotion failed (spawn failure {self._spawn_failures}/"
                    f"{self.MAX_SPAWN_ATTEMPTS}) — will retry on next heartbeat"
                )
            return False

    def get_role_info(self) -> dict:
        """Get role information for the MCP tool."""
        info = {
            "role": self._role,
            "pid": os.getpid(),
            "degraded": self._permanently_degraded,
            "spawn_failures": self._spawn_failures,
        }
        if self._role == "leader" and self._leader:
            info["port"] = self._leader.port
        elif self._role in ("follower", "degraded"):
            leader_info = _read_leader_info()
            if leader_info:
                info["leader_port"] = leader_info.get("port")
                info["leader_pid"] = leader_info.get("pid")
        return info

    def start_heartbeat(self):
        """Start the background heartbeat thread.

        - Leader: periodically updates leader.json with a fresh timestamp.
        - Follower: periodically checks if the leader is alive. If dead,
          attempts promotion to leader.
        - Degraded: does nothing (gave up).
        """
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="heartbeat"
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self):
        """Periodic heartbeat: leader writes timestamp, follower checks liveness.

        Adds ±25% jitter to prevent synchronized detection across multiple
        followers (thundering herd on leader death).
        """
        config = get_config()
        base_interval = config.server.heartbeat_interval_seconds

        while True:
            # Jitter prevents all followers from detecting death simultaneously
            jitter = base_interval * 0.25 * (2 * random.random() - 1)
            time.sleep(base_interval + jitter)
            try:
                if self._permanently_degraded:
                    # Retry from degraded state after the retry interval expires
                    if (
                        self._degraded_since
                        and (time.time() - self._degraded_since)
                        > self.DEGRADED_RETRY_INTERVAL
                    ):
                        logger.info(
                            "Retrying leader election from degraded state "
                            f"(degraded for {int(time.time() - self._degraded_since)}s)"
                        )
                        self._spawn_failures = 0
                        self._permanently_degraded = False
                        self._degraded_since = None
                        if self.promote_to_leader(self._search_handler):
                            from .background_indexer import get_background_indexer
                            indexer = get_background_indexer()
                            indexer.start()
                            from .search import get_search_index
                            get_search_index()
                        else:
                            logger.info(
                                "Degraded retry failed; will try again in "
                                f"{self.DEGRADED_RETRY_INTERVAL}s"
                            )
                    continue

                if self.is_leader:
                    # Leader: refresh the timestamp in leader.json
                    if self._leader:
                        _write_leader_info(self._leader.port, os.getpid())
                else:
                    # Follower: check if leader is still alive
                    if self._follower and not self._follower.is_leader_alive():
                        # Before attempting promotion, check if another follower
                        # already won the election (avoids counting a lost race
                        # as a spawn failure).
                        leader_info = _read_leader_info()
                        if leader_info:
                            lport = leader_info.get("port")
                            lpid = leader_info.get("pid")
                            if lport and lpid and _is_leader_healthy(lport, lpid):
                                logger.info(
                                    "Heartbeat: another process became leader "
                                    "— staying as follower"
                                )
                                self._follower._refresh_leader_port()
                                continue

                        logger.info("Heartbeat: leader appears dead, attempting promotion")
                        if self.promote_to_leader(self._search_handler):
                            # Successfully promoted — start the indexer
                            from .background_indexer import get_background_indexer
                            indexer = get_background_indexer()
                            indexer.start()
                            from .search import get_search_index
                            get_search_index()
                        else:
                            logger.info(
                                "Heartbeat: promotion failed; will retry on "
                                "next heartbeat cycle"
                            )
            except Exception as e:
                logger.debug(f"Heartbeat error: {e}")


# Global singleton
_instance_manager: InstanceManager | None = None


def get_instance_manager() -> InstanceManager:
    """Get or create the global instance manager."""
    global _instance_manager
    if _instance_manager is None:
        _instance_manager = InstanceManager()
    return _instance_manager
