"""Leader-follower pattern for multi-instance coordination.

Only one process (the leader) holds embeddings in RAM and runs the indexer.
All other processes (followers) forward requests to the leader via localhost HTTP.
Automatic failover: if the leader dies, a follower promotes itself.
"""

import json
import logging
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from filelock import FileLock, Timeout

from .config import get_config, expand_path

logger = logging.getLogger(__name__)

# Lock file location
_LOCK_FILE_NAME = "leader.lock"
_LEADER_INFO_NAME = "leader.json"


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
    """Write leader info to disk so followers can find it."""
    info_path = _get_leader_info_path()
    info = {"port": port, "pid": pid, "started_at": time.time()}
    with open(info_path, "w") as f:
        json.dump(info, f)


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

    def start_http_server(self, search_handler):
        """Start the localhost HTTP API server in a background thread.

        Args:
            search_handler: Callable that handles search requests.
                           Signature: search_handler(request_dict) -> response_dict
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

            def _read_body(self) -> dict:
                length = int(self.headers.get("Content-Length", 0))
                if length == 0:
                    return {}
                body = self.rfile.read(length)
                return json.loads(body)

            def do_POST(self):
                try:
                    body = self._read_body()

                    if self.path == "/search":
                        result = leader_instance._search_handler(body)
                        self._send_json(result)

                    elif self.path == "/reindex":
                        from .background_indexer import get_background_indexer
                        indexer = get_background_indexer()
                        indexer.trigger_reindex()
                        self._send_json({"status": "reindex_triggered"})

                    else:
                        self._send_json({"error": "not found"}, 404)

                except Exception as e:
                    self._send_json({"error": str(e)}, 500)

            def do_GET(self):
                try:
                    if self.path == "/status":
                        from .background_indexer import get_background_indexer
                        indexer = get_background_indexer()
                        self._send_json(indexer.status.to_dict())

                    elif self.path == "/config":
                        # Import here to avoid circular
                        from .server import get_recall_config
                        self._send_json(get_recall_config())

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

        try:
            self._http_server = HTTPServer(("127.0.0.1", self._port), RequestHandler)
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
                self._http_server = HTTPServer(("127.0.0.1", self._port), RequestHandler)
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
            info = _read_leader_info()
            if info:
                self._leader_port = info.get("port")
        return self._leader_port

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
        """Check if the leader is responding."""
        port = self.leader_port
        if port is None:
            return False
        return _is_port_open(port)

    def search(self, request: dict) -> dict:
        """Forward a search request to the leader."""
        session = self._get_session()
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

    def get_role(self) -> dict:
        """Get leader's role info."""
        session = self._get_session()
        resp = session.get(f"{self._base_url()}/role", timeout=5)
        resp.raise_for_status()
        return resp.json()


class InstanceManager:
    """Manages the lifecycle of this process as either leader or follower.

    Handles initial role assignment and automatic failover.
    """

    def __init__(self):
        self._leader: LeaderInstance | None = None
        self._follower: FollowerInstance | None = None
        self._role: str = "undecided"  # "leader" | "follower" | "undecided"
        self._lock = threading.Lock()

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

    def initialize(self, search_handler) -> str:
        """Determine role and set up accordingly.

        Args:
            search_handler: The search function to serve if this becomes leader.

        Returns:
            "leader" or "follower"
        """
        with self._lock:
            leader = LeaderInstance()
            if leader.try_acquire_leadership():
                self._leader = leader
                self._role = "leader"
                leader.start_http_server(search_handler)
                return "leader"
            else:
                # Check if the existing leader is actually alive
                follower = FollowerInstance()
                if follower.is_leader_alive():
                    self._follower = follower
                    self._role = "follower"
                    return "follower"
                else:
                    # Leader is dead, force acquire
                    logger.warning("Stored leader is unresponsive, forcing lock acquisition")
                    lock_path = _get_lock_path()
                    # Remove stale lock and retry
                    try:
                        lock_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    leader2 = LeaderInstance()
                    if leader2.try_acquire_leadership():
                        self._leader = leader2
                        self._role = "leader"
                        leader2.start_http_server(search_handler)
                        return "leader"
                    else:
                        # Another process beat us to it
                        self._follower = FollowerInstance()
                        self._role = "follower"
                        return "follower"

    def promote_to_leader(self, search_handler) -> bool:
        """Attempt to promote this follower to leader (failover).

        Returns True if promotion succeeded.
        """
        with self._lock:
            leader = LeaderInstance()
            # Try to remove stale lock first
            lock_path = _get_lock_path()
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass

            if leader.try_acquire_leadership():
                self._leader = leader
                self._follower = None
                self._role = "leader"
                leader.start_http_server(search_handler)
                logger.info("Promoted to leader after failover")
                return True
            return False

    def get_role_info(self) -> dict:
        """Get role information for the MCP tool."""
        info = {
            "role": self._role,
            "pid": os.getpid(),
        }
        if self._role == "leader" and self._leader:
            info["port"] = self._leader.port
        elif self._role == "follower":
            leader_info = _read_leader_info()
            if leader_info:
                info["leader_port"] = leader_info.get("port")
                info["leader_pid"] = leader_info.get("pid")
        return info


# Global singleton
_instance_manager: InstanceManager | None = None


def get_instance_manager() -> InstanceManager:
    """Get or create the global instance manager."""
    global _instance_manager
    if _instance_manager is None:
        _instance_manager = InstanceManager()
    return _instance_manager
