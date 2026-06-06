"""System tests for the leader HTTP server and follower HTTP forwarding.

Starts a real HTTP server on a random port and tests each endpoint.
"""

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from kiro_ception.coordination import LeaderInstance


@pytest.fixture
def leader_server(tmp_path, monkeypatch):
    """Start a real leader HTTP server on a free port."""
    lock_path = tmp_path / "leader.lock"
    info_path = tmp_path / "leader.json"
    monkeypatch.setattr("kiro_ception.coordination._get_lock_path", lambda: lock_path)
    monkeypatch.setattr("kiro_ception.coordination._get_leader_info_path", lambda: info_path)

    # Use a random high port to avoid conflicts
    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    from kiro_ception.config import Config, ServerConfig
    config = Config(server=ServerConfig(leader_port=port))
    monkeypatch.setattr("kiro_ception.coordination.get_config", lambda: config)

    leader = LeaderInstance()
    leader._port = port
    leader._is_leader = True

    search_handler = MagicMock()
    search_handler.return_value = {
        "results": [{"score": 0.95, "content": "test result"}],
        "query": "test",
        "total_matches": 1,
    }

    leader.start_http_server(search_handler)
    # Give server a moment to start
    time.sleep(0.1)

    yield leader, port, search_handler

    leader._http_server.shutdown()


class TestLeaderHTTPSearch:
    def test_search_endpoint(self, leader_server):
        leader, port, search_handler = leader_server

        resp = requests.post(
            f"http://127.0.0.1:{port}/search",
            json={"query": "test query", "max_results": 5},
            timeout=5,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_matches"] == 1
        assert data["results"][0]["score"] == 0.95
        # Verify handler was called with the request body
        search_handler.assert_called_once_with({"query": "test query", "max_results": 5})


class TestLeaderHTTPStatus:
    def test_status_endpoint(self, leader_server):
        leader, port, _ = leader_server

        # Mock the background indexer's status
        mock_status = MagicMock()
        mock_status.to_dict.return_value = {
            "state": "idle",
            "sessions_total": 100,
            "embedding_count": 5000,
        }

        mock_indexer = MagicMock()
        mock_indexer.status = mock_status

        with patch("kiro_ception.background_indexer.get_background_indexer", return_value=mock_indexer):
            resp = requests.get(f"http://127.0.0.1:{port}/status", timeout=5)

        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "idle"
        assert data["sessions_total"] == 100


class TestLeaderHTTPHealth:
    def test_health_endpoint(self, leader_server):
        leader, port, _ = leader_server

        resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["role"] == "leader"
        assert data["pid"] == os.getpid()


class TestLeaderHTTPRole:
    def test_role_endpoint(self, leader_server):
        leader, port, _ = leader_server

        resp = requests.get(f"http://127.0.0.1:{port}/role", timeout=5)

        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "leader"
        assert data["port"] == port


class TestLeaderHTTPReindex:
    def test_reindex_endpoint(self, leader_server):
        leader, port, _ = leader_server

        mock_indexer = MagicMock()
        with patch("kiro_ception.background_indexer.get_background_indexer", return_value=mock_indexer):
            resp = requests.post(f"http://127.0.0.1:{port}/reindex", json={}, timeout=5)

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "reindex_triggered"
        mock_indexer.trigger_reindex.assert_called_once()


class TestLeaderHTTPRescan:
    def test_rescan_endpoint(self, leader_server):
        leader, port, _ = leader_server

        mock_indexer = MagicMock()
        with patch("kiro_ception.background_indexer.get_background_indexer", return_value=mock_indexer):
            resp = requests.post(f"http://127.0.0.1:{port}/rescan", json={}, timeout=5)

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "rescan_triggered"
        mock_indexer.trigger_rescan.assert_called_once()


class TestLeaderHTTPReloadConfig:
    def test_reload_config_no_changes(self, leader_server):
        leader, port, _ = leader_server

        from kiro_ception.config import Config
        with patch("kiro_ception.config.reload_config", return_value=(Config(), Config())):
            with patch("kiro_ception.config.diff_configs", return_value=[]):
                resp = requests.post(f"http://127.0.0.1:{port}/reload-config", json={}, timeout=5)

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "no_changes"

    def test_reload_config_with_changes(self, leader_server):
        leader, port, _ = leader_server

        changes = [{"key": "indexing.throttle_ms", "old": 0, "new": 100, "impact": "safe"}]
        from kiro_ception.config import Config
        with patch("kiro_ception.config.reload_config", return_value=(Config(), Config())):
            with patch("kiro_ception.config.diff_configs", return_value=changes):
                resp = requests.post(f"http://127.0.0.1:{port}/reload-config", json={}, timeout=5)

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "config_reloaded"
        assert "indexing.throttle_ms" in data["applied"]


class TestLeaderHTTP404:
    def test_unknown_get_path(self, leader_server):
        leader, port, _ = leader_server

        resp = requests.get(f"http://127.0.0.1:{port}/nonexistent", timeout=5)
        assert resp.status_code == 404

    def test_unknown_post_path(self, leader_server):
        leader, port, _ = leader_server

        resp = requests.post(f"http://127.0.0.1:{port}/nonexistent", json={}, timeout=5)
        assert resp.status_code == 404
