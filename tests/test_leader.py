"""System tests for engine-follower coordination.

Tests InstanceManager initialization, role assignment, promotion,
and FollowerInstance HTTP forwarding behavior.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kiro_ception.coordination import (
    FollowerInstance,
    InstanceManager,
    EngineInstance,
    _read_engine_info,
    _write_engine_info,
)


# --- Engine info file operations ---


class TestEngineInfoFile:
    def test_write_and_read(self, tmp_path, monkeypatch):
        info_path = tmp_path / "engine.json"
        monkeypatch.setattr("kiro_ception.coordination._get_engine_info_path", lambda: info_path)

        _write_engine_info(port=19742, pid=12345)
        info = _read_engine_info()

        assert info is not None
        assert info["port"] == 19742
        assert info["pid"] == 12345

    def test_read_nonexistent_returns_none(self, tmp_path, monkeypatch):
        info_path = tmp_path / "nonexistent.json"
        monkeypatch.setattr("kiro_ception.coordination._get_engine_info_path", lambda: info_path)

        assert _read_engine_info() is None

    def test_read_corrupted_returns_none(self, tmp_path, monkeypatch):
        info_path = tmp_path / "engine.json"
        info_path.write_text("not valid json{{{")
        monkeypatch.setattr("kiro_ception.coordination._get_engine_info_path", lambda: info_path)

        assert _read_engine_info() is None


# --- EngineInstance ---


class TestEngineInstance:
    def test_initial_state(self):
        engine = EngineInstance()
        assert engine.is_engine is False

    def test_acquire_engineship(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "engine.lock"
        info_path = tmp_path / "engine.json"
        monkeypatch.setattr("kiro_ception.coordination._get_lock_path", lambda: lock_path)
        monkeypatch.setattr("kiro_ception.coordination._get_engine_info_path", lambda: info_path)

        engine = EngineInstance()
        acquired = engine.try_acquire_engineship()

        assert acquired is True
        assert engine.is_engine is True
        assert engine.port > 0

        # Clean up
        engine.release_engineship()

    def test_release_engineship(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "engine.lock"
        info_path = tmp_path / "engine.json"
        monkeypatch.setattr("kiro_ception.coordination._get_lock_path", lambda: lock_path)
        monkeypatch.setattr("kiro_ception.coordination._get_engine_info_path", lambda: info_path)

        engine = EngineInstance()
        engine.try_acquire_engineship()
        engine.release_engineship()

        assert engine.is_engine is False


# --- FollowerInstance ---


class TestFollowerInstance:
    def test_engine_port_from_info_file(self, tmp_path, monkeypatch):
        info_path = tmp_path / "engine.json"
        info_path.write_text(json.dumps({"port": 19742, "pid": 99}))
        monkeypatch.setattr("kiro_ception.coordination._get_engine_info_path", lambda: info_path)

        follower = FollowerInstance()
        assert follower.engine_port == 19742

    def test_engine_port_none_when_no_info(self, tmp_path, monkeypatch):
        info_path = tmp_path / "nonexistent.json"
        monkeypatch.setattr("kiro_ception.coordination._get_engine_info_path", lambda: info_path)

        follower = FollowerInstance()
        assert follower.engine_port is None

    def test_search_forwards_to_engine(self, tmp_path, monkeypatch):
        info_path = tmp_path / "engine.json"
        info_path.write_text(json.dumps({"port": 19742, "pid": 99}))
        monkeypatch.setattr("kiro_ception.coordination._get_engine_info_path", lambda: info_path)

        follower = FollowerInstance()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [], "total_matches": 0}

        with patch.object(follower, "_get_session") as mock_session:
            mock_session.return_value.post.return_value = mock_response

            result = follower.search({"query": "test", "max_results": 10})

            assert result == {"results": [], "total_matches": 0}
            mock_session.return_value.post.assert_called_once()

    def test_get_status_forwards_to_engine(self, tmp_path, monkeypatch):
        info_path = tmp_path / "engine.json"
        info_path.write_text(json.dumps({"port": 19742, "pid": 99}))
        monkeypatch.setattr("kiro_ception.coordination._get_engine_info_path", lambda: info_path)

        follower = FollowerInstance()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"state": "idle", "sessions_total": 100}

        with patch.object(follower, "_get_session") as mock_session:
            mock_session.return_value.get.return_value = mock_response

            result = follower.get_status()

            assert result["state"] == "idle"

    def test_trigger_reindex_forwards_to_engine(self, tmp_path, monkeypatch):
        info_path = tmp_path / "engine.json"
        info_path.write_text(json.dumps({"port": 19742, "pid": 99}))
        monkeypatch.setattr("kiro_ception.coordination._get_engine_info_path", lambda: info_path)

        follower = FollowerInstance()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "force_reindex_triggered"}

        with patch.object(follower, "_get_session") as mock_session:
            mock_session.return_value.post.return_value = mock_response

            result = follower.trigger_reindex()

            assert result["status"] == "force_reindex_triggered"


# --- InstanceManager ---


class TestInstanceManager:
    def test_initial_state(self):
        manager = InstanceManager()
        assert manager.role == "undecided"
        assert manager.is_engine is False

    def test_becomes_engine_when_no_existing(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "engine.lock"
        info_path = tmp_path / "engine.json"
        monkeypatch.setattr("kiro_ception.coordination._get_lock_path", lambda: lock_path)
        monkeypatch.setattr("kiro_ception.coordination._get_engine_info_path", lambda: info_path)

        manager = InstanceManager()
        search_handler = MagicMock()
        role = manager.initialize(search_handler=search_handler)

        assert role == "engine"
        assert manager.is_engine is True
        assert manager.engine_instance is not None

        # Clean up
        manager.engine_instance.release_engineship()

    def test_becomes_follower_when_engine_exists(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "engine.lock"
        info_path = tmp_path / "engine.json"
        monkeypatch.setattr("kiro_ception.coordination._get_lock_path", lambda: lock_path)
        monkeypatch.setattr("kiro_ception.coordination._get_engine_info_path", lambda: info_path)

        # First instance becomes engine
        engine_manager = InstanceManager()
        engine_manager.initialize(search_handler=MagicMock())
        assert engine_manager.is_engine is True

        # Second instance — lock is held, engine is healthy, should become follower
        with patch("kiro_ception.coordination._is_engine_healthy", return_value=True):
            follower_manager = InstanceManager()
            role = follower_manager.initialize(search_handler=MagicMock())

        assert role == "follower"
        assert follower_manager.is_engine is False
        assert follower_manager.follower_instance is not None

        # Clean up
        engine_manager.engine_instance.release_engineship()

    def test_get_role_info_as_engine(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "engine.lock"
        info_path = tmp_path / "engine.json"
        monkeypatch.setattr("kiro_ception.coordination._get_lock_path", lambda: lock_path)
        monkeypatch.setattr("kiro_ception.coordination._get_engine_info_path", lambda: info_path)

        manager = InstanceManager()
        manager.initialize(search_handler=MagicMock())

        info = manager.get_role_info()
        assert info["role"] == "engine"
        assert "pid" in info
        assert "port" in info

        manager.engine_instance.release_engineship()

    def test_promote_to_engine(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "engine.lock"
        info_path = tmp_path / "engine.json"
        monkeypatch.setattr("kiro_ception.coordination._get_lock_path", lambda: lock_path)
        monkeypatch.setattr("kiro_ception.coordination._get_engine_info_path", lambda: info_path)

        # Start as engine, then release so promotion is possible
        first = InstanceManager()
        first.initialize(search_handler=MagicMock())
        first.engine_instance.release_engineship()

        # New instance tries to promote
        manager = InstanceManager()
        # Simulate being a follower whose engine died
        manager._role = "follower"
        manager._follower = FollowerInstance()

        result = manager.promote_to_engine(search_handler=MagicMock())
        assert result is True
        assert manager.is_engine is True

        manager.engine_instance.release_engineship()
