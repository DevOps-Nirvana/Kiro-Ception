"""System tests for leader-follower coordination.

Tests InstanceManager initialization, role assignment, promotion,
and FollowerInstance HTTP forwarding behavior.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kiro_ception.leader import (
    FollowerInstance,
    InstanceManager,
    LeaderInstance,
    _read_leader_info,
    _write_leader_info,
)


# --- Leader info file operations ---


class TestLeaderInfoFile:
    def test_write_and_read(self, tmp_path, monkeypatch):
        info_path = tmp_path / "leader.json"
        monkeypatch.setattr("kiro_ception.leader._get_leader_info_path", lambda: info_path)

        _write_leader_info(port=19742, pid=12345)
        info = _read_leader_info()

        assert info is not None
        assert info["port"] == 19742
        assert info["pid"] == 12345

    def test_read_nonexistent_returns_none(self, tmp_path, monkeypatch):
        info_path = tmp_path / "nonexistent.json"
        monkeypatch.setattr("kiro_ception.leader._get_leader_info_path", lambda: info_path)

        assert _read_leader_info() is None

    def test_read_corrupted_returns_none(self, tmp_path, monkeypatch):
        info_path = tmp_path / "leader.json"
        info_path.write_text("not valid json{{{")
        monkeypatch.setattr("kiro_ception.leader._get_leader_info_path", lambda: info_path)

        assert _read_leader_info() is None


# --- LeaderInstance ---


class TestLeaderInstance:
    def test_initial_state(self):
        leader = LeaderInstance()
        assert leader.is_leader is False

    def test_acquire_leadership(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "leader.lock"
        info_path = tmp_path / "leader.json"
        monkeypatch.setattr("kiro_ception.leader._get_lock_path", lambda: lock_path)
        monkeypatch.setattr("kiro_ception.leader._get_leader_info_path", lambda: info_path)

        leader = LeaderInstance()
        acquired = leader.try_acquire_leadership()

        assert acquired is True
        assert leader.is_leader is True
        assert leader.port > 0

        # Clean up
        leader.release_leadership()

    def test_release_leadership(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "leader.lock"
        info_path = tmp_path / "leader.json"
        monkeypatch.setattr("kiro_ception.leader._get_lock_path", lambda: lock_path)
        monkeypatch.setattr("kiro_ception.leader._get_leader_info_path", lambda: info_path)

        leader = LeaderInstance()
        leader.try_acquire_leadership()
        leader.release_leadership()

        assert leader.is_leader is False


# --- FollowerInstance ---


class TestFollowerInstance:
    def test_leader_port_from_info_file(self, tmp_path, monkeypatch):
        info_path = tmp_path / "leader.json"
        info_path.write_text(json.dumps({"port": 19742, "pid": 99}))
        monkeypatch.setattr("kiro_ception.leader._get_leader_info_path", lambda: info_path)

        follower = FollowerInstance()
        assert follower.leader_port == 19742

    def test_leader_port_none_when_no_info(self, tmp_path, monkeypatch):
        info_path = tmp_path / "nonexistent.json"
        monkeypatch.setattr("kiro_ception.leader._get_leader_info_path", lambda: info_path)

        follower = FollowerInstance()
        assert follower.leader_port is None

    def test_search_forwards_to_leader(self, tmp_path, monkeypatch):
        info_path = tmp_path / "leader.json"
        info_path.write_text(json.dumps({"port": 19742, "pid": 99}))
        monkeypatch.setattr("kiro_ception.leader._get_leader_info_path", lambda: info_path)

        follower = FollowerInstance()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [], "total_matches": 0}

        with patch.object(follower, "_get_session") as mock_session:
            mock_session.return_value.post.return_value = mock_response

            result = follower.search({"query": "test", "max_results": 10})

            assert result == {"results": [], "total_matches": 0}
            mock_session.return_value.post.assert_called_once()

    def test_get_status_forwards_to_leader(self, tmp_path, monkeypatch):
        info_path = tmp_path / "leader.json"
        info_path.write_text(json.dumps({"port": 19742, "pid": 99}))
        monkeypatch.setattr("kiro_ception.leader._get_leader_info_path", lambda: info_path)

        follower = FollowerInstance()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"state": "idle", "sessions_total": 100}

        with patch.object(follower, "_get_session") as mock_session:
            mock_session.return_value.get.return_value = mock_response

            result = follower.get_status()

            assert result["state"] == "idle"

    def test_trigger_reindex_forwards_to_leader(self, tmp_path, monkeypatch):
        info_path = tmp_path / "leader.json"
        info_path.write_text(json.dumps({"port": 19742, "pid": 99}))
        monkeypatch.setattr("kiro_ception.leader._get_leader_info_path", lambda: info_path)

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
        assert manager.is_leader is False

    def test_becomes_leader_when_no_existing(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "leader.lock"
        info_path = tmp_path / "leader.json"
        monkeypatch.setattr("kiro_ception.leader._get_lock_path", lambda: lock_path)
        monkeypatch.setattr("kiro_ception.leader._get_leader_info_path", lambda: info_path)

        manager = InstanceManager()
        search_handler = MagicMock()
        role = manager.initialize(search_handler=search_handler)

        assert role == "leader"
        assert manager.is_leader is True
        assert manager.leader_instance is not None

        # Clean up
        manager.leader_instance.release_leadership()

    def test_becomes_follower_when_leader_exists(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "leader.lock"
        info_path = tmp_path / "leader.json"
        monkeypatch.setattr("kiro_ception.leader._get_lock_path", lambda: lock_path)
        monkeypatch.setattr("kiro_ception.leader._get_leader_info_path", lambda: info_path)

        # First instance becomes leader
        leader_manager = InstanceManager()
        leader_manager.initialize(search_handler=MagicMock())
        assert leader_manager.is_leader is True

        # Second instance — lock is held, port is open, should become follower
        with patch("kiro_ception.leader._is_port_open", return_value=True):
            follower_manager = InstanceManager()
            role = follower_manager.initialize(search_handler=MagicMock())

        assert role == "follower"
        assert follower_manager.is_leader is False
        assert follower_manager.follower_instance is not None

        # Clean up
        leader_manager.leader_instance.release_leadership()

    def test_get_role_info_as_leader(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "leader.lock"
        info_path = tmp_path / "leader.json"
        monkeypatch.setattr("kiro_ception.leader._get_lock_path", lambda: lock_path)
        monkeypatch.setattr("kiro_ception.leader._get_leader_info_path", lambda: info_path)

        manager = InstanceManager()
        manager.initialize(search_handler=MagicMock())

        info = manager.get_role_info()
        assert info["role"] == "leader"
        assert "pid" in info
        assert "port" in info

        manager.leader_instance.release_leadership()

    def test_promote_to_leader(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "leader.lock"
        info_path = tmp_path / "leader.json"
        monkeypatch.setattr("kiro_ception.leader._get_lock_path", lambda: lock_path)
        monkeypatch.setattr("kiro_ception.leader._get_leader_info_path", lambda: info_path)

        # Start as leader, then release so promotion is possible
        first = InstanceManager()
        first.initialize(search_handler=MagicMock())
        first.leader_instance.release_leadership()

        # New instance tries to promote
        manager = InstanceManager()
        # Simulate being a follower whose leader died
        manager._role = "follower"
        manager._follower = FollowerInstance()

        result = manager.promote_to_leader(search_handler=MagicMock())
        assert result is True
        assert manager.is_leader is True

        manager.leader_instance.release_leadership()
