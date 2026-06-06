"""Unit tests for MCP tool endpoint functions in server.py.

Each tool is tested with mocked dependencies to verify:
- Correct response structure
- Proper leader/follower routing
- Error handling / graceful degradation
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# --- Shared fixtures ---


@pytest.fixture
def mock_leader_setup():
    """Set up mocks for a leader instance with a functional search index."""
    manager = MagicMock()
    manager.is_leader = True

    indexer = MagicMock()
    indexer.backend = MagicMock()
    indexer.backend.fingerprint.return_value = "openai-compatible:localhost:qwen3:1024"
    indexer.cache = MagicMock()
    indexer.cache.db_path = "/tmp/cache.db"
    indexer.cache.embedding_count = 1000
    indexer.cache.message_count = 2000
    indexer.cache.indexed_session_count = 100

    # Mock status for get_indexing_status
    indexer.status = MagicMock()
    indexer.status.to_dict.return_value = {
        "state": "idle",
        "sessions_total": 100,
        "sessions_processed": 100,
        "embedding_count": 1000,
        "errors": 0,
    }

    return manager, indexer


@pytest.fixture
def mock_follower_setup():
    """Set up mocks for a follower instance."""
    manager = MagicMock()
    manager.is_leader = False
    manager.follower_instance = MagicMock()
    return manager


# --- search_project_history ---


class TestSearchProjectHistory:
    def test_returns_dict_structure(self, mock_leader_setup):
        manager, indexer = mock_leader_setup

        # Mock the embedding backend to return a query vector
        indexer.backend.encode_query.return_value = np.random.randn(1024).astype(np.float32)

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.search._search_index", None),
        ):
            from kiro_ception.server import search_project_history

            result = search_project_history(query="test query")

            assert isinstance(result, dict)
            assert "query" in result
            assert "results" in result or "hint" in result or "error" in result

    def test_passes_workspace_filter(self, mock_leader_setup):
        manager, indexer = mock_leader_setup
        indexer.backend.encode_query.return_value = np.random.randn(1024).astype(np.float32)

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.search._search_index", None),
            patch("kiro_ception.server._get_current_workspace", return_value="/my/project"),
            patch("kiro_ception.server.search") as mock_search,
        ):
            mock_search.return_value = {"results": [], "query": "test", "total_matches": 0}
            from kiro_ception.server import search_project_history

            search_project_history(query="test")

            # Verify workspace was passed to search()
            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["workspace"] == "/my/project"

    def test_workspace_dir_from_config(self, mock_leader_setup):
        """search.workspace_dir in config overrides auto-detection."""
        manager, indexer = mock_leader_setup

        from kiro_ception.config import Config, SearchConfig
        config = Config(search=SearchConfig(workspace_dir="~/my-project"))

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.server._get_config", return_value=config),
            patch("kiro_ception.server.search") as mock_search,
        ):
            mock_search.return_value = {"results": [], "query": "test", "total_matches": 0}
            from kiro_ception.server import search_project_history

            search_project_history(query="test")

            call_kwargs = mock_search.call_args[1]
            # Should be the expanded path from config
            assert "/my-project" in call_kwargs["workspace"]
            assert "~" not in call_kwargs["workspace"]

    def test_workspace_dir_empty_falls_back_to_env(self, mock_leader_setup, monkeypatch):
        """Empty workspace_dir falls back to KIRO_WORKSPACE env var."""
        manager, indexer = mock_leader_setup
        monkeypatch.setenv("KIRO_WORKSPACE", "/env/workspace")

        from kiro_ception.config import Config, SearchConfig
        config = Config(search=SearchConfig(workspace_dir=""))

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.server._get_config", return_value=config),
            patch("kiro_ception.server.search") as mock_search,
        ):
            mock_search.return_value = {"results": [], "query": "test", "total_matches": 0}
            from kiro_ception.server import search_project_history

            search_project_history(query="test")

            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["workspace"] == "/env/workspace"


# --- search_global_history ---


class TestSearchGlobalHistory:
    def test_passes_no_workspace_filter(self, mock_leader_setup):
        manager, indexer = mock_leader_setup
        indexer.backend.encode_query.return_value = np.random.randn(1024).astype(np.float32)

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.search._search_index", None),
            patch("kiro_ception.server.search") as mock_search,
        ):
            mock_search.return_value = {"results": [], "query": "test", "total_matches": 0}
            from kiro_ception.server import search_global_history

            search_global_history(query="test")

            # Verify workspace is None (global)
            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["workspace"] is None

    def test_source_defaults_to_all(self, mock_leader_setup):
        manager, indexer = mock_leader_setup

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.search._search_index", None),
            patch("kiro_ception.server.search") as mock_search,
        ):
            mock_search.return_value = {"results": [], "query": "test", "total_matches": 0}
            from kiro_ception.server import search_global_history

            search_global_history(query="test")

            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["source"] is None

    def test_source_cli_filter(self, mock_leader_setup):
        manager, indexer = mock_leader_setup

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.search._search_index", None),
            patch("kiro_ception.server.search") as mock_search,
        ):
            mock_search.return_value = {"results": [], "query": "test", "total_matches": 0}
            from kiro_ception.server import search_global_history
            from kiro_ception.models import Source

            search_global_history(query="test", source="cli")

            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["source"] == Source.CLI

    def test_source_ide_filter(self, mock_leader_setup):
        manager, indexer = mock_leader_setup

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.search._search_index", None),
            patch("kiro_ception.server.search") as mock_search,
        ):
            mock_search.return_value = {"results": [], "query": "test", "total_matches": 0}
            from kiro_ception.server import search_global_history
            from kiro_ception.models import Source

            search_global_history(query="test", source="ide")

            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["source"] == Source.IDE


# --- get_indexing_status ---


class TestGetIndexingStatus:
    def test_leader_returns_status(self, mock_leader_setup):
        manager, indexer = mock_leader_setup

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
        ):
            from kiro_ception.server import get_indexing_status

            result = get_indexing_status()

            assert result["state"] == "idle"
            assert result["sessions_total"] == 100
            assert result["embedding_count"] == 1000

    def test_follower_forwards_to_leader(self, mock_follower_setup):
        manager = mock_follower_setup
        manager.follower_instance.get_status.return_value = {
            "state": "indexing",
            "sessions_total": 50,
        }

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server._initialized", True),
        ):
            from kiro_ception.server import get_indexing_status

            result = get_indexing_status()

            assert result["state"] == "indexing"
            manager.follower_instance.get_status.assert_called_once()

    def test_follower_handles_leader_unavailable(self, mock_follower_setup):
        manager = mock_follower_setup
        manager.follower_instance.get_status.side_effect = ConnectionError("refused")

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server._initialized", True),
        ):
            from kiro_ception.server import get_indexing_status

            result = get_indexing_status()

            assert result["state"] == "unknown"
            assert "error" in result

    def test_includes_search_ready_when_index_loaded(self, mock_leader_setup):
        """When the search index has messages, search_ready should be True."""
        manager, indexer = mock_leader_setup

        mock_search_index = MagicMock()
        mock_search_index.message_count = 47000

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server.get_search_index", return_value=mock_search_index),
            patch("kiro_ception.server._initialized", True),
        ):
            from kiro_ception.server import get_indexing_status

            result = get_indexing_status()

            assert result["search_ready"] is True
            assert result["search_message_count"] == 47000

    def test_search_ready_false_when_index_empty(self, mock_leader_setup):
        """When the search index is empty, search_ready should be False."""
        manager, indexer = mock_leader_setup

        mock_search_index = MagicMock()
        mock_search_index.message_count = 0

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server.get_search_index", return_value=mock_search_index),
            patch("kiro_ception.server._initialized", True),
        ):
            from kiro_ception.server import get_indexing_status

            result = get_indexing_status()

            assert result["search_ready"] is False
            assert result["search_message_count"] == 0

    def test_includes_last_completed_at(self, mock_leader_setup):
        """Status should include last_completed_at from the indexer."""
        manager, indexer = mock_leader_setup

        # Simulate a prior successful run
        indexer.status.to_dict.return_value = {
            "state": "indexing",
            "sessions_total": 4333,
            "sessions_processed": 0,
            "embedding_count": 26000,
            "last_completed_at": "2026-06-05T22:00:00",
            "errors": 0,
        }

        mock_search_index = MagicMock()
        mock_search_index.message_count = 47000

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server.get_search_index", return_value=mock_search_index),
            patch("kiro_ception.server._initialized", True),
        ):
            from kiro_ception.server import get_indexing_status

            result = get_indexing_status()

            assert result["last_completed_at"] == "2026-06-05T22:00:00"
            assert result["search_ready"] is True


# --- rescan ---


class TestRescan:
    def test_leader_triggers_rescan(self, mock_leader_setup):
        manager, indexer = mock_leader_setup

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
        ):
            from kiro_ception.server import rescan

            result = rescan()

            assert result["status"] == "rescan_triggered"
            indexer.trigger_rescan.assert_called_once()
            indexer.trigger_reindex.assert_not_called()

    def test_leader_triggers_full_rescan(self, mock_leader_setup):
        manager, indexer = mock_leader_setup

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
        ):
            from kiro_ception.server import rescan

            result = rescan(full=True)

            assert result["status"] == "full_rescan_triggered"
            indexer.trigger_reindex.assert_called_once()
            indexer.trigger_rescan.assert_not_called()

    def test_follower_forwards_rescan(self, mock_follower_setup):
        manager = mock_follower_setup
        manager.follower_instance.trigger_rescan.return_value = {"status": "rescan_triggered"}

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server._initialized", True),
        ):
            from kiro_ception.server import rescan

            result = rescan()

            assert result["status"] == "rescan_triggered"
            manager.follower_instance.trigger_rescan.assert_called_once()

    def test_follower_forwards_full_rescan(self, mock_follower_setup):
        manager = mock_follower_setup
        manager.follower_instance.trigger_reindex.return_value = {"status": "full_rescan_triggered"}

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server._initialized", True),
        ):
            from kiro_ception.server import rescan

            result = rescan(full=True)

            assert result["status"] == "full_rescan_triggered"
            manager.follower_instance.trigger_reindex.assert_called_once()


# --- reload_config ---


class TestReloadConfig:
    def test_no_changes(self, mock_leader_setup):
        manager, indexer = mock_leader_setup

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.config.reload_config") as mock_reload,
            patch("kiro_ception.config.diff_configs", return_value=[]),
        ):
            from kiro_ception.config import Config
            mock_reload.return_value = (Config(), Config())

            from kiro_ception.server import reload_config

            result = reload_config()

            assert result["status"] == "no_changes"

    def test_safe_changes_applied(self, mock_leader_setup):
        manager, indexer = mock_leader_setup

        safe_change = [{
            "key": "indexing.throttle_ms",
            "old": 0,
            "new": 100,
            "impact": "safe",
        }]

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.config.reload_config") as mock_reload,
            patch("kiro_ception.config.diff_configs", return_value=safe_change),
        ):
            from kiro_ception.config import Config
            mock_reload.return_value = (Config(), Config())

            from kiro_ception.server import reload_config

            result = reload_config()

            assert result["status"] == "config_reloaded"
            assert "indexing.throttle_ms" in result["applied"]
            assert result["warnings"] == []

    def test_breaking_changes_deferred(self, mock_leader_setup):
        manager, indexer = mock_leader_setup

        breaking_change = [{
            "key": "embedding.model",
            "old": "old-model",
            "new": "new-model",
            "impact": "requires_reindex",
        }]

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.config.reload_config") as mock_reload,
            patch("kiro_ception.config.diff_configs", return_value=breaking_change),
        ):
            from kiro_ception.config import Config
            mock_reload.return_value = (Config(), Config())

            from kiro_ception.server import reload_config

            result = reload_config()

            assert result["status"] == "config_reloaded"
            assert len(result["deferred"]) == 1
            assert "rescan" in result["deferred"][0]
            assert len(result["warnings"]) == 1


# --- get_config ---


class TestGetConfig:
    """Tests for the get_config MCP tool.

    Validates that the tool returns a properly structured config dict
    without hitting infinite recursion (the original bug: the tool function
    was named get_config() and called get_config() internally, which recursed
    into itself instead of calling the config module's loader).
    """

    @pytest.fixture
    def mock_config_setup(self):
        """Create mocks with specific cache values for get_config tests."""
        manager = MagicMock()
        manager.is_leader = True

        indexer = MagicMock()
        indexer.backend = MagicMock()
        indexer.backend.fingerprint.return_value = (
            "openai-compatible:http://localhost:11434/v1:qwen3-embedding:4b:1024"
        )
        indexer.cache = MagicMock()
        indexer.cache.db_path = "/Users/test/.cache/kiro-ception/cache_abc123.db"
        indexer.cache.embedding_count = 26000
        indexer.cache.message_count = 30000
        indexer.cache.indexed_session_count = 4300
        return manager, indexer

    def test_does_not_recurse(self, mock_config_setup):
        """The get_config tool must not infinitely recurse."""
        manager, indexer = mock_config_setup
        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.server.get_memory_limit", return_value=4 * 1024 * 1024 * 1024),
        ):
            from kiro_ception.server import get_config

            # This would raise RecursionError before the fix
            result = get_config()
            assert isinstance(result, dict)

    def test_returns_expected_structure(self, mock_config_setup):
        """The get_config tool should return all expected top-level sections."""
        manager, indexer = mock_config_setup
        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.server.get_memory_limit", return_value=4 * 1024 * 1024 * 1024),
        ):
            from kiro_ception.server import get_config

            result = get_config()

            expected_keys = {
                "instance", "version", "paths", "sources", "embedding",
                "search", "memory", "indexing", "server", "peers", "cache",
            }
            assert set(result.keys()) == expected_keys

    def test_sources_structure(self, mock_config_setup):
        """The sources section should contain cli and ide sub-sections."""
        manager, indexer = mock_config_setup
        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.server.get_memory_limit", return_value=4 * 1024 * 1024 * 1024),
        ):
            from kiro_ception.server import get_config

            result = get_config()

            assert "cli" in result["sources"]
            assert "ide" in result["sources"]
            assert "enabled" in result["sources"]["cli"]
            assert "enabled" in result["sources"]["ide"]
            assert "patterns" in result["sources"]["ide"]

    def test_embedding_section(self, mock_config_setup):
        """The embedding section should expose backend, model, and fingerprint."""
        manager, indexer = mock_config_setup
        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.server.get_memory_limit", return_value=4 * 1024 * 1024 * 1024),
        ):
            from kiro_ception.server import get_config

            result = get_config()

            emb = result["embedding"]
            assert "backend" in emb
            assert "model" in emb
            assert "dimensions" in emb
            assert "batch_size" in emb
            assert "fingerprint" in emb
            assert "openai-compatible" in emb["fingerprint"]

    def test_cache_section(self, mock_config_setup):
        """The cache section should report counts from the indexer's cache."""
        manager, indexer = mock_config_setup
        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.server.get_memory_limit", return_value=4 * 1024 * 1024 * 1024),
        ):
            from kiro_ception.server import get_config

            result = get_config()

            cache = result["cache"]
            assert cache["embedding_count"] == 26000
            assert cache["message_count"] == 30000
            assert cache["indexed_sessions"] == 4300

    def test_memory_effective_limit(self, mock_config_setup):
        """The memory section should compute effective_limit_mb from get_memory_limit()."""
        manager, indexer = mock_config_setup
        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.server.get_memory_limit", return_value=4 * 1024 * 1024 * 1024),
        ):
            from kiro_ception.server import get_config

            result = get_config()

            # 4 GB = 4096 MB
            assert result["memory"]["effective_limit_mb"] == 4096

    def test_instance_section(self, mock_config_setup):
        """The instance section should include role info from the instance manager."""
        manager, indexer = mock_config_setup
        manager.get_role_info.return_value = {
            "role": "leader",
            "pid": 12345,
            "port": 19742,
        }
        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.server.get_memory_limit", return_value=4 * 1024 * 1024 * 1024),
        ):
            from kiro_ception.server import get_config

            result = get_config()

            assert result["instance"]["role"] == "leader"
            assert result["instance"]["pid"] == 12345
            assert result["instance"]["port"] == 19742

    def test_paths_section(self, mock_config_setup):
        """The paths section should include config file and cache paths."""
        manager, indexer = mock_config_setup
        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.server.get_memory_limit", return_value=4 * 1024 * 1024 * 1024),
        ):
            from kiro_ception.server import get_config

            result = get_config()

            paths = result["paths"]
            assert "config_file" in paths
            assert "config.toml" in paths["config_file"]
            assert "config_exists" in paths
            assert "cache_dir" in paths
            assert "cache_database" in paths
            assert "leader_lock" in paths
            assert "leader_info" in paths


# --- leader_search edge cases ---


class TestLeaderSearch:
    def test_backend_not_ready(self, mock_leader_setup):
        manager, indexer = mock_leader_setup
        indexer.backend = None

        with (
            patch("kiro_ception.search.get_instance_manager", return_value=manager),
            patch("kiro_ception.search.get_background_indexer", return_value=indexer),
            patch("kiro_ception.search._search_index", None),
            patch("kiro_ception.search.get_embedding_backend", side_effect=RuntimeError("not ready")),
        ):
            from kiro_ception.search import leader_search

            result = leader_search(
                query="test",
                workspace=None,
                source=None,
                after=None,
                before=None,
                context_size=3,
                threshold=0.2,
                max_results=10,
                offset=0,
            )

            assert "error" in result
            assert "Backend not ready" in result["error"]

    def test_empty_index_still_loading(self, mock_leader_setup):
        manager, indexer = mock_leader_setup
        indexer.backend.encode_query.return_value = np.random.randn(1024).astype(np.float32)

        # Create a search index that appears to be loading
        mock_index = MagicMock()
        mock_index.message_count = 0
        mock_index.is_loading = True
        mock_index.embedding_count_hint = 5000

        with (
            patch("kiro_ception.search.get_background_indexer", return_value=indexer),
            patch("kiro_ception.search.get_search_index", return_value=mock_index),
        ):
            from kiro_ception.search import leader_search

            result = leader_search(
                query="test",
                workspace=None,
                source=None,
                after=None,
                before=None,
                context_size=3,
                threshold=0.2,
                max_results=10,
                offset=0,
            )

            assert "still loading" in result["hint"]
            assert "5000" in result["hint"]

    def test_query_embedding_failure(self, mock_leader_setup):
        manager, indexer = mock_leader_setup
        indexer.backend.encode_query.side_effect = RuntimeError("model crashed")

        mock_index = MagicMock()
        mock_index.message_count = 100
        mock_index.is_loading = False

        with (
            patch("kiro_ception.search.get_background_indexer", return_value=indexer),
            patch("kiro_ception.search.get_search_index", return_value=mock_index),
        ):
            from kiro_ception.search import leader_search

            result = leader_search(
                query="test",
                workspace=None,
                source=None,
                after=None,
                before=None,
                context_size=3,
                threshold=0.2,
                max_results=10,
                offset=0,
            )

            assert "error" in result
            assert "Failed to embed query" in result["error"]
