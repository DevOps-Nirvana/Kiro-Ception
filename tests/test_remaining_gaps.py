"""Tests for remaining coverage gaps.

- cli_loader._extract_text_from_content edge cases
- Follower promotion on engine failure
- get_config() loading from a real TOML file
- ide_loader._build_execution_index incremental behavior
"""

import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kiro_ception.config import Config


# --- CLI _extract_text_from_content ---


class TestCLIExtractTextFromContent:
    """Tests for cli_loader's version of _extract_text_from_content."""

    def test_none_returns_empty(self):
        from kiro_ception.cli_loader import _extract_text_from_content
        assert _extract_text_from_content(None) == ""

    def test_string_passthrough(self):
        from kiro_ception.cli_loader import _extract_text_from_content
        assert _extract_text_from_content("hello world") == "hello world"

    def test_prompt_structure(self):
        """The CLI-specific {"Prompt": {"prompt": "..."}} format."""
        from kiro_ception.cli_loader import _extract_text_from_content
        content = {"Prompt": {"prompt": "How do I fix this bug?"}}
        assert _extract_text_from_content(content) == "How do I fix this bug?"

    def test_prompt_structure_empty_prompt(self):
        from kiro_ception.cli_loader import _extract_text_from_content
        content = {"Prompt": {"prompt": ""}}
        assert _extract_text_from_content(content) == ""

    def test_prompt_structure_missing_prompt_key(self):
        from kiro_ception.cli_loader import _extract_text_from_content
        content = {"Prompt": {"other": "value"}}
        assert _extract_text_from_content(content) == ""

    def test_dict_with_text_key(self):
        from kiro_ception.cli_loader import _extract_text_from_content
        assert _extract_text_from_content({"text": "from text key"}) == "from text key"

    def test_dict_with_prompt_key(self):
        from kiro_ception.cli_loader import _extract_text_from_content
        assert _extract_text_from_content({"prompt": "from prompt key"}) == "from prompt key"

    def test_list_of_text_items(self):
        from kiro_ception.cli_loader import _extract_text_from_content
        content = [
            {"type": "text", "text": "part one"},
            {"type": "text", "text": "part two"},
        ]
        result = _extract_text_from_content(content)
        assert "part one" in result
        assert "part two" in result

    def test_list_of_strings(self):
        from kiro_ception.cli_loader import _extract_text_from_content
        result = _extract_text_from_content(["line one", "line two"])
        assert "line one" in result
        assert "line two" in result

    def test_empty_dict(self):
        from kiro_ception.cli_loader import _extract_text_from_content
        assert _extract_text_from_content({}) == ""

    def test_unknown_type(self):
        from kiro_ception.cli_loader import _extract_text_from_content
        assert _extract_text_from_content(12345) == ""


# --- Follower promotion on engine failure ---


class TestFollowerPromotion:
    """Removed: follower promotion no longer exists in the search layer.

    The engine process is always the engine — there's no follower state in search.py.
    The MCP client handles engine lifecycle via engine_client.py.
    """
    pass


# --- get_config with real TOML file ---


class TestGetConfigFromTOML:
    """Test loading config from actual TOML files on disk."""

    def test_loads_from_user_config(self, tmp_path, monkeypatch):
        """get_config() should load from the user config file."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[embedding]
backend = "openai-compatible"
model = "test-model"
api_base = "http://localhost:11434/v1"
dimensions = 512
batch_size = 4

[search]
default_threshold = 0.3
default_max_results = 20

[indexing]
throttle_ms = 50
rescan_interval_minutes = 5

[server]
engine_port = 12345
""")

        monkeypatch.setattr("kiro_ception.config.CONFIG_FILE", config_file)

        # Clear the lru_cache
        from kiro_ception.config import get_config
        get_config.cache_clear()

        try:
            config = get_config()

            assert config.embedding.backend == "openai-compatible"
            assert config.embedding.model == "test-model"
            assert config.embedding.api_base == "http://localhost:11434/v1"
            assert config.embedding.dimensions == 512
            assert config.embedding.batch_size == 4
            assert config.search.default_threshold == 0.3
            assert config.search.default_max_results == 20
            assert config.indexing.throttle_ms == 50
            assert config.indexing.rescan_interval_minutes == 5
            assert config.server.engine_port == 12345
        finally:
            get_config.cache_clear()

    def test_loads_from_default_config(self, tmp_path, monkeypatch):
        """Falls back to default config when user config doesn't exist."""
        nonexistent = tmp_path / "nonexistent" / "config.toml"
        default_config = tmp_path / "config.default.toml"
        default_config.write_text("""
[embedding]
model = "default-model"
batch_size = 16
""")

        monkeypatch.setattr("kiro_ception.config.CONFIG_FILE", nonexistent)
        monkeypatch.setattr("kiro_ception.config.DEFAULT_CONFIG", default_config)

        from kiro_ception.config import get_config
        get_config.cache_clear()

        try:
            config = get_config()
            assert config.embedding.model == "default-model"
            assert config.embedding.batch_size == 16
        finally:
            get_config.cache_clear()

    def test_falls_back_to_hardcoded_defaults(self, tmp_path, monkeypatch):
        """No config files at all → use hardcoded defaults."""
        monkeypatch.setattr("kiro_ception.config.CONFIG_FILE", tmp_path / "nope.toml")
        monkeypatch.setattr("kiro_ception.config.DEFAULT_CONFIG", tmp_path / "also_nope.toml")

        from kiro_ception.config import get_config
        get_config.cache_clear()

        try:
            config = get_config()
            # Should use dataclass defaults
            assert config.embedding.backend == "sentence-transformers"
            assert config.embedding.model == "all-MiniLM-L6-v2"
            assert config.search.default_threshold == 0.2
        finally:
            get_config.cache_clear()

    def test_partial_toml_merges_with_defaults(self, tmp_path, monkeypatch):
        """A TOML with only some sections still uses defaults for the rest."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[search]
default_threshold = 0.5
""")

        monkeypatch.setattr("kiro_ception.config.CONFIG_FILE", config_file)

        from kiro_ception.config import get_config
        get_config.cache_clear()

        try:
            config = get_config()
            # Specified value
            assert config.search.default_threshold == 0.5
            # Everything else is default
            assert config.embedding.backend == "sentence-transformers"
            assert config.server.engine_port == 19742
        finally:
            get_config.cache_clear()

    def test_sources_section(self, tmp_path, monkeypatch):
        """Sources configuration loads correctly."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[sources.cli]
enabled = false

[sources.ide]
enabled = true
patterns = ["/custom/path/*.chat"]
""")

        monkeypatch.setattr("kiro_ception.config.CONFIG_FILE", config_file)

        from kiro_ception.config import get_config
        get_config.cache_clear()

        try:
            config = get_config()
            assert config.cli.enabled is False
            assert config.ide.enabled is True
            assert config.ide.patterns == ["/custom/path/*.chat"]
        finally:
            get_config.cache_clear()


# --- ide_loader._build_execution_index incremental behavior ---


class TestBuildExecutionIndex:
    """Test the incremental execution index build logic."""

    def test_builds_index_from_files(self, tmp_path):
        """Should parse execution logs and map chatSessionId → file paths."""
        exec_dir = tmp_path / "exec_logs"
        exec_dir.mkdir()

        # Create two exec log files
        (exec_dir / "exec_file_1").write_text(json.dumps({
            "chatSessionId": "session-A",
            "actions": [{"actionType": "say", "output": {"message": "hello"}}],
        }))
        (exec_dir / "exec_file_2").write_text(json.dumps({
            "chatSessionId": "session-B",
            "actions": [],
        }))

        # Mock the cache and exec log dirs
        mock_cache = MagicMock()
        mock_cache.get_all_exec_file_mtimes.return_value = {}
        mock_cache.conn.execute.return_value = [
            (str(exec_dir / "exec_file_1"), "session-A"),
            (str(exec_dir / "exec_file_2"), "session-B"),
        ]
        mock_cache.get_exec_index_count.return_value = 2

        import kiro_ception.ide_loader as loader
        # Reset the module-level cache
        loader._execution_index = None
        loader._execution_index_built = False

        with (
            patch("kiro_ception.ide_loader._get_execution_log_dirs", return_value=[exec_dir]),
            patch("kiro_ception.ide_loader._get_exec_cache", return_value=mock_cache),
        ):
            index = loader._build_execution_index()

        assert "session-A" in index
        assert "session-B" in index

        # Clean up module state
        loader._execution_index = None
        loader._execution_index_built = False

    def test_skips_unchanged_files(self, tmp_path):
        """Files with matching mtime should not be re-parsed."""
        exec_dir = tmp_path / "exec_logs"
        exec_dir.mkdir()

        exec_file = exec_dir / "exec_file_1"
        exec_file.write_text(json.dumps({
            "chatSessionId": "session-A",
            "actions": [],
        }))
        file_mtime = exec_file.stat().st_mtime

        # Mock cache already has this file at the current mtime
        mock_cache = MagicMock()
        mock_cache.get_all_exec_file_mtimes.return_value = {
            str(exec_file): file_mtime  # Already indexed at same mtime
        }
        mock_cache.conn.execute.return_value = [
            (str(exec_file), "session-A"),
        ]
        mock_cache.get_exec_index_count.return_value = 1

        import kiro_ception.ide_loader as loader
        loader._execution_index = None
        loader._execution_index_built = False

        with (
            patch("kiro_ception.ide_loader._get_execution_log_dirs", return_value=[exec_dir]),
            patch("kiro_ception.ide_loader._get_exec_cache", return_value=mock_cache),
        ):
            index = loader._build_execution_index()

        # Should NOT have called put_exec_files_batch (nothing new to store)
        mock_cache.put_exec_files_batch.assert_not_called()

        loader._execution_index = None
        loader._execution_index_built = False
