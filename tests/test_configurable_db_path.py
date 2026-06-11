"""Property-based tests for configurable database path.

Verifies that all artifact paths are correctly derived from the configured
cache_dir, ensuring instance isolation via path containment and expansion.
"""

import string
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from kiro_ception.config import Config, EmbeddingConfig, expand_path


# --- Strategies ---

# Characters valid in directory names (avoiding OS-reserved and whitespace-only)
_path_segment_chars = st.characters(
    whitelist_categories=("L", "N"),
    whitelist_characters="_-.",
)

# Strategy for a single path segment (directory name)
_path_segment = st.text(
    alphabet=_path_segment_chars,
    min_size=1,
    max_size=20,
).filter(lambda s: s.strip(".") != "")  # No segments that are only dots


@st.composite
def absolute_path_strategy(draw) -> str:
    """Generate random absolute paths (posix-style for consistency)."""
    segments = draw(st.lists(_path_segment, min_size=1, max_size=5))
    return "/" + "/".join(segments)


@st.composite
def tilde_path_strategy(draw) -> str:
    """Generate random tilde-prefixed paths."""
    segments = draw(st.lists(_path_segment, min_size=1, max_size=5))
    return "~/" + "/".join(segments)


# Combined strategy: either absolute or tilde-prefixed
valid_cache_dir = st.one_of(absolute_path_strategy(), tilde_path_strategy())


def _make_config_with_cache_dir(cache_dir: str) -> Config:
    """Create a Config with the given cache_dir."""
    return Config(embedding=EmbeddingConfig(cache_dir=cache_dir))


# --- Property 1 Tests ---


class TestProperty1AllArtifactsDeriveFromCacheDir:
    """Property 1: All artifacts derive from configured cache directory.

    *For any* valid cache_dir string (absolute path or tilde-prefixed path),
    all derived artifact paths — the engine lock file, engine info file, and
    embedding cache database — SHALL be located within the expanded cache_dir
    directory.

    **Validates: Requirements 1.1, 4.1, 4.2, 5.1**
    """

    @given(cache_dir=valid_cache_dir)
    @settings(max_examples=200)
    def test_lock_path_within_cache_dir(self, cache_dir: str):
        """_get_lock_path() returns a path within the expanded cache directory."""
        config = _make_config_with_cache_dir(cache_dir)
        expected_parent = expand_path(cache_dir)

        with patch("kiro_ception.coordination.get_config", return_value=config):
            with patch.object(Path, "mkdir"):  # Don't create real dirs
                from kiro_ception.coordination import _get_lock_path
                lock_path = _get_lock_path()

        assert lock_path.parent == expected_parent, (
            f"Lock path {lock_path} is not within cache dir {expected_parent}"
        )
        assert str(lock_path).startswith(str(expected_parent)), (
            f"Lock path {lock_path} does not start with {expected_parent}"
        )

    @given(cache_dir=valid_cache_dir)
    @settings(max_examples=200)
    def test_engine_info_path_within_cache_dir(self, cache_dir: str):
        """_get_engine_info_path() returns a path within the expanded cache directory."""
        config = _make_config_with_cache_dir(cache_dir)
        expected_parent = expand_path(cache_dir)

        with patch("kiro_ception.coordination.get_config", return_value=config):
            from kiro_ception.coordination import _get_engine_info_path
            engine_info_path = _get_engine_info_path()

        assert engine_info_path.parent == expected_parent, (
            f"Engine info path {engine_info_path} not within cache dir {expected_parent}"
        )
        assert str(engine_info_path).startswith(str(expected_parent)), (
            f"Engine info path {engine_info_path} does not start with {expected_parent}"
        )

    @given(cache_dir=valid_cache_dir, fingerprint=st.text(min_size=1, max_size=50))
    @settings(max_examples=200)
    def test_cache_db_path_within_cache_dir(self, cache_dir: str, fingerprint: str):
        """_get_cache_db_path(fingerprint) returns a path within the expanded cache dir."""
        config = _make_config_with_cache_dir(cache_dir)
        expected_parent = expand_path(cache_dir)

        with patch("kiro_ception.cache.get_config", return_value=config):
            with patch.object(Path, "mkdir"):  # Don't create real dirs
                from kiro_ception.cache import _get_cache_db_path
                db_path = _get_cache_db_path(fingerprint)

        assert db_path.parent == expected_parent, (
            f"Cache DB path {db_path} is not within cache dir {expected_parent}"
        )
        assert str(db_path).startswith(str(expected_parent)), (
            f"Cache DB path {db_path} does not start with {expected_parent}"
        )

    @given(cache_dir=valid_cache_dir, fingerprint=st.text(min_size=1, max_size=50))
    @settings(max_examples=200)
    def test_all_artifacts_share_same_parent(self, cache_dir: str, fingerprint: str):
        """All three artifact paths share the same parent directory (the cache dir)."""
        config = _make_config_with_cache_dir(cache_dir)
        expected_parent = expand_path(cache_dir)

        with patch("kiro_ception.coordination.get_config", return_value=config):
            with patch("kiro_ception.cache.get_config", return_value=config):
                with patch.object(Path, "mkdir"):
                    from kiro_ception.coordination import (
                        _get_lock_path,
                        _get_engine_info_path,
                    )
                    from kiro_ception.cache import _get_cache_db_path

                    lock_path = _get_lock_path()
                    info_path = _get_engine_info_path()
                    db_path = _get_cache_db_path(fingerprint)

        assert lock_path.parent == expected_parent
        assert info_path.parent == expected_parent
        assert db_path.parent == expected_parent


# --- Property 3 Tests ---


class TestProperty3PathExpansionCorrectness:
    """Property 3: Path expansion correctness.

    *For any* cache_dir string, if it starts with `~` then the expanded path
    SHALL begin with the user's home directory and not contain a literal `~`;
    if it is already an absolute path without `~`, the expanded path SHALL be
    identical to the input path.

    **Validates: Requirements 3.1, 3.2**
    """

    @given(path=tilde_path_strategy())
    @settings(max_examples=200)
    def test_tilde_paths_expand_to_home_directory(self, path: str) -> None:
        """Tilde paths expand to start with user home and contain no literal ~.

        **Validates: Requirements 3.1**
        """
        expanded = expand_path(path)
        home = Path.home()

        # Expanded path must start with the user's home directory
        assert str(expanded).startswith(str(home)), (
            f"Expanded path {expanded} does not start with home {home}"
        )

        # Expanded path must not contain a literal tilde
        assert "~" not in str(expanded), (
            f"Expanded path {expanded} still contains literal '~'"
        )

    @given(path=absolute_path_strategy())
    @settings(max_examples=200)
    def test_absolute_paths_returned_unchanged(self, path: str) -> None:
        """Absolute paths without ~ are returned identical to input.

        **Validates: Requirements 3.2**
        """
        expanded = expand_path(path)

        # Absolute path without tilde should be returned unchanged
        # Compare as Path objects to handle platform separator normalization
        assert expanded == Path(path), (
            f"Absolute path was modified: input={Path(path)}, output={expanded}"
        )


# --- Property 2 Tests ---


def _get_all_artifact_paths(cache_dir: str, fingerprint: str = "test-fp") -> set[Path]:
    """Compute all artifact paths that would be derived from a given cache_dir.

    Replicates the logic from coordination.py and cache.py without calling
    get_config() or creating any filesystem state.
    """
    import hashlib

    cache_path = expand_path(cache_dir)
    fp_hash = hashlib.md5(fingerprint.encode()).hexdigest()[:12]
    return {
        cache_path / "engine.lock",
        cache_path / "engine.json",
        cache_path / f"cache_{fp_hash}.db",
    }


class TestProperty2DistinctCacheDirsNoOverlap:
    """Property 2: Distinct cache directories produce non-overlapping artifact sets.

    *For any* two distinct cache_dir values that expand to different absolute
    paths, the set of artifact paths derived from one SHALL have no path in
    common with the set derived from the other.

    **Validates: Requirements 1.4, 5.2**
    """

    @given(path_a=valid_cache_dir, path_b=valid_cache_dir)
    @settings(max_examples=200)
    def test_distinct_dirs_no_artifact_overlap(self, path_a: str, path_b: str):
        """Distinct cache directories produce completely disjoint artifact sets."""
        expanded_a = expand_path(path_a)
        expanded_b = expand_path(path_b)
        assume(expanded_a != expanded_b)

        artifacts_a = _get_all_artifact_paths(path_a)
        artifacts_b = _get_all_artifact_paths(path_b)

        overlap = artifacts_a & artifacts_b
        assert overlap == set(), (
            f"Artifact overlap detected between cache_dir={path_a!r} and "
            f"cache_dir={path_b!r}: {overlap}"
        )

    @given(
        path_a=valid_cache_dir,
        path_b=valid_cache_dir,
        fingerprint=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=200)
    def test_distinct_dirs_no_overlap_any_fingerprint(
        self, path_a: str, path_b: str, fingerprint: str
    ):
        """Non-overlapping holds regardless of the fingerprint used for cache DB naming."""
        expanded_a = expand_path(path_a)
        expanded_b = expand_path(path_b)
        assume(expanded_a != expanded_b)

        artifacts_a = _get_all_artifact_paths(path_a, fingerprint)
        artifacts_b = _get_all_artifact_paths(path_b, fingerprint)

        overlap = artifacts_a & artifacts_b
        assert overlap == set(), (
            f"Artifact overlap with fingerprint={fingerprint!r}: "
            f"cache_dir_a={path_a!r}, cache_dir_b={path_b!r}, overlap={overlap}"
        )


# --- Task 4.3: Documentation Content Unit Tests ---


class TestConfigDefaultTomlDocumentation:
    """Unit tests for documentation content in config.default.toml.

    Validates that the default config file contains proper documentation for
    multi-instance usage, cache_dir instance isolation, and engine_port uniqueness.

    **Validates: Requirements 2.1, 2.2, 2.3**
    """

    CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.default.toml"

    def _read_config(self) -> str:
        return self.CONFIG_PATH.read_text(encoding="utf-8")

    def test_contains_multi_instance_example_section(self):
        """config.default.toml contains the Multi-Instance Example section.

        **Validates: Requirements 2.1**
        """
        content = self._read_config()
        assert "Multi-Instance Example" in content, (
            "config.default.toml is missing the 'Multi-Instance Example' section heading"
        )

    def test_cache_dir_comment_mentions_instance_isolation(self):
        """cache_dir comment documents instance isolation for artifacts.

        **Validates: Requirements 2.2**
        """
        content = self._read_config()
        lower = content.lower()
        has_isolation = (
            "instance isolation" in lower or "instance-local artifacts" in lower
        )
        assert has_isolation, (
            "config.default.toml cache_dir comment does not mention "
            "'instance isolation' or 'instance-local artifacts'"
        )

    def test_engine_port_comment_mentions_unique_port(self):
        """engine_port comment explains that each instance requires a unique port.

        **Validates: Requirements 2.3**
        """
        content = self._read_config()
        assert (
            "unique engine_port" in content.lower()
            or "each concurrent kiro-ception instance requires a unique engine_port value"
            in content.lower()
        ), (
            "config.default.toml engine_port comment does not mention "
            "'unique' port requirement per instance"
        )


# --- Unit Tests: Default Behavior and Directory Creation (Task 4.1) ---


class TestDefaultCacheDir:
    """Unit tests for default cache_dir behavior.

    **Validates: Requirements 1.2**
    """

    def test_default_cache_dir_value(self):
        """EmbeddingConfig defaults to ~/.cache/kiro-ception when no config file exists."""
        config = EmbeddingConfig()
        assert config.cache_dir == "~/.cache/kiro-ception"

    def test_default_config_uses_default_cache_dir(self):
        """A Config() with no arguments uses ~/.cache/kiro-ception as cache_dir."""
        config = Config()
        assert config.embedding.cache_dir == "~/.cache/kiro-ception"

    def test_default_cache_path_expands_tilde(self):
        """The default cache_path property expands ~ to the user home directory."""
        config = EmbeddingConfig()
        expected = Path.home() / ".cache" / "kiro-ception"
        assert config.cache_path == expected


class TestDirectoryCreation:
    """Unit tests for directory creation in _get_lock_path and _get_cache_db_path.

    Verifies that mkdir(parents=True, exist_ok=True) is called before returning
    paths from functions that write artifacts.

    **Validates: Requirements 1.3**
    """

    def test_get_lock_path_calls_mkdir(self):
        """_get_lock_path() creates the cache directory with parents=True, exist_ok=True."""
        config = Config(embedding=EmbeddingConfig(cache_dir="/tmp/test-kiro-cache"))

        with patch("kiro_ception.coordination.get_config", return_value=config):
            with patch.object(Path, "mkdir") as mock_mkdir:
                from kiro_ception.coordination import _get_lock_path
                _get_lock_path()

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_get_lock_path_returns_correct_filename(self):
        """_get_lock_path() returns cache_dir / 'engine.lock'."""
        config = Config(embedding=EmbeddingConfig(cache_dir="/tmp/test-kiro-cache"))

        with patch("kiro_ception.coordination.get_config", return_value=config):
            with patch.object(Path, "mkdir"):
                from kiro_ception.coordination import _get_lock_path
                result = _get_lock_path()

        assert result == Path("/tmp/test-kiro-cache/engine.lock")

    def test_get_cache_db_path_calls_mkdir(self):
        """_get_cache_db_path() creates the cache directory with parents=True, exist_ok=True."""
        config = Config(embedding=EmbeddingConfig(cache_dir="/tmp/test-kiro-cache"))

        with patch("kiro_ception.cache.get_config", return_value=config):
            with patch.object(Path, "mkdir") as mock_mkdir:
                from kiro_ception.cache import _get_cache_db_path
                _get_cache_db_path("test-fingerprint")

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_get_cache_db_path_returns_path_in_cache_dir(self):
        """_get_cache_db_path() returns a .db file inside the configured cache directory."""
        config = Config(embedding=EmbeddingConfig(cache_dir="/tmp/test-kiro-cache"))

        with patch("kiro_ception.cache.get_config", return_value=config):
            with patch.object(Path, "mkdir"):
                from kiro_ception.cache import _get_cache_db_path
                result = _get_cache_db_path("test-fingerprint")

        assert result.parent == Path("/tmp/test-kiro-cache")
        assert result.name.startswith("cache_")
        assert result.name.endswith(".db")


# --- Unit Tests: Task 4.2 - Config Reload Propagation ---


class TestConfigReloadPropagation:
    """Test that when cache_dir changes and config is reloaded, derived paths use the new directory.

    Validates: Requirement 4.3
    """

    def test_lock_path_uses_new_cache_dir_after_reload(self, tmp_path):
        """After config reload with new cache_dir, _get_lock_path returns path in new dir."""
        old_dir = str(tmp_path / "old-cache")
        new_dir = str(tmp_path / "new-cache")

        old_config = _make_config_with_cache_dir(old_dir)
        new_config = _make_config_with_cache_dir(new_dir)

        with patch("kiro_ception.coordination.get_config", return_value=old_config):
            from kiro_ception.coordination import _get_lock_path
            old_lock_path = _get_lock_path()

        assert old_lock_path.parent == Path(old_dir)

        # Simulate config reload: now get_config returns new config
        with patch("kiro_ception.coordination.get_config", return_value=new_config):
            new_lock_path = _get_lock_path()

        assert new_lock_path.parent == Path(new_dir)
        assert old_lock_path != new_lock_path

    def test_engine_info_path_uses_new_cache_dir_after_reload(self, tmp_path):
        """After config reload with new cache_dir, _get_engine_info_path uses new dir."""
        old_dir = str(tmp_path / "old-cache")
        new_dir = str(tmp_path / "new-cache")

        old_config = _make_config_with_cache_dir(old_dir)
        new_config = _make_config_with_cache_dir(new_dir)

        with patch("kiro_ception.coordination.get_config", return_value=old_config):
            from kiro_ception.coordination import _get_engine_info_path
            old_info_path = _get_engine_info_path()

        assert old_info_path.parent == Path(old_dir)

        # Simulate config reload: now get_config returns new config
        with patch("kiro_ception.coordination.get_config", return_value=new_config):
            new_info_path = _get_engine_info_path()

        assert new_info_path.parent == Path(new_dir)
        assert old_info_path != new_info_path

    def test_cache_db_path_uses_new_cache_dir_after_reload(self, tmp_path):
        """After config reload with new cache_dir, _get_cache_db_path uses new dir."""
        old_dir = str(tmp_path / "old-cache")
        new_dir = str(tmp_path / "new-cache")

        old_config = _make_config_with_cache_dir(old_dir)
        new_config = _make_config_with_cache_dir(new_dir)

        fingerprint = "test-model-v1"

        with (
            patch("kiro_ception.cache.get_config", return_value=old_config),
            patch.object(Path, "mkdir"),
        ):
            from kiro_ception.cache import _get_cache_db_path
            old_db_path = _get_cache_db_path(fingerprint)

        assert old_db_path.parent == Path(old_dir)

        # Simulate config reload: now get_config returns new config
        with (
            patch("kiro_ception.cache.get_config", return_value=new_config),
            patch.object(Path, "mkdir"),
        ):
            new_db_path = _get_cache_db_path(fingerprint)

        assert new_db_path.parent == Path(new_dir)
        assert old_db_path != new_db_path

    def test_all_paths_switch_together_after_reload(self, tmp_path):
        """All derived paths consistently use the new cache_dir after reload."""
        dir_a = str(tmp_path / "instance-a")
        dir_b = str(tmp_path / "instance-b")

        config_a = _make_config_with_cache_dir(dir_a)
        config_b = _make_config_with_cache_dir(dir_b)

        fingerprint = "reload-test"

        from kiro_ception.cache import _get_cache_db_path
        from kiro_ception.coordination import _get_engine_info_path, _get_lock_path

        # All paths under dir_a
        with (
            patch("kiro_ception.coordination.get_config", return_value=config_a),
            patch("kiro_ception.cache.get_config", return_value=config_a),
            patch.object(Path, "mkdir"),
        ):
            lock_a = _get_lock_path()
            info_a = _get_engine_info_path()
            db_a = _get_cache_db_path(fingerprint)

        assert lock_a.parent == Path(dir_a)
        assert info_a.parent == Path(dir_a)
        assert db_a.parent == Path(dir_a)

        # After reload, all paths under dir_b
        with (
            patch("kiro_ception.coordination.get_config", return_value=config_b),
            patch("kiro_ception.cache.get_config", return_value=config_b),
            patch.object(Path, "mkdir"),
        ):
            lock_b = _get_lock_path()
            info_b = _get_engine_info_path()
            db_b = _get_cache_db_path(fingerprint)

        assert lock_b.parent == Path(dir_b)
        assert info_b.parent == Path(dir_b)
        assert db_b.parent == Path(dir_b)

        # No overlap between the two sets
        paths_a = {lock_a, info_a, db_a}
        paths_b = {lock_b, info_b, db_b}
        assert paths_a.isdisjoint(paths_b)

    def test_db_filename_preserved_after_cache_dir_change(self, tmp_path):
        """The database filename (cache_{hash}.db) stays the same; only the directory changes."""
        dir_old = str(tmp_path / "old")
        dir_new = str(tmp_path / "new")

        config_old = _make_config_with_cache_dir(dir_old)
        config_new = _make_config_with_cache_dir(dir_new)

        fingerprint = "same-fingerprint"

        from kiro_ception.cache import _get_cache_db_path

        with (
            patch("kiro_ception.cache.get_config", return_value=config_old),
            patch.object(Path, "mkdir"),
        ):
            old_path = _get_cache_db_path(fingerprint)

        with (
            patch("kiro_ception.cache.get_config", return_value=config_new),
            patch.object(Path, "mkdir"),
        ):
            new_path = _get_cache_db_path(fingerprint)

        # Filename is identical (derived from fingerprint hash)
        assert old_path.name == new_path.name
        # But directories differ
        assert old_path.parent != new_path.parent
