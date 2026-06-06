"""Configuration management for Kiro Ception."""

import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# Default paths
CONFIG_DIR = Path.home() / ".config" / "kiro-ception"
CONFIG_FILE = CONFIG_DIR / "config.toml"
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config.default.toml"

# Embedding constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Memory constants
BYTES_PER_MESSAGE = 2600
DEFAULT_MEMORY_FRACTION = 1 / 3


def expand_path(path: str) -> Path:
    """Expand ~ and return Path."""
    return Path(path).expanduser()


def find_first_existing(paths: list[str]) -> Path | None:
    """Return first existing path from list."""
    for p in paths:
        expanded = expand_path(p)
        if expanded.exists():
            return expanded
    return None


@dataclass
class CLISourceConfig:
    """CLI source configuration."""

    enabled: bool = True
    paths: list[str] = field(default_factory=lambda: [
        "~/Library/Application Support/kiro-cli/data.sqlite3",
        "~/.local/share/kiro-cli/data.sqlite3",
        "~/AppData/Roaming/kiro-cli/data.sqlite3",
    ])

    @property
    def database_path(self) -> Path | None:
        """Get first existing database path."""
        return find_first_existing(self.paths)


@dataclass
class IDESourceConfig:
    """IDE source configuration."""

    enabled: bool = True
    patterns: list[str] = field(default_factory=lambda: [
        "~/Library/Application Support/Kiro/User/globalStorage/kiro.kiroagent/*/*.chat",
        "~/.config/Kiro/User/globalStorage/kiro.kiroagent/*/*.chat",
        "~/AppData/Roaming/Kiro/User/globalStorage/kiro.kiroagent/*/*.chat",
    ])

    def get_chat_files(self) -> list[Path]:
        """Get all .chat files matching patterns."""
        for pattern in self.patterns:
            expanded = expand_path(pattern)
            parts = expanded.parts
            parent_parts = []
            glob_pattern_parts = []
            in_glob = False
            for part in parts:
                if "*" in part or in_glob:
                    in_glob = True
                    glob_pattern_parts.append(part)
                else:
                    parent_parts.append(part)
            parent = Path(*parent_parts) if parent_parts else Path(".")
            glob_pattern = str(Path(*glob_pattern_parts)) if glob_pattern_parts else "*"
            if parent.exists():
                files = list(parent.glob(glob_pattern))
                if files:
                    return sorted(files)
        return []


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""

    backend: str = "sentence-transformers"  # "sentence-transformers" or "openai-compatible"
    model: str = EMBEDDING_MODEL
    cache_dir: str = "~/.cache/kiro-ception"
    # OpenAI-compatible backend settings
    api_base: str = ""  # e.g. "http://localhost:11434/v1" for Ollama
    api_key: str = ""  # Optional, for hosted providers (OpenAI, etc.)
    dimensions: int | None = None  # Output dimensions (None = model default)
    batch_size: int = 16  # Messages per embedding request (1 = simplest, higher = faster for small messages)

    @property
    def cache_path(self) -> Path:
        return expand_path(self.cache_dir)


@dataclass
class SearchConfig:
    """Search configuration."""

    default_threshold: float = 0.2
    default_max_results: int = 10
    default_context_window: int = 3
    recency_floor: float = 0.85  # Minimum recency multiplier (oldest message gets this)
    workspace_dir: str = ""  # Override workspace for search_project_history (empty = auto-detect)


@dataclass
class MemoryConfig:
    """Memory configuration."""

    fraction: float = DEFAULT_MEMORY_FRACTION
    limit_mb: int | None = None


@dataclass
class IndexingConfig:
    """Indexing behavior configuration."""

    throttle_ms: int = 0  # Sleep between batches (0 = full speed)
    rescan_interval_minutes: int = 10  # Minutes between automatic rescans (0 = disabled)


@dataclass
class ServerConfig:
    """Server/inter-process communication configuration."""

    leader_port: int = 19742  # Localhost-only HTTP port for leader


@dataclass
class PeersConfig:
    """Peer-to-peer federation configuration."""

    enabled: bool = False
    nodes: list[str] = field(default_factory=list)  # ["host:port", ...]
    secret: str = ""  # Shared passphrase for encryption (empty = unencrypted)
    timeout_seconds: int = 5  # Per-peer request timeout


@dataclass
class Config:
    """Main configuration."""

    cli: CLISourceConfig = field(default_factory=CLISourceConfig)
    ide: IDESourceConfig = field(default_factory=IDESourceConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    peers: PeersConfig = field(default_factory=PeersConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create config from dictionary."""
        cli_data = data.get("sources", {}).get("cli", {})
        ide_data = data.get("sources", {}).get("ide", {})
        emb_data = data.get("embedding", {})
        search_data = data.get("search", {})
        mem_data = data.get("memory", {})
        idx_data = data.get("indexing", {})
        srv_data = data.get("server", {})
        peers_data = data.get("peers", {})

        return cls(
            cli=CLISourceConfig(**cli_data) if cli_data else CLISourceConfig(),
            ide=IDESourceConfig(**ide_data) if ide_data else IDESourceConfig(),
            embedding=EmbeddingConfig(**emb_data) if emb_data else EmbeddingConfig(),
            search=SearchConfig(**search_data) if search_data else SearchConfig(),
            memory=MemoryConfig(**mem_data) if mem_data else MemoryConfig(),
            indexing=IndexingConfig(**idx_data) if idx_data else IndexingConfig(),
            server=ServerConfig(**srv_data) if srv_data else ServerConfig(),
            peers=PeersConfig(**peers_data) if peers_data else PeersConfig(),
        )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load configuration (cached)."""
    # Try user config first
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "rb") as f:
            return Config.from_dict(tomllib.load(f))

    # Fall back to default config
    if DEFAULT_CONFIG.exists():
        with open(DEFAULT_CONFIG, "rb") as f:
            return Config.from_dict(tomllib.load(f))

    # Use hardcoded defaults
    return Config()


def reload_config() -> tuple[Config, Config]:
    """Reload configuration from disk, clearing the cache.

    Returns:
        Tuple of (old_config, new_config)
    """
    old_config = get_config()
    get_config.cache_clear()
    new_config = get_config()
    return old_config, new_config


def diff_configs(old: Config, new: Config) -> list[dict]:
    """Compare two configs and return a list of changes with impact assessment.

    Returns list of dicts: {key, old, new, impact}
    impact is "safe" (hot-reloadable) or "requires_reindex"
    """
    changes = []

    # Embedding settings that require reindex
    reindex_keys = [
        ("embedding.backend", old.embedding.backend, new.embedding.backend),
        ("embedding.model", old.embedding.model, new.embedding.model),
        ("embedding.dimensions", old.embedding.dimensions, new.embedding.dimensions),
        ("embedding.api_base", old.embedding.api_base, new.embedding.api_base),
    ]

    # Settings that are safe to hot-reload
    safe_keys = [
        ("embedding.batch_size", old.embedding.batch_size, new.embedding.batch_size),
        ("embedding.api_key", old.embedding.api_key, new.embedding.api_key),
        ("indexing.throttle_ms", old.indexing.throttle_ms, new.indexing.throttle_ms),
        ("indexing.rescan_interval_minutes", old.indexing.rescan_interval_minutes, new.indexing.rescan_interval_minutes),
        ("search.default_threshold", old.search.default_threshold, new.search.default_threshold),
        ("search.default_max_results", old.search.default_max_results, new.search.default_max_results),
        ("search.default_context_window", old.search.default_context_window, new.search.default_context_window),
        ("search.recency_floor", old.search.recency_floor, new.search.recency_floor),
        ("search.workspace_dir", old.search.workspace_dir, new.search.workspace_dir),
        ("memory.fraction", old.memory.fraction, new.memory.fraction),
        ("memory.limit_mb", old.memory.limit_mb, new.memory.limit_mb),
        ("server.leader_port", old.server.leader_port, new.server.leader_port),
        ("sources.cli.enabled", old.cli.enabled, new.cli.enabled),
        ("sources.ide.enabled", old.ide.enabled, new.ide.enabled),
        ("peers.enabled", old.peers.enabled, new.peers.enabled),
        ("peers.nodes", old.peers.nodes, new.peers.nodes),
        ("peers.secret", "***" if old.peers.secret else "", "***" if new.peers.secret else ""),
        ("peers.timeout_seconds", old.peers.timeout_seconds, new.peers.timeout_seconds),
    ]

    for key, old_val, new_val in reindex_keys:
        if old_val != new_val:
            changes.append({
                "key": key,
                "old": old_val if old_val != "" else None,
                "new": new_val if new_val != "" else None,
                "impact": "requires_reindex",
            })

    for key, old_val, new_val in safe_keys:
        if old_val != new_val:
            changes.append({
                "key": key,
                "old": old_val,
                "new": new_val,
                "impact": "safe",
            })

    return changes
