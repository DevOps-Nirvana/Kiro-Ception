"""Database schema migrations for the embedding cache.

Each migration is a function that takes a sqlite3.Connection and applies
schema changes. Migrations run sequentially and are tracked via the
'schema_version' key in the meta table.

Version history:
- 1: Baseline schema (embeddings, messages, session_state, execution_index, meta)
- 2: Add FTS5 full-text search index on messages.searchable_text
- 3: Add content_tier and tool_name columns for two-tier content model
"""

import logging
import sqlite3
import time

logger = logging.getLogger(__name__)

# Current schema version — bump this when adding a new migration
CURRENT_SCHEMA_VERSION = 3


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Get the current schema version from the database.

    Returns 0 if the meta table doesn't exist (brand new DB before first init),
    or 1 if meta exists but no schema_version is set (pre-migration DB).
    """
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row:
            return int(row[0])
    except sqlite3.OperationalError:
        # meta table doesn't exist yet — this is a fresh DB
        return 0

    # meta table exists but no schema_version key — this is a v1 DB
    # (existed before we added migration tracking)
    return 1


def run_migrations(conn: sqlite3.Connection):
    """Run any pending migrations to bring the DB up to CURRENT_SCHEMA_VERSION.

    This is idempotent — calling it multiple times is safe.
    """
    current = get_schema_version(conn)

    if current >= CURRENT_SCHEMA_VERSION:
        return  # Already up to date

    if current < 1:
        # Brand new DB — tables will be created by _create_tables().
        # Just set the version after tables are created.
        # (This function is called after _create_tables, so tables exist.)
        pass

    if current < 2:
        _migrate_v1_to_v2(conn)

    if current < 3:
        _migrate_v2_to_v3(conn)

    # Record the final schema version
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(CURRENT_SCHEMA_VERSION),),
    )
    conn.commit()
    logger.info(f"Database migrated from v{current} to v{CURRENT_SCHEMA_VERSION}")


def _migrate_v1_to_v2(conn: sqlite3.Connection):
    """Migration v1 → v2: Add FTS5 full-text search index.

    Creates an FTS5 virtual table backed by messages.searchable_text.
    Uses a content-sync approach where the FTS table stores its own copy
    of the text (content table), allowing independent FTS queries.
    """
    logger.info("Running migration v1 → v2: Adding FTS5 full-text search index...")

    # Create the FTS5 virtual table
    # We use content="" (contentless) would save space but prevents highlight().
    # Instead we use a regular FTS5 table with uuid as a rowid-linked column
    # for joining back to the messages table.
    conn.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            searchable_text,
            content='messages',
            content_rowid='rowid',
            tokenize='porter unicode61'
        );

        -- Triggers to keep FTS in sync with the messages table
        CREATE TRIGGER IF NOT EXISTS messages_fts_insert
        AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, searchable_text)
            VALUES (new.rowid, new.searchable_text);
        END;

        CREATE TRIGGER IF NOT EXISTS messages_fts_delete
        AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, searchable_text)
            VALUES ('delete', old.rowid, old.searchable_text);
        END;

        CREATE TRIGGER IF NOT EXISTS messages_fts_update
        AFTER UPDATE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, searchable_text)
            VALUES ('delete', old.rowid, old.searchable_text);
            INSERT INTO messages_fts(rowid, searchable_text)
            VALUES (new.rowid, new.searchable_text);
        END;
    """)

    # Populate FTS from existing messages
    row_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    if row_count > 0:
        logger.info(f"Populating FTS5 index from {row_count} existing messages...")
        conn.execute("""
            INSERT INTO messages_fts(rowid, searchable_text)
            SELECT rowid, searchable_text FROM messages
        """)
        conn.commit()
        logger.info("FTS5 index populated.")
    else:
        conn.commit()


def _migrate_v2_to_v3(conn: sqlite3.Connection):
    """Migration v2 → v3: Add content_tier and tool_name columns.

    Adds columns for the two-tier content model:
    - content_tier: classifies messages as 'conversation' or 'tool_context'
    - tool_name: populated only for tool_context messages
    - An index on content_tier for filtered queries

    Handles "column already exists" gracefully and retries on database lock
    with exponential backoff.
    """
    logger.info("Running migration v2 → v3: Adding content_tier and tool_name columns...")

    _execute_with_retry(
        conn,
        "ALTER TABLE messages ADD COLUMN content_tier TEXT NOT NULL DEFAULT 'conversation'",
        "content_tier column",
    )

    _execute_with_retry(
        conn,
        "ALTER TABLE messages ADD COLUMN tool_name TEXT",
        "tool_name column",
    )

    _execute_with_retry(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_messages_content_tier ON messages(content_tier)",
        "content_tier index",
    )

    conn.commit()


def _execute_with_retry(
    conn: sqlite3.Connection, sql: str, description: str, max_attempts: int = 3
):
    """Execute a SQL statement with exponential backoff on database lock.

    Gracefully handles "column already exists" errors by logging and skipping.
    Retries up to max_attempts times on database lock with 1s/2s/4s delays.
    """
    for attempt in range(max_attempts):
        try:
            conn.execute(sql)
            return
        except sqlite3.OperationalError as e:
            error_msg = str(e).lower()

            # Column/index already exists — skip gracefully
            if "duplicate column" in error_msg or "already exists" in error_msg:
                logger.info(f"Skipping {description}: already exists")
                return

            # Database is locked — retry with exponential backoff
            if "database is locked" in error_msg:
                delay = 2**attempt  # 1s, 2s, 4s
                if attempt < max_attempts - 1:
                    logger.warning(
                        f"Database locked while adding {description}, "
                        f"retrying in {delay}s (attempt {attempt + 1}/{max_attempts})"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Database locked after {max_attempts} attempts "
                        f"while adding {description}"
                    )
                    raise
            else:
                # Unexpected error — re-raise
                raise
