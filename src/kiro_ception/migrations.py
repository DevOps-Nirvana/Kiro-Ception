"""Database schema migrations for the embedding cache.

Each migration is a function that takes a sqlite3.Connection and applies
schema changes. Migrations run sequentially and are tracked via the
'schema_version' key in the meta table.

Version history:
- 1: Baseline schema (embeddings, messages, session_state, execution_index, meta)
- 2: Add FTS5 full-text search index on messages.searchable_text
"""

import logging
import sqlite3

logger = logging.getLogger(__name__)

# Current schema version — bump this when adding a new migration
CURRENT_SCHEMA_VERSION = 2


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
