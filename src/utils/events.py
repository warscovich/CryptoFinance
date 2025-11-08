# utils/events.py
from databricks import sql
from contextlib import contextmanager
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import os
import json
import datetime as dt

# Load environment variables from .env file
# This ensures variables are loaded when running as a standalone script
load_dotenv()

# ---- Configuration (env-driven; no secrets hardcoded) ----
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").replace("https://", "")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
DATABRICKS_WAREHOUSE = os.getenv("DATABRICKS_WAREHOUSE_PATH", "")  
DATABRICKS_USER_ID = os.getenv("DATABRICKS_USER_ID", os.getenv("USER", "unknown_user"))

DEFAULT_CATALOG = os.getenv("DB_CATALOG", "workspace")
DEFAULT_SCHEMA  = os.getenv("DB_SCHEMA", "default")
DEFAULT_TABLE   = os.getenv("DB_EVENTS_TABLE", "app_events")   # fully qualified → default.default.app_events


def _require_env() -> None:
    missing = [k for k, v in {
        "DATABRICKS_HOST": DATABRICKS_HOST,
        "DATABRICKS_TOKEN": DATABRICKS_TOKEN,
        "DATABRICKS_WAREHOUSE_PATH": DATABRICKS_WAREHOUSE
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


@contextmanager
def sql_connection():
    """Context-managed Databricks SQL connection."""
    _require_env()
    conn = sql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_WAREHOUSE,
        access_token=DATABRICKS_TOKEN
    )
    try:
        yield conn
    finally:
        conn.close()


def ensure_events_table(
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    table: str = DEFAULT_TABLE
) -> None:
    """Create the events table if it doesn't exist (Delta, UC-governed)."""
    fq = f"{catalog}.{schema}.{table}"
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {fq} (
      event_ts   TIMESTAMP,
      user_id    STRING,
      event_type STRING,
      source     STRING,
      payload    STRING
    ) USING DELTA
    """
    with sql_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


def log_event(
    event_type: str,
    source: str,
    payload: Optional[Dict[str, Any]] = None,
    user_id: str = DATABRICKS_USER_ID,

    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    table: str = DEFAULT_TABLE
) -> None:
    """
    Insert a single event row into UC Delta table.
    Automatically ensures the table exists before inserting.
    payload is stored as JSON string (keep it small & serializable).
    """
    # Ensure table exists before attempting insert (idempotent, safe to call)
    ensure_events_table(catalog, schema, table)
    
    fq = f"{catalog}.{schema}.{table}"
    payload_json = json.dumps(payload or {}, separators=(",", ":"), ensure_ascii=False)

    sql_insert = f"""
    INSERT INTO {fq} (event_ts, user_id, event_type, source, payload)
    VALUES (current_timestamp(), ?, ?, ?, ?)
    """

    with sql_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_insert, (user_id, event_type, source, payload_json))
        conn.commit()



if __name__ == "__main__":
    ensure_events_table()  # one-time (safe to call repeatedly)

    log_event(
        event_type="initialization",
        source="cryptofinance",
        payload={"file": "src/utils/events.py", "method": "__name__"}
    )
    print("✓ Event logged")
