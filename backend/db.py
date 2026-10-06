"""
MySQL history storage for Retina Guard AI.

Every prediction is logged to a `predictions` table so you can later
build an admin dashboard / history view on top of it.

This module is written to FAIL SOFT: if MySQL isn't installed, isn't
running, or the credentials below don't match your setup, the app will
print a one-line warning and keep working WITHOUT history logging,
instead of crashing. This means you can run the whole app immediately
without setting up MySQL first, and wire up the database whenever
you're ready.

--- MySQL setup ---
1. Install MySQL Server and make sure it's running.
2. Create the database and table:
       mysql -u root -p < backend/schema.sql
3. Edit the DB_CONFIG below (or set the matching environment variables)
   to match your MySQL username/password.
"""

import os
import datetime

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME", "retina_guard"),
}

_connector_available = False
try:
    import mysql.connector  # noqa: F401
    _connector_available = True
except ImportError:
    _connector_available = False

_warned_once = False


def _warn(msg):
    global _warned_once
    if not _warned_once:
        print(f"[db] {msg} -- continuing without history logging.")
        _warned_once = True


def get_connection():
    if not _connector_available:
        _warn("mysql-connector-python is not installed")
        return None
    try:
        import mysql.connector
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Exception as exc:
        _warn(f"could not connect to MySQL ({exc})")
        return None


def save_prediction(filename, grade, confidence, lesion_count, mode):
    conn = get_connection()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (filename, grade, confidence, lesion_count, mode, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (filename, grade, confidence, lesion_count, mode, datetime.datetime.now()),
        )
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as exc:
        _warn(f"could not save prediction ({exc})")
        return False


def get_history(limit=50):
    conn = get_connection()
    if conn is None:
        return []
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        for row in rows:
            if isinstance(row.get("created_at"), datetime.datetime):
                row["created_at"] = row["created_at"].isoformat()
        return rows
    except Exception as exc:
        _warn(f"could not fetch history ({exc})")
        return []
