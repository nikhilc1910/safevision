-- SafeVision SQLite schema
-- Designed for a single-site deployment. If multi-site comes up in v2.0,
-- add a site_id FK rather than duplicating camera_id across tables.

PRAGMA journal_mode = WAL;  -- concurrent reads during dashboard polling

-- ── violations ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS violations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id       TEXT    NOT NULL,
    zone_id         TEXT    NOT NULL,
    vtype           TEXT    NOT NULL,   -- 'no-helmet', 'no-safety-vest', etc.
    confidence      REAL    NOT NULL,
    bbox_x1         INTEGER,
    bbox_y1         INTEGER,
    bbox_x2         INTEGER,
    bbox_y2         INTEGER,
    timestamp_utc   TEXT    NOT NULL,   -- ISO 8601, UTC
    shift_id        TEXT,               -- populated by caller if shift tracking is active
    frame_snap      BLOB,               -- JPEG bytes; nullable — only stored if STORE_SNAPS=true
    false_positive  INTEGER DEFAULT 0   -- operator-marked FPs feed active learning loop in v2.0
);

-- Query patterns: recent violations per zone, per shift, dashboard refresh
CREATE INDEX IF NOT EXISTS idx_violations_ts    ON violations (timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_violations_zone  ON violations (zone_id);
CREATE INDEX IF NOT EXISTS idx_violations_shift ON violations (shift_id);

-- No face crops, no biometric fields — bbox coordinates only.
-- Frame review works on box coordinates; re-identification is out of scope.


-- ── zones ─────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS zones (
    zone_id     TEXT PRIMARY KEY,
    camera_id   TEXT NOT NULL,
    label       TEXT,
    polygon_json TEXT NOT NULL,   -- JSON array of [x, y] points
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);


-- ── cameras ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS cameras (
    camera_id   TEXT PRIMARY KEY,
    label       TEXT,
    stream_url  TEXT NOT NULL,
    active      INTEGER DEFAULT 1
);
