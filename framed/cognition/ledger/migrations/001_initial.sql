-- cognition_schema_v1 — Slice A amended

CREATE TABLE IF NOT EXISTS schema_version (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
  episode_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  actor_id TEXT NOT NULL,
  asset_id TEXT NOT NULL,
  goal_type TEXT NOT NULL,
  goal_instance_id TEXT,
  status TEXT NOT NULL CHECK (status IN ('open','closed','quarantined','failed')),
  source_kind TEXT NOT NULL DEFAULT 'live',
  legacy_source TEXT,
  asset_filename TEXT,
  created_at TEXT NOT NULL,
  closed_at TEXT,
  perception_artefact_hash TEXT,
  state_version_id TEXT NOT NULL,
  request_fingerprint TEXT,
  final_fingerprint TEXT
);

CREATE TABLE IF NOT EXISTS cognitive_runs (
  run_id TEXT PRIMARY KEY,
  episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
  mode TEXT NOT NULL CHECK (mode IN ('baseline','memory_enabled','control','replay')),
  state_version_id TEXT NOT NULL,
  context_fingerprint TEXT,
  retrieval_enabled INTEGER NOT NULL,
  model_provenance_json TEXT NOT NULL,
  prompt_provenance_json TEXT NOT NULL,
  started_at TEXT NOT NULL,
  completed_at TEXT
);

CREATE TABLE IF NOT EXISTS episode_events (
  event_id TEXT PRIMARY KEY,
  episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
  run_id TEXT NOT NULL REFERENCES cognitive_runs(run_id),
  event_type TEXT NOT NULL,
  sequence_num INTEGER NOT NULL,
  recorded_at TEXT NOT NULL,
  artefact_hash TEXT,
  payload_json TEXT NOT NULL,
  UNIQUE(episode_id, sequence_num)
);

CREATE TABLE IF NOT EXISTS memory_references (
  memory_ref_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES cognitive_runs(run_id),
  target_episode_id TEXT NOT NULL,
  source_episode_id TEXT NOT NULL,
  source_event_id TEXT NOT NULL,
  ref_type TEXT NOT NULL,
  epistemic_status TEXT NOT NULL,
  lifecycle_status TEXT NOT NULL,
  memory_role TEXT NOT NULL,
  trust_level TEXT NOT NULL,
  category_score REAL NOT NULL,
  scene_score REAL NOT NULL,
  goal_score REAL NOT NULL,
  relation_score REAL NOT NULL,
  recency_score REAL NOT NULL,
  final_score REAL NOT NULL,
  contamination_flags_json TEXT NOT NULL,
  match_reason TEXT NOT NULL,
  artefact_hash TEXT,
  retrieved_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cognitive_state_versions (
  state_version_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  parent_version_id TEXT,
  label TEXT NOT NULL,
  created_at TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 0,
  snapshot_artefact_hash TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_state_per_workspace
  ON cognitive_state_versions(workspace_id)
  WHERE is_active = 1;

CREATE TABLE IF NOT EXISTS retrieval_index (
  episode_id TEXT PRIMARY KEY REFERENCES episodes(episode_id),
  workspace_id TEXT NOT NULL,
  actor_id TEXT NOT NULL,
  asset_id TEXT NOT NULL,
  scene_signature TEXT,
  category_signature TEXT,
  goal_type TEXT,
  goal_instance_id TEXT,
  recorded_at TEXT NOT NULL,
  closed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artefacts (
  artefact_hash TEXT PRIMARY KEY,
  schema_name TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  relative_path TEXT NOT NULL,
  byte_length INTEGER NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS episode_events_no_update
BEFORE UPDATE ON episode_events
BEGIN
  SELECT RAISE(ABORT, 'episode_events append-only');
END;

CREATE TRIGGER IF NOT EXISTS episode_events_no_delete
BEFORE DELETE ON episode_events
BEGIN
  SELECT RAISE(ABORT, 'episode_events append-only');
END;
