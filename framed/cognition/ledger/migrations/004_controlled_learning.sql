-- cognition_schema_v4 — Slice B controlled learning (additive)

CREATE TABLE IF NOT EXISTS outcomes (
  outcome_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  source_episode_id TEXT NOT NULL,
  source_run_id TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('human_feedback','testdaemon_eval')),
  verdict TEXT NOT NULL CHECK (verdict IN ('useful','not_useful','correction')),
  created_by TEXT NOT NULL CHECK (created_by IN ('human','testdaemon')),
  artefact_hash TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS update_proposals (
  proposal_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  base_state_version_id TEXT NOT NULL,
  outcome_id TEXT NOT NULL REFERENCES outcomes(outcome_id),
  kind TEXT NOT NULL CHECK (kind IN ('promote_episode_belief')),
  created_by TEXT NOT NULL CHECK (created_by = 'proposal_generator'),
  artefact_hash TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proposal_evaluations (
  evaluation_id TEXT PRIMARY KEY,
  proposal_id TEXT NOT NULL REFERENCES update_proposals(proposal_id),
  status TEXT NOT NULL CHECK (status IN ('pass','fail')),
  artefact_hash TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS promotion_decisions (
  decision_id TEXT PRIMARY KEY,
  proposal_id TEXT NOT NULL REFERENCES update_proposals(proposal_id),
  evaluation_id TEXT,
  action TEXT NOT NULL CHECK (action IN ('accept','reject')),
  authority_kind TEXT NOT NULL CHECK (authority_kind IN ('human','testdaemon')),
  actor_id TEXT NOT NULL,
  resulting_state_version_id TEXT,
  artefact_hash TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_one_decision_per_proposal
  ON promotion_decisions(proposal_id);

CREATE TABLE IF NOT EXISTS rollback_records (
  rollback_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  from_state_version_id TEXT NOT NULL,
  to_state_version_id TEXT NOT NULL,
  authority_kind TEXT NOT NULL CHECK (authority_kind IN ('human','testdaemon')),
  actor_id TEXT NOT NULL,
  artefact_hash TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS outcomes_no_update
BEFORE UPDATE ON outcomes
BEGIN
  SELECT RAISE(ABORT, 'outcomes append-only');
END;

CREATE TRIGGER IF NOT EXISTS outcomes_no_delete
BEFORE DELETE ON outcomes
BEGIN
  SELECT RAISE(ABORT, 'outcomes append-only');
END;

CREATE TRIGGER IF NOT EXISTS update_proposals_no_update
BEFORE UPDATE ON update_proposals
BEGIN
  SELECT RAISE(ABORT, 'update_proposals append-only');
END;

CREATE TRIGGER IF NOT EXISTS update_proposals_no_delete
BEFORE DELETE ON update_proposals
BEGIN
  SELECT RAISE(ABORT, 'update_proposals append-only');
END;

CREATE TRIGGER IF NOT EXISTS proposal_evaluations_no_update
BEFORE UPDATE ON proposal_evaluations
BEGIN
  SELECT RAISE(ABORT, 'proposal_evaluations append-only');
END;

CREATE TRIGGER IF NOT EXISTS proposal_evaluations_no_delete
BEFORE DELETE ON proposal_evaluations
BEGIN
  SELECT RAISE(ABORT, 'proposal_evaluations append-only');
END;

CREATE TRIGGER IF NOT EXISTS promotion_decisions_no_update
BEFORE UPDATE ON promotion_decisions
BEGIN
  SELECT RAISE(ABORT, 'promotion_decisions append-only');
END;

CREATE TRIGGER IF NOT EXISTS promotion_decisions_no_delete
BEFORE DELETE ON promotion_decisions
BEGIN
  SELECT RAISE(ABORT, 'promotion_decisions append-only');
END;

CREATE TRIGGER IF NOT EXISTS rollback_records_no_update
BEFORE UPDATE ON rollback_records
BEGIN
  SELECT RAISE(ABORT, 'rollback_records append-only');
END;

CREATE TRIGGER IF NOT EXISTS rollback_records_no_delete
BEFORE DELETE ON rollback_records
BEGIN
  SELECT RAISE(ABORT, 'rollback_records append-only');
END;
