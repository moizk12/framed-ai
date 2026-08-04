-- cognition_schema_v2 — run purpose and retrieval eligibility

ALTER TABLE cognitive_runs ADD COLUMN run_purpose TEXT NOT NULL DEFAULT 'migration';
ALTER TABLE cognitive_runs ADD COLUMN baseline_run_id TEXT;
ALTER TABLE cognitive_runs ADD COLUMN comparison_group_id TEXT;
ALTER TABLE cognitive_runs ADD COLUMN retrieval_eligible INTEGER NOT NULL DEFAULT 0;

ALTER TABLE retrieval_index ADD COLUMN source_run_id TEXT;
ALTER TABLE retrieval_index ADD COLUMN run_purpose TEXT;

ALTER TABLE memory_references ADD COLUMN source_run_id TEXT;
ALTER TABLE memory_references ADD COLUMN source_asset_id TEXT;
ALTER TABLE memory_references ADD COLUMN source_run_purpose TEXT;
ALTER TABLE memory_references ADD COLUMN eligibility_decision TEXT;
