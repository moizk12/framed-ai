-- cognition_schema_v3 — integrity hardening (additive)

ALTER TABLE cognitive_runs ADD COLUMN provenance_manifest_json TEXT;
ALTER TABLE cognitive_runs ADD COLUMN failure_code TEXT;
ALTER TABLE cognitive_runs ADD COLUMN failure_stage TEXT;

ALTER TABLE episodes ADD COLUMN failure_code TEXT;
ALTER TABLE episodes ADD COLUMN failure_message TEXT;
