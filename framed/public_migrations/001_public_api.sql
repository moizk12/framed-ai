CREATE TABLE IF NOT EXISTS public_analyses (
    analysis_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT public_analyses_id_format CHECK (analysis_id ~ '^ana-[0-9a-f]{32}$')
);

CREATE TABLE IF NOT EXISTS public_feedback (
    feedback_id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL REFERENCES public_analyses (analysis_id) ON DELETE CASCADE,
    useful BOOLEAN NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT public_feedback_id_format CHECK (feedback_id ~ '^fdb-[0-9a-f]{32}$'),
    CONSTRAINT public_feedback_comment_length CHECK (char_length(comment) <= 2000)
);

CREATE INDEX IF NOT EXISTS public_feedback_analysis_id_idx
    ON public_feedback (analysis_id, recorded_at);
