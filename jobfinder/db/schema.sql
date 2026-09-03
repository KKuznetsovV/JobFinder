-- Shared datastore for the cloud ("brain") and local ("hands") components.
-- SQLite is sufficient for a single-user tool; both processes open the same file.

CREATE TABLE IF NOT EXISTS job_postings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,          -- linkedin | gotfriends | devjobs | jobinfo
    external_id     TEXT NOT NULL,          -- stable id from the source, used for dedup
    title           TEXT NOT NULL,
    company         TEXT NOT NULL,
    description     TEXT NOT NULL,
    url             TEXT NOT NULL,
    posted_date     TEXT,                   -- ISO date, may be unknown
    apply_method    TEXT NOT NULL CHECK (apply_method IN ('email', 'company_site', 'linkedin_easy_apply')),
    apply_email     TEXT,                   -- set when apply_method = 'email'
    discovered_at   TEXT NOT NULL,
    UNIQUE (source, external_id)
);

CREATE TABLE IF NOT EXISTS applications (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    job_posting_id        INTEGER NOT NULL REFERENCES job_postings(id),
    resume_variant        TEXT NOT NULL CHECK (resume_variant IN ('fullstack', 'project_manager')),
    resume_choice_reason  TEXT,
    cover_letter          TEXT,
    status                TEXT NOT NULL DEFAULT 'found' CHECK (status IN (
                              'found', 'notified', 'pending_stage_a_approval', 'approved',
                              'rejected', 'rejected_tier0', 'rejected_stage_a', 'rejected_stage_b',
                              'revision_requested', 'submitting', 'awaiting_my_click',
                              'sent', 'stuck', 'failed'
                          )),
    relevance_score       REAL,
    gmail_thread_id       TEXT,
    gmail_last_message_id TEXT,
    revision_count        INTEGER NOT NULL DEFAULT 0,
    apply_log             TEXT NOT NULL DEFAULT '[]',  -- JSON array of {ts, event, detail}
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    tier0_score           REAL,     -- local embedding-similarity score (0-1), pre-Claude
    tier0_resume_hint     TEXT CHECK (tier0_resume_hint IN ('fullstack', 'project_manager') OR tier0_resume_hint IS NULL),
    stage_a_approved_at   TEXT,     -- set when the user approves the job+resume match (Stage A)
    stage_a_revision_count INTEGER NOT NULL DEFAULT 0,
    cover_letter_requirement TEXT NOT NULL DEFAULT 'full_letter' CHECK (cover_letter_requirement IN (
                              'none', 'short_note', 'full_letter'
                          )),
    cover_letter_char_limit INTEGER  -- known char limit for short_note, when the form exposes one
);

CREATE INDEX IF NOT EXISTS idx_applications_status ON applications(status);
CREATE INDEX IF NOT EXISTS idx_applications_thread ON applications(gmail_thread_id);

-- Gmail message ids already acted on, so polling loops never double-process a reply
-- or re-import the same LinkedIn job-alert email.
CREATE TABLE IF NOT EXISTS processed_gmail_messages (
    message_id   TEXT PRIMARY KEY,
    purpose      TEXT NOT NULL,   -- job_alert | approval_reply
    processed_at TEXT NOT NULL
);

-- Labeled examples for fine-tuning the tier-1 local classifier (see
-- jobfinder/ai/tier1_classifier.py). Logged every time a genuine Claude
-- relevance/resume decision is made (source='production'), so training data
-- accumulates during normal operation even before the model exists;
-- synthetic examples generated offline (source='synthetic') are added later
-- to supplement real volume.
CREATE TABLE IF NOT EXISTS tier1_training_examples (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    job_posting_id     INTEGER REFERENCES job_postings(id),
    title              TEXT NOT NULL,
    company            TEXT NOT NULL,
    description        TEXT NOT NULL,
    relevant            INTEGER NOT NULL CHECK (relevant IN (0, 1)),
    relevance_score     REAL NOT NULL,
    relevance_reason    TEXT NOT NULL,
    resume_variant       TEXT CHECK (resume_variant IN ('fullstack', 'project_manager') OR resume_variant IS NULL),
    resume_reason        TEXT,     -- NULL when not relevant (resume was never chosen)
    source               TEXT NOT NULL DEFAULT 'production' CHECK (source IN ('production', 'synthetic')),
    created_at           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tier1_training_examples_source ON tier1_training_examples(source);

-- One row per tier1-routed posting: which path handled it, and (when
-- spot-checked against Claude) whether the two agreed. Lets real-world
-- tier1 accuracy be queried over time, not just at initial eval.
CREATE TABLE IF NOT EXISTS tier1_decisions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    job_posting_id        INTEGER REFERENCES job_postings(id),
    path                  TEXT NOT NULL CHECK (path IN ('claude', 'claude_fallback', 'tier1', 'tier1_spot_checked')),
    relevance_confidence  REAL,
    resume_confidence     REAL,
    agreement             INTEGER CHECK (agreement IN (0, 1) OR agreement IS NULL),
    created_at            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tier1_decisions_path ON tier1_decisions(path);
