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
    stage_a_revision_count INTEGER NOT NULL DEFAULT 0
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
