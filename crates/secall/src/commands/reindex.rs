use anyhow::Result;
use secall_core::{
    ingest::markdown::{extract_body_text, parse_session_frontmatter},
    store::{get_default_db_path, Database, SessionRepo},
    vault::Config,
};

pub fn run(from_vault: bool) -> Result<()> {
    if !from_vault {
        anyhow::bail!("--from-vault flag is required");
    }

    let config = Config::load_or_default();
    let db = Database::open(&get_default_db_path())?;

    let sessions_dir = config.vault.path.join(secall_core::vault::sessions_reldir());
    if !sessions_dir.exists() {
        println!("No vault sessions directory found.");
        return Ok(());
    }

    let mut indexed = 0usize;
    let mut skipped = 0usize;
    let mut errors = 0usize;

    for entry in walkdir::WalkDir::new(&sessions_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "md").unwrap_or(false))
    {
        let path = entry.path();
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "failed to read");
                errors += 1;
                continue;
            }
        };

        let fm = match parse_session_frontmatter(&content) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "failed to parse frontmatter");
                errors += 1;
                continue;
            }
        };

        if fm.session_id.is_empty() {
            tracing::warn!(path = %path.display(), "frontmatter missing session_id");
            errors += 1;
            continue;
        }

        let session_already_exists = match db.session_exists(&fm.session_id) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "DB check failed");
                errors += 1;
                continue;
            }
        };

        let vault_path = path
            .strip_prefix(&config.vault.path)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        let body = extract_body_text(&content);

        if session_already_exists {
            // Session row exists. Check whether turns are missing or incomplete
            // so we can backfill them idempotently. `insert_turn` uses
            // INSERT OR IGNORE and `turns` has UNIQUE(session_id, turn_index),
            // so re-running the loop for already-present turns is safe.
            let db_turn_count = match db.count_turns_for_session(&fm.session_id) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(session_id = %fm.session_id, error = %e, "count_turns_for_session failed");
                    skipped += 1;
                    continue;
                }
            };
            let expected_turns = fm.turns.unwrap_or(0) as usize;
            if db_turn_count >= expected_turns && expected_turns > 0 {
                // Turns are already complete — fast path.
                skipped += 1;
                continue;
            }
            // Falls through to the turns-insertion loop below (no session insert needed).
        } else {
            match db.insert_session_from_vault(&fm, &body, &vault_path) {
                Ok(()) => indexed += 1,
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "reindex failed");
                    errors += 1;
                    continue;
                }
            }
        }

        // #1021: vault-pulled md only got `sessions` + `turns_fts` before this.
        // Reverse-parse the body into turns so the session gains `turns` rows;
        // the hourly `secall embed` pass then backfills `turn_vectors`.
        // No inline embedding here.
        //
        // Re-entry is safe: INSERT OR IGNORE on UNIQUE(session_id, turn_index)
        // skips already-present turns, so a previously interrupted reindex or a
        // session backfilled from a pre-fix index will self-heal here.
        let parsed_turns =
            secall_core::ingest::parse_turns_from_body(&body, &fm.date);
        // Cross-check against frontmatter `turns:` count; log on mismatch but
        // still insert what parsed (best-effort, never block the reindex).
        if let Some(expected) = fm.turns {
            if parsed_turns.len() as u32 != expected {
                tracing::warn!(
                    session_id = %fm.session_id,
                    expected, parsed = parsed_turns.len(),
                    "vault reparse turn-count mismatch"
                );
            }
        }
        for turn in &parsed_turns {
            if let Err(e) = db.insert_turn(&fm.session_id, turn) {
                // Warn and continue — re-entry heals partial state on the next
                // reindex run, so a single failed insert does not corrupt the session.
                tracing::warn!(
                    session_id = %fm.session_id,
                    error = %e,
                    "insert_turn failed"
                );
            }
        }
    }

    eprintln!(
        "\nReindex: {} indexed, {} skipped (duplicate), {} errors",
        indexed, skipped, errors
    );
    Ok(())
}
