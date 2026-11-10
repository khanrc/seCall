use std::path::{Path, PathBuf};

use anyhow::Result;
use secall_core::{
    ingest::markdown::with_frontmatter_session_type,
    store::{get_default_db_path, Database},
    vault::Config,
};

/// Reclassify every session with the configured rules. The type goes into the
/// frontmatter of the vault Markdown this host rendered (consumers of the vault
/// read it there), then into the database.
pub async fn run_backfill(dry_run: bool) -> Result<()> {
    let config = Config::load_or_default();
    let classification = &config.ingest.classification;

    if classification.rules.is_empty() {
        eprintln!(
            "No classification rules found in config. \
             Add [[ingest.classification.rules]] to .secall.toml"
        );
        return Ok(());
    }

    let db_path = get_default_db_path();
    let db = Database::open(&db_path)?;
    let compiled_rules = super::ingest::compile_classification_rules(&config)?;
    let local_host = gethostname::gethostname().to_string_lossy().to_string();

    let sessions = db.get_all_sessions_for_classify()?;
    let mut changed = 0usize;
    let mut rewritten = 0usize;
    let mut failed = 0usize;
    for session in &sessions {
        let new_type = super::ingest::apply_classification(
            &compiled_rules,
            &session.first_user_content,
            session.project.as_deref(),
            &classification.default,
        );
        let type_changed = new_type != session.session_type;
        let short_id = &session.id[..8.min(session.id.len())];
        // Every file is checked, not only those whose type changes: before
        // classify wrote Markdown it changed only the database, leaving files
        // with the type they were rendered with.
        //
        // The Markdown goes first: the database type decides whether a later
        // run sees a change, so a file left behind is retried only while the
        // database keeps the old type.
        if let Some(path) = own_markdown(&db, &config, &session.id, &local_host)? {
            match update_markdown_session_type(&path, &new_type, dry_run) {
                Ok(MarkdownUpdate::Rewritten) => rewritten += 1,
                Ok(MarkdownUpdate::Current) => {}
                Ok(MarkdownUpdate::NoField) if type_changed => {
                    tracing::warn!(path = %path.display(), "no session_type in frontmatter")
                }
                Ok(MarkdownUpdate::NoField) => {}
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "Markdown not updated; retried next run");
                    failed += 1;
                    continue;
                }
            }
        }
        if !type_changed {
            continue;
        }
        changed += 1;
        if dry_run {
            eprintln!(
                "  [dry-run] {short_id}: {} → {new_type}",
                session.session_type
            );
            continue;
        }
        db.update_session_type(&session.id, &new_type)?;
        tracing::debug!(session = short_id, session_type = new_type, "classified");
    }

    eprintln!(
        "Classify {}complete: {changed} of {} sessions changed type, {rewritten} Markdown files {}",
        if dry_run { "(dry-run) " } else { "" },
        sessions.len(),
        if dry_run { "to update" } else { "updated" },
    );
    if failed > 0 {
        eprintln!(
            "  {failed} Markdown files could not be updated; their sessions keep their database type until a run can update them"
        );
    }
    Ok(())
}

/// The Markdown file this host rendered for a session, if it exists. Another
/// host's Markdown is that host's to rewrite when it reclassifies, as each host
/// keeps its own classification.
fn own_markdown(
    db: &Database,
    config: &Config,
    session_id: &str,
    local_host: &str,
) -> Result<Option<PathBuf>> {
    if db.foreign_vault_path(session_id, local_host)?.is_some() {
        return Ok(None);
    }
    Ok(db
        .get_session_vault_path(session_id)?
        .filter(|p| !p.is_empty())
        .map(|p| config.vault.path.join(p))
        .filter(|path| path.is_file()))
}

enum MarkdownUpdate {
    Rewritten,
    Current,
    NoField,
}

/// Set the frontmatter `session_type` of the Markdown at `path`, writing the
/// file only when the type differs and this is not a dry run.
fn update_markdown_session_type(
    path: &Path,
    session_type: &str,
    dry_run: bool,
) -> std::io::Result<MarkdownUpdate> {
    let content = std::fs::read_to_string(path)?;
    let Some(updated) = with_frontmatter_session_type(&content, session_type) else {
        return Ok(MarkdownUpdate::NoField);
    };
    if updated == content {
        return Ok(MarkdownUpdate::Current);
    }
    if !dry_run {
        std::fs::write(path, updated)?;
    }
    Ok(MarkdownUpdate::Rewritten)
}
