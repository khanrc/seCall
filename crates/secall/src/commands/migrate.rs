use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::Result;
use secall_core::{
    ingest::markdown::{
        extract_summary_from_body, full_id_filename, parse_session_frontmatter, SessionFrontmatter,
    },
    store::{get_default_db_path, Database},
    vault::Config,
};
use serde::Serialize;

use crate::output::OutputFormat;

pub fn run_summary(dry_run: bool) -> Result<()> {
    let config = Config::load_or_default();
    let db = Database::open(&get_default_db_path())?;

    let sessions_dir = config
        .vault
        .path
        .join(secall_core::vault::sessions_reldir());
    if !sessions_dir.exists() {
        println!("No vault sessions directory found.");
        return Ok(());
    }

    let mut updated = 0usize;
    let mut skipped_has_summary = 0usize;
    let mut skipped_no_user_turn = 0usize;
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

        if fm.summary.is_some() {
            skipped_has_summary += 1;
            continue;
        }

        let summary = match extract_summary_from_body(&content) {
            Some(s) => s,
            None => {
                skipped_no_user_turn += 1;
                continue;
            }
        };

        if dry_run {
            println!("[dry-run] {} → \"{}\"", path.display(), summary);
            updated += 1;
            continue;
        }

        // MD 파일 frontmatter에 summary 삽입
        match insert_summary_into_frontmatter(&content, &summary) {
            Ok(new_content) => {
                // atomic write: 임시 파일 후 rename
                let tmp_path = path.with_extension("md.tmp");
                if let Err(e) = std::fs::write(&tmp_path, &new_content) {
                    tracing::warn!(path = %path.display(), error = %e, "failed to write tmp");
                    errors += 1;
                    continue;
                }
                if let Err(e) = std::fs::rename(&tmp_path, path) {
                    tracing::warn!(path = %path.display(), error = %e, "failed to rename");
                    let _ = std::fs::remove_file(&tmp_path);
                    errors += 1;
                    continue;
                }
            }
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "frontmatter insert failed");
                errors += 1;
                continue;
            }
        }

        // DB 업데이트
        if !fm.session_id.is_empty() {
            if let Err(e) = db.update_session_summary(&fm.session_id, &summary) {
                tracing::warn!(session_id = %fm.session_id, error = %e, "DB update failed");
            }
        }

        eprintln!("  updated: {}", path.display());
        updated += 1;
    }

    eprintln!(
        "\nMigrate summary: {} updated, {} skipped (already has summary), {} skipped (no user turn), {} errors",
        updated, skipped_has_summary, skipped_no_user_turn, errors
    );
    Ok(())
}

/// `secall migrate filenames`: rename vault session files from the first 8
/// characters of the session id to the whole id, then point index rows at the
/// renamed files.
pub fn run_filenames(dry_run: bool, format: &OutputFormat) -> Result<()> {
    let config = Config::load_or_default();
    let db = Database::open(&get_default_db_path())?;
    let report = migrate_filenames(&db, &config.vault.path, dry_run)?;
    match format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&report)?),
        OutputFormat::Text => {
            let prefix = if dry_run { "[dry-run] " } else { "" };
            eprintln!(
                "{prefix}Renamed {} files; {} more were already rendered under the whole id.",
                report.renamed.len(),
                report.merged.len()
            );
            eprintln!(
                "{prefix}Pointed {} index rows at renamed files.",
                report.repointed
            );
            if !report.missing.is_empty() {
                eprintln!(
                    "{prefix}{} sessions have no Markdown; ingest renders them again on the host \
                     that holds their source:",
                    report.missing.len()
                );
                for id in &report.missing {
                    eprintln!("  {id}");
                }
            }
            if report.unreadable > 0 {
                eprintln!(
                    "{prefix}Left {} files without session frontmatter alone.",
                    report.unreadable
                );
            }
        }
    }
    Ok(())
}

/// What [`migrate_filenames`] did, with vault-relative paths.
#[derive(Debug, Default, Serialize)]
pub struct FilenameMigration {
    /// Files renamed to the whole-id name.
    pub renamed: Vec<Rename>,
    /// Files whose session a newer release had already rendered under the
    /// whole-id name in the same directory. Sessions only grow, so the render
    /// with more turns stays, under the whole-id name.
    pub merged: Vec<Rename>,
    /// Index rows pointed from a renamed file's old path to its new one.
    pub repointed: usize,
    /// Sessions whose Markdown is in no file: another session given the same
    /// name overwrote it.
    pub missing: Vec<String>,
    /// Markdown files without session frontmatter, left alone.
    pub unreadable: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Rename {
    pub from: String,
    pub to: String,
}

fn migrate_filenames(db: &Database, vault_root: &Path, dry_run: bool) -> Result<FilenameMigration> {
    let mut report = FilenameMigration::default();
    // Paths the renames remove and create, so a dry run repoints as a real run would.
    let mut gone = HashSet::new();
    let mut created = HashSet::new();

    let sessions_dir = vault_root.join(secall_core::vault::sessions_reldir());
    // Every file and its turn count, read before anything is renamed: a
    // directory walk can revisit renamed entries, and a merge decides from the
    // files as they were, so a dry run decides as a real run would.
    let files: Vec<(PathBuf, Option<SessionFrontmatter>)> = walkdir::WalkDir::new(&sessions_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .map(walkdir::DirEntry::into_path)
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "md"))
        .map(|p| {
            let fm = read_frontmatter(&p);
            (p, fm)
        })
        .collect();
    let turns: HashMap<PathBuf, Option<u32>> = files
        .iter()
        .map(|(p, fm)| (p.clone(), fm.as_ref().map(|fm| fm.turns.unwrap_or(0))))
        .collect();
    for (path, fm) in files {
        let Some(fm) = fm else {
            report.unreadable += 1;
            continue;
        };
        let Some(new_name) = path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|name| full_id_filename(name, &fm.session_id))
        else {
            continue;
        };
        let target = path.with_file_name(new_name);
        let rename = Rename {
            from: relative(vault_root, &path),
            to: relative(vault_root, &target),
        };
        gone.insert(rename.from.clone());
        created.insert(rename.to.clone());
        if let Some(&newer_turns) = turns.get(&target) {
            let keep_old = newer_turns.map_or(true, |newer| fm.turns.unwrap_or(0) > newer);
            if !dry_run {
                if keep_old {
                    std::fs::rename(&path, &target)?;
                } else {
                    std::fs::remove_file(&path)?;
                }
            }
            report.merged.push(rename);
        } else {
            if !dry_run {
                std::fs::rename(&path, &target)?;
            }
            report.renamed.push(rename);
        }
    }

    let exists = |rel: &str| {
        created.contains(rel) || (!gone.contains(rel) && vault_root.join(rel).is_file())
    };
    let mut repoint = Vec::new();
    for (session_id, vault_path) in db.list_session_vault_paths()? {
        let Some(vault_path) = vault_path.filter(|p| !p.is_empty()) else {
            continue;
        };
        if exists(&vault_path) {
            continue;
        }
        let old = Path::new(&vault_path);
        let renamed = old
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|name| full_id_filename(name, &session_id))
            .map(|name| old.with_file_name(name).to_string_lossy().into_owned())
            .filter(|path| exists(path));
        match renamed {
            Some(path) => repoint.push((session_id, path)),
            None => report.missing.push(session_id),
        }
    }
    report.repointed = repoint.len();
    if !dry_run {
        db.with_transaction(|| {
            for (session_id, path) in &repoint {
                db.move_session_vault_path(session_id, path)?;
            }
            Ok(())
        })?;
    }
    Ok(report)
}

fn read_frontmatter(path: &Path) -> Option<SessionFrontmatter> {
    let content = std::fs::read_to_string(path).ok()?;
    parse_session_frontmatter(&content)
        .ok()
        .filter(|fm| !fm.session_id.is_empty())
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

/// frontmatter에서 `status:` 라인 직전에 `summary: "..."` 삽입.
/// status: 라인이 없으면 `---` 종결자 직전에 삽입.
fn insert_summary_into_frontmatter(content: &str, summary: &str) -> Result<String> {
    let after_open = content
        .strip_prefix("---\n")
        .ok_or_else(|| anyhow::anyhow!("no opening --- found"))?;

    let (fm_str, rest) = after_open
        .split_once("\n---\n")
        .ok_or_else(|| anyhow::anyhow!("no closing --- found"))?;

    let escaped = fm_str
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        // summary 자체의 이스케이프는 이미 extract_summary_from_body에서 안된 상태 — 여기서 직접 escape
        ;
    // summary 값 이스케이프 (실제 summary 문자열에 적용)
    let escaped_summary = summary.replace('\\', "\\\\").replace('"', "\\\"");
    let summary_line = format!("summary: \"{escaped_summary}\"\n");

    let new_fm = if let Some(pos) = fm_str.find("\nstatus:") {
        // `\nstatus:` 앞에 삽입 → `\nsummary: "..."\nstatus:`
        let (before, after) = fm_str.split_at(pos);
        format!("{before}\n{summary_line}{}", &after[1..]) // after[1..] to skip the leading \n
    } else {
        // status: 없으면 끝에 추가 — 마지막 필드와 같은 줄이 되지 않도록 개행 보장
        if fm_str.ends_with('\n') {
            format!("{fm_str}{summary_line}")
        } else {
            format!("{fm_str}\n{summary_line}")
        }
    };

    let _ = escaped; // suppress unused warning
    Ok(format!("---\n{new_fm}\n---\n{rest}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use secall_core::ingest::{AgentKind, Session, TokenUsage};
    use secall_core::store::SessionRepo;

    const DAY: &str = "raw/.sessions/2026-11-01";
    // A and B are Codex sessions started within a minute: one old file name.
    const A: &str = "019dc309-0000-7000-8000-00000000000a";
    const B: &str = "019dc309-0000-7000-8000-00000000000b";
    const C: &str = "c0ffee00-1111-4111-8111-111111111111";
    const D: &str = "d00d0000-3333-4333-8333-333333333333";
    const E: &str = "e0e0e0e0-2222-4222-8222-222222222222";
    const F: &str = "f00f0000-4444-4444-8444-444444444444";
    const G: &str = "9a9a0000-5555-4555-8555-555555555555";

    fn write_md(vault: &Path, name: &str, session_id: &str, turns: u32) {
        let path = vault.join(DAY).join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            path,
            format!("---\nsession_id: {session_id}\nturns: {turns}\n---\nbody\n"),
        )
        .unwrap();
    }

    fn turns_in(vault: &Path, name: &str) -> Option<u32> {
        read_frontmatter(&vault.join(DAY).join(name)).and_then(|fm| fm.turns)
    }

    fn add_row(db: &Database, id: &str, vault_name: &str) {
        db.insert_session(&Session {
            id: id.to_string(),
            agent: AgentKind::Codex,
            model: None,
            project: Some("log".to_string()),
            cwd: None,
            git_branch: None,
            host: None,
            start_time: chrono::Utc::now(),
            end_time: None,
            turns: vec![],
            total_tokens: TokenUsage::default(),
            session_type: "interactive".to_string(),
            archived: false,
            archived_at: None,
        })
        .unwrap();
        db.update_session_vault_path(id, &format!("{DAY}/{vault_name}"))
            .unwrap();
    }

    fn row_path(db: &Database, id: &str) -> String {
        db.get_session_vault_path(id).unwrap().unwrap()
    }

    #[test]
    fn filenames_migration_renames_merges_and_repoints() {
        let dir = tempfile::tempdir().unwrap();
        let vault = dir.path();
        let db = Database::open_memory().unwrap();

        // A's file overwrote B's; both rows name it.
        write_md(vault, "codex_log_019dc309.md", A, 3);
        add_row(&db, A, "codex_log_019dc309.md");
        add_row(&db, B, "codex_log_019dc309.md");
        // C and E each have an old file next to a newer release's whole-id
        // render; the render with more turns must survive.
        write_md(vault, "codex_log_c0ffee00.md", C, 2);
        write_md(vault, &format!("codex_log_{C}.md"), C, 5);
        add_row(&db, C, "codex_log_c0ffee00.md");
        write_md(vault, "codex_log_e0e0e0e0.md", E, 9);
        write_md(vault, &format!("codex_log_{E}.md"), E, 4);
        add_row(&db, E, &format!("codex_log_{E}.md"));
        // G's old file sits next to a whole-id file without frontmatter.
        write_md(vault, "codex_log_9a9a0000.md", G, 3);
        std::fs::write(vault.join(DAY).join(format!("codex_log_{G}.md")), "torn\n").unwrap();
        // Another host's session, followed by `secall sync`.
        write_md(vault, "codex_log_f00f0000.md", F, 2);
        add_row(&db, F, "codex_log_f00f0000.md");
        db.conn()
            .execute(
                "UPDATE sessions SET status = 'reindexed', host = 'macbook' WHERE id = ?1",
                [F],
            )
            .unwrap();
        // Already migrated, and a file that is not a session.
        write_md(vault, &format!("codex_log_{D}.md"), D, 1);
        add_row(&db, D, &format!("codex_log_{D}.md"));
        std::fs::write(vault.join(DAY).join("notes.md"), "no frontmatter\n").unwrap();

        let check = |report: &FilenameMigration| {
            let rename = |from: &str, id: &str| Rename {
                from: format!("{DAY}/{from}"),
                to: format!("{DAY}/codex_log_{id}.md"),
            };
            let sorted = |renames: &[Rename]| {
                let mut renames = renames.to_vec();
                renames.sort_by(|a, b| a.from.cmp(&b.from));
                renames
            };
            assert_eq!(
                sorted(&report.renamed),
                vec![
                    rename("codex_log_019dc309.md", A),
                    rename("codex_log_f00f0000.md", F)
                ]
            );
            assert_eq!(
                sorted(&report.merged),
                vec![
                    rename("codex_log_9a9a0000.md", G),
                    rename("codex_log_c0ffee00.md", C),
                    rename("codex_log_e0e0e0e0.md", E)
                ]
            );
            assert_eq!(report.repointed, 3);
            assert_eq!(report.missing, vec![B.to_string()]);
            assert_eq!(report.unreadable, 2);
        };

        let planned = migrate_filenames(&db, vault, true).unwrap();
        check(&planned);
        assert_eq!(
            turns_in(vault, "codex_log_019dc309.md"),
            Some(3),
            "dry run renames nothing"
        );
        assert_eq!(row_path(&db, A), format!("{DAY}/codex_log_019dc309.md"));

        let done = migrate_filenames(&db, vault, false).unwrap();
        check(&done);
        assert_eq!(turns_in(vault, &format!("codex_log_{A}.md")), Some(3));
        assert_eq!(turns_in(vault, &format!("codex_log_{C}.md")), Some(5));
        assert_eq!(turns_in(vault, &format!("codex_log_{E}.md")), Some(9));
        assert_eq!(turns_in(vault, &format!("codex_log_{G}.md")), Some(3));
        for old in [
            "codex_log_019dc309.md",
            "codex_log_c0ffee00.md",
            "codex_log_e0e0e0e0.md",
            "codex_log_9a9a0000.md",
            "codex_log_f00f0000.md",
        ] {
            assert!(!vault.join(DAY).join(old).exists(), "{old} is gone");
        }
        for id in [A, C, D, E, F] {
            assert_eq!(row_path(&db, id), format!("{DAY}/codex_log_{id}.md"));
        }
        assert_eq!(
            db.foreign_vault_path(F, "this-host").unwrap(),
            Some(format!("{DAY}/codex_log_{F}.md")),
            "sync still follows the other host's session"
        );
        assert_eq!(row_path(&db, B), format!("{DAY}/codex_log_019dc309.md"));

        let again = migrate_filenames(&db, vault, false).unwrap();
        assert!(again.renamed.is_empty() && again.merged.is_empty());
        assert_eq!(again.repointed, 0);
        assert_eq!(again.missing, vec![B.to_string()]);
    }

    #[test]
    fn test_insert_summary_before_status() {
        let content = "---\nagent: claude-code\nstatus: raw\n---\nbody\n";
        let result = insert_summary_into_frontmatter(content, "test summary").unwrap();
        assert!(result.contains("summary: \"test summary\"\nstatus: raw"));
    }

    #[test]
    fn test_insert_summary_escape() {
        let content = "---\nagent: claude-code\nstatus: raw\n---\nbody\n";
        let result = insert_summary_into_frontmatter(content, r#"say "hi""#).unwrap();
        assert!(result.contains(r#"summary: "say \"hi\"""#));
    }

    #[test]
    fn test_insert_summary_no_status_newline() {
        // status: 라인이 없는 frontmatter — 마지막 필드와 summary가 같은 줄에 합쳐지면 안 됨
        let content = "---\nagent: claude-code\nhost: myhost\n---\nbody\n";
        let result = insert_summary_into_frontmatter(content, "hello").unwrap();
        // "host: myhost\nsummary: ..." 형태여야 함
        assert!(
            result.contains("host: myhost\nsummary: \"hello\""),
            "summary must be on its own line: {result}"
        );
    }
}
