//! Each Codex session keeps its own vault Markdown. Every test isolates both
//! database and vault.
use secall_core::store::Database;
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

// UUIDv7 ids of sessions started within the same minute share their first 8
// characters, which earlier releases used as the file name's id.
const A: &str = "019dc309-1a2b-7c3d-8e4f-00000000000a";
const B: &str = "019dc309-9f00-7aaa-bbbb-00000000000b";
const CLAUDE: &str = "5c0d1e2f-3a4b-4c5d-8e6f-7a8b9c0d1e2f";

fn rollout(root: &Path, id: &str) -> PathBuf {
    let path = root.join(format!("rollout-2026-09-13T00-00-00-{id}.jsonl"));
    let mut records = vec![
        json!({"type":"session_meta","payload":{"id":id,"cwd":"/workspace/app","timestamp":"2026-09-13T00:00:00Z"}}),
    ];
    for n in 0..2 {
        records.push(json!({"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":format!("Evidence {id} turn {n}")}]}}));
    }
    fs::write(
        &path,
        records
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    path
}

/// Writes a session source for an id under a directory.
type WriteSource = fn(&Path, &str) -> PathBuf;

/// A Claude Code transcript, named after its session id as Claude Code names it.
fn claude_transcript(root: &Path, id: &str) -> PathBuf {
    let path = root.join(format!("{id}.jsonl"));
    let records = [
        json!({"type":"user","message":{"role":"user","content":"list the files"},"timestamp":"2026-09-13T00:00:00Z","sessionId":id,"cwd":"/workspace/app","gitBranch":"main","version":"1.0"}),
        json!({"type":"assistant","message":{"role":"assistant","id":"msg_1","model":"claude-opus-4-6","content":[{"type":"text","text":"Two files."}],"usage":{"input_tokens":5,"output_tokens":10,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}},"timestamp":"2026-09-13T00:00:01Z","sessionId":id}),
    ];
    fs::write(
        &path,
        records
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    path
}

fn ingest(root: &Path, path: &Path) {
    let config = root.join("config.toml");
    fs::write(
        &config,
        format!("[vault]\npath = {:?}\n", root.join("vault")),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_secall"))
        .args([
            "ingest",
            path.to_str().unwrap(),
            "--no-semantic",
            "--no-embed",
        ])
        .env("SECALL_CONFIG_PATH", config)
        .env("SECALL_DB_PATH", root.join("index.sqlite"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// `(file, session_id in its frontmatter)` for every session file.
fn session_files(root: &Path) -> Vec<(PathBuf, String)> {
    walkdir::WalkDir::new(root.join("vault"))
        .into_iter()
        .filter_map(|e| e.ok())
        .map(walkdir::DirEntry::into_path)
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .filter_map(|p| {
            let content = fs::read_to_string(&p).ok()?;
            let id = content
                .lines()
                .find_map(|line| line.strip_prefix("session_id: "))?
                .to_string();
            Some((p, id))
        })
        .collect()
}

fn stored_turns(root: &Path, id: &str) -> i64 {
    let db = Database::open(&root.join("index.sqlite")).unwrap();
    db.conn()
        .query_row(
            "SELECT count(*) FROM turns WHERE session_id = ?1",
            [id],
            |r| r.get(0),
        )
        .unwrap()
}

#[test]
fn codex_sessions_started_together_keep_separate_markdown() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    ingest(root, &rollout(root, A));
    ingest(root, &rollout(root, B));

    let mut files = session_files(root);
    files.sort_by(|a, b| a.1.cmp(&b.1));
    assert_eq!(files.len(), 2, "{files:?}");
    for ((path, id), expected) in files.iter().zip([A, B]) {
        assert_eq!(id, expected);
        assert!(
            path.ends_with(format!("codex_app_{expected}.md")),
            "{path:?}"
        );
    }
}

#[test]
fn ingest_restores_a_session_markdown_that_is_gone() {
    // Ingest finds a Claude Code transcript's row by its file name and a
    // Codex rollout's by parsing it; both must notice the missing file.
    let sources: [(WriteSource, &str); 2] = [(rollout, A), (claude_transcript, CLAUDE)];
    for (write_source, id) in sources {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let source = write_source(root, id);
        ingest(root, &source);
        let [(path, _)] = session_files(root).try_into().unwrap();
        fs::remove_file(&path).unwrap();

        // The source is unchanged, so there are no new turns to index.
        ingest(root, &source);
        assert_eq!(session_files(root), vec![(path, id.to_string())]);
        assert_eq!(stored_turns(root, id), 2);
    }
}
