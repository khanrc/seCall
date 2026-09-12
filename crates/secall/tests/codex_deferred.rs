use secall_core::store::Database;
use serde_json::{json, Value};
use std::{fs, process::Command};

const ID: &str = "44444444-4444-4444-8444-444444444444";
fn run(root: &std::path::Path, path: &std::path::Path) -> (bool, Value) {
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
            "--no-embed",
            "--no-semantic",
            "--format",
            "json",
        ])
        .env("SECALL_CONFIG_PATH", config)
        .env("SECALL_DB_PATH", root.join("index.sqlite"))
        .output()
        .unwrap();
    let result = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|_| panic!("{}", String::from_utf8_lossy(&output.stderr)));
    (output.status.success(), result)
}
fn review() -> String {
    [json!({"type":"session_meta","payload":{"id":ID,"cwd":"/workspace/app","source":"exec"}}),
     json!({"type":"event_msg","payload":{"type":"entered_review_mode"}}),
     json!({"type":"event_msg","payload":{"type":"user_message","message":"Review this change"}}),
     json!({"type":"event_msg","payload":{"type":"mcp_tool_call_end","result":"retained tool evidence"}})]
        .iter().map(ToString::to_string).collect::<Vec<_>>().join("\n")
}
#[test]
fn codex_deferred_review_rechecks_changes_and_parser_revision_across_processes() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let path = root.join(format!("rollout-{ID}.jsonl"));
    let original = review();
    fs::write(&path, &original).unwrap();
    for _ in 0..2 {
        let (ok, result) = run(root, &path);
        assert!(ok, "{result}");
        assert_eq!(result["summary"]["deferred"], 1);
        assert_eq!(result["summary"]["errors"], 0);
    }
    let db = Database::open(&root.join("index.sqlite")).unwrap();
    let count: i64 = db
        .conn()
        .query_row("SELECT count(*) FROM sessions", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);
    // A cached pass must not overwrite the receipt (which a fresh parse does).
    db.conn()
        .execute("UPDATE deferred_sources SET reason='cached-sentinel'", [])
        .unwrap();
    assert!(run(root, &path).0);
    let reason: String = db
        .conn()
        .query_row("SELECT reason FROM deferred_sources", [], |r| r.get(0))
        .unwrap();
    assert_eq!(reason, "cached-sentinel");
    let revision: String = db
        .conn()
        .query_row("SELECT parser_revision FROM deferred_sources", [], |r| {
            r.get(0)
        })
        .unwrap();
    db.conn()
        .execute(
            "UPDATE deferred_sources SET parser_revision='old-parser'",
            [],
        )
        .unwrap();
    assert!(run(root, &path).0);
    let revised: String = db
        .conn()
        .query_row("SELECT parser_revision FROM deferred_sources", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(revised, revision);
    // Same-size rewrite must recheck; mtime/size alone are insufficient.
    fs::write(&path, original.replace("user_message", "agent_messag")).unwrap();
    let (ok, result) = run(root, &path);
    assert!(!ok);
    assert_eq!(result["summary"]["errors"], 1);
    let message = json!({"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Review this change"}]}});
    fs::write(&path, format!("{original}\n{message}\n")).unwrap();
    let (ok, result) = run(root, &path);
    assert!(ok, "{result}");
    assert_eq!(result["summary"]["ingested"], 1);
    assert_eq!(result["summary"]["deferred"], 0);
    let turns: i64 = db
        .conn()
        .query_row("SELECT count(*) FROM turns", [], |r| r.get(0))
        .unwrap();
    assert_eq!(turns, 1, "event and response must not duplicate turns");
    let receipts: i64 = db
        .conn()
        .query_row("SELECT count(*) FROM deferred_sources", [], |r| r.get(0))
        .unwrap();
    assert_eq!(receipts, 0);
    assert!(fs::read_to_string(path)
        .unwrap()
        .contains("retained tool evidence"));
}

#[test]
fn codex_deferred_does_not_hide_malformed_or_unknown_input() {
    for suffix in [
        "{broken",
        r#"{"type":"event_msg","payload":{"type":"agent_message","message":"Real answer"}}"#,
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(format!("rollout-{ID}.jsonl"));
        fs::write(&path, format!("{}\n{suffix}\n", review())).unwrap();
        let (ok, result) = run(tmp.path(), &path);
        assert!(!ok);
        assert_eq!(result["summary"]["errors"], 1);
    }
}

#[test]
fn migration_adds_deferred_sources_to_existing_v13_index() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("index.sqlite");
    let db = Database::open(&path).unwrap();
    db.conn().execute_batch("DROP TABLE deferred_sources; UPDATE config SET value='13' WHERE key='schema_version'; INSERT INTO config(key,value) VALUES ('preserved','value');").unwrap();
    drop(db);
    let db = Database::open(&path).unwrap();
    let preserved: String = db
        .conn()
        .query_row("SELECT value FROM config WHERE key='preserved'", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(preserved, "value");
    let count: i64 = db
        .conn()
        .query_row("SELECT count(*) FROM deferred_sources", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);
}
