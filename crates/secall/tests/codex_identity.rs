//! Real ingest boundaries: every test isolates both database and vault.
use secall_core::store::{Database, VectorRepo};
use serde_json::json;
use std::{fs, path::Path, process::Command};

const A: &str = "11111111-1111-4111-8111-111111111111";
const B: &str = "22222222-2222-4222-8222-222222222222";
const PARENT: &str = "33333333-3333-4333-8333-333333333333";

fn rollout(root: &Path, id: &str, inherited: bool, messages: usize) -> std::path::PathBuf {
    let path = root.join(format!("rollout-2026-09-13T00-00-00-{id}.jsonl"));
    let mut records = vec![
        json!({"type":"session_meta","payload":{"id":id,"cwd":"/workspace/app","timestamp":"2026-09-13T00:00:00Z"}}),
    ];
    if inherited {
        records.push(json!({"type":"session_meta","payload":{"id":PARENT,"cwd":"/workspace/parent","timestamp":"2026-07-01T00:00:00Z"}}));
    }
    for n in 0..messages {
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

fn ingest(root: &Path, path: &Path) -> std::process::Output {
    let config = root.join("config.toml");
    fs::write(
        &config,
        format!("[vault]\npath = {:?}\n", root.join("vault")),
    )
    .unwrap();
    Command::new(env!("CARGO_BIN_EXE_secall"))
        .args([
            "ingest",
            path.to_str().unwrap(),
            "--no-semantic",
            "--no-embed",
        ])
        .env("SECALL_CONFIG_PATH", config)
        .env("SECALL_DB_PATH", root.join("index.sqlite"))
        .output()
        .unwrap()
}

#[test]
fn codex_rollout_identity_survives_inherited_metadata_and_ingest_order() {
    for reverse in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let parent = rollout(root, PARENT, false, 4);
        assert!(ingest(root, &parent).status.success());
        let a = rollout(root, A, true, 3);
        let b = rollout(root, B, true, 1);
        let paths = if reverse { [&b, &a] } else { [&a, &b] };
        for path in paths {
            assert!(ingest(root, path).status.success());
        }
        let db = Database::open(&root.join("index.sqlite")).unwrap();
        db.init_vector_table().unwrap();
        for (id, turns) in [(PARENT, 4), (A, 3), (B, 1)] {
            let count: i64 = db
                .conn()
                .query_row(
                    "SELECT count(*) FROM turns WHERE session_id=?1",
                    [id],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(count, turns, "wrong identity: {id}, reverse={reverse}");
        }
        db.conn()
            .execute(
                "UPDATE sessions SET is_favorite=1, notes='preserve me', is_archived=1 WHERE id=?1",
                [A],
            )
            .unwrap();
        db.conn().execute("INSERT INTO turn_vectors(session_id,turn_index,chunk_seq,model,embedded_at,embedding) VALUES (?1,0,0,'fixture','2026-09-13',X'00000000')",[A]).unwrap();
        let vault: String = db
            .conn()
            .query_row("SELECT vault_path FROM sessions WHERE id=?1", [A], |r| {
                r.get(0)
            })
            .unwrap();
        let before = fs::read(root.join("vault").join(&vault)).unwrap();
        for path in [&b, &a, &parent] {
            assert!(ingest(root, path).status.success());
        }
        assert_eq!(fs::read(root.join("vault").join(vault)).unwrap(), before);
        let metadata: (i64, String, i64) = db
            .conn()
            .query_row(
                "SELECT is_favorite,notes,is_archived FROM sessions WHERE id=?1",
                [A],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(metadata, (1, "preserve me".into(), 1));
        let vectors: i64 = db
            .conn()
            .query_row(
                "SELECT count(*) FROM turn_vectors WHERE session_id=?1",
                [A],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(vectors, 1);
    }
}

#[test]
fn codex_rollout_filename_conflict_cannot_write_another_session() {
    let tmp = tempfile::tempdir().unwrap();
    let path = rollout(tmp.path(), A, false, 1);
    let text = fs::read_to_string(&path).unwrap().replace(A, B);
    fs::write(&path, text).unwrap();
    let output = ingest(tmp.path(), &path);
    assert!(
        !output.status.success(),
        "conflicting identity must fail before writes"
    );
    let db = Database::open(&tmp.path().join("index.sqlite")).unwrap();
    let count: i64 = db
        .conn()
        .query_row("SELECT count(*) FROM sessions", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);
}
