//! Real ingest boundaries: every test isolates both database and vault.
use secall_core::store::{Database, VectorRepo};
use serde_json::json;
use std::{fs, path::Path, process::Command};

const A: &str = "11111111-1111-4111-8111-111111111111";
const B: &str = "22222222-2222-4222-8222-222222222222";
const PARENT: &str = "33333333-3333-4333-8333-333333333333";

fn rollout(root: &Path, id: &str, inherited: bool, messages: usize) -> std::path::PathBuf {
    let path = root.join(format!("rollout-2026-09-13T00-00-00-{id}.jsonl"));
    write_rollout(&path, id, inherited, messages);
    path
}

/// `thread/revert` names the replacement rollout after the thread and a new
/// rollout ID; its header still names the thread.
fn reverted_rollout(root: &Path, thread: &str, rollout_id: &str) -> std::path::PathBuf {
    let path = root.join(format!(
        "rollout-2026-09-14T00-00-00-{thread}_{rollout_id}.jsonl"
    ));
    write_rollout(&path, thread, false, 1);
    path
}

fn write_rollout(path: &Path, id: &str, inherited: bool, messages: usize) {
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
        path,
        records
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
}

fn turn_count(root: &Path, id: &str) -> i64 {
    Database::open(&root.join("index.sqlite"))
        .unwrap()
        .conn()
        .query_row(
            "SELECT count(*) FROM turns WHERE session_id=?1",
            [id],
            |r| r.get(0),
        )
        .unwrap()
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
fn reverted_codex_rollout_is_its_own_session() {
    for reverse in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let original = rollout(root, A, false, 3);
        let reverted = reverted_rollout(root, A, B);
        let paths = if reverse {
            [&reverted, &original]
        } else {
            [&original, &reverted]
        };
        for path in paths.into_iter().chain([&original, &reverted]) {
            let output = ingest(root, path);
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        assert_eq!(turn_count(root, A), 3, "reverse={reverse}");
        assert_eq!(turn_count(root, B), 1, "reverse={reverse}");
    }
}

#[test]
fn codex_rollout_filename_conflict_cannot_write_another_session() {
    let tmp = tempfile::tempdir().unwrap();
    let ordinary = rollout(tmp.path(), A, false, 1);
    let text = fs::read_to_string(&ordinary).unwrap().replace(A, B);
    fs::write(&ordinary, text).unwrap();
    // A replacement rollout must still name its thread in the header.
    let reverted = reverted_rollout(tmp.path(), PARENT, B);
    let text = fs::read_to_string(&reverted).unwrap().replace(PARENT, A);
    fs::write(&reverted, text).unwrap();
    for path in [ordinary, reverted] {
        let output = ingest(tmp.path(), &path);
        assert!(
            !output.status.success(),
            "conflicting identity must fail before writes: {}",
            path.display()
        );
    }
    let db = Database::open(&tmp.path().join("index.sqlite")).unwrap();
    let count: i64 = db
        .conn()
        .query_row("SELECT count(*) FROM sessions", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);
}

#[cfg(unix)]
#[test]
fn unchanged_codex_rollout_is_skipped_without_reading_it() {
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // Under .codex/sessions the parser is chosen by path, so nothing before the
    // duplicate check needs the file's content.
    let dir = root.join(".codex/sessions/2026/09/13");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("rollout-2026-09-13T00-00-00-{A}.jsonl"));
    let jsonl = |texts: &[String]| {
        let mut records = vec![
            json!({"type":"session_meta","payload":{"id":A,"cwd":"/workspace/app","timestamp":"2026-09-13T00:00:00Z"}}),
        ];
        for text in texts {
            records.push(json!({"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":text}]}}));
        }
        records
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    };
    let turns = || -> i64 {
        Database::open(&root.join("index.sqlite"))
            .unwrap()
            .conn()
            .query_row("SELECT count(*) FROM turns WHERE session_id=?1", [A], |r| {
                r.get(0)
            })
            .unwrap()
    };
    let set_mtime = |mtime: std::time::SystemTime| {
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(mtime)
            .unwrap();
    };
    let set_mode =
        |mode: u32| fs::set_permissions(&path, fs::Permissions::from_mode(mode)).unwrap();

    let original = jsonl(&[format!("first {}", "x".repeat(400))]);
    fs::write(&path, &original).unwrap();
    assert!(ingest(root, &path).status.success());
    assert_eq!(turns(), 1);
    let mtime = fs::metadata(&path).unwrap().modified().unwrap();

    // One more turn at the same size and mtime, in a file nobody may read: the
    // ingest succeeds only if it decides on the file's size and mtime alone.
    // (Root ignores the mode, so there this shows the skip but not the absent read.)
    let shorter = jsonl(&["first".into(), "second".into()]);
    let pad = original.len() - shorter.len() - 1;
    let grown = jsonl(&["first".into(), format!("second {}", "y".repeat(pad))]);
    assert_eq!(grown.len(), original.len());
    fs::write(&path, &grown).unwrap();
    set_mtime(mtime);
    set_mode(0o000);
    let skipped = ingest(root, &path);
    set_mode(0o644);
    assert!(
        skipped.status.success(),
        "{}",
        String::from_utf8_lossy(&skipped.stderr)
    );
    assert_eq!(turns(), 1);

    // A new mtime marks it changed again.
    set_mtime(mtime + std::time::Duration::from_secs(10));
    assert!(ingest(root, &path).status.success());
    assert_eq!(turns(), 2);
}
