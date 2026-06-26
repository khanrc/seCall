use rusqlite::Connection;
use secall_core::store::Database;

#[test]
fn migrate_v8_to_v9() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join("migration.sqlite");

    let conn = Connection::open(&db_path).expect("open sqlite");
    conn.execute_batch(
        "
        CREATE TABLE config (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        INSERT INTO config(key, value) VALUES ('schema_version', '8');
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY
        );
        ",
    )
    .expect("seed v8 schema");
    drop(conn);

    let db = Database::open(&db_path).expect("migrate to v9");
    let schema_version: String = db
        .conn()
        .query_row(
            "SELECT value FROM config WHERE key = 'schema_version'",
            [],
            |row| row.get(0),
        )
        .expect("schema version");
    let wiki_vectors_exists: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='wiki_vectors'",
            [],
            |row| row.get(0),
        )
        .expect("wiki_vectors exists");

    // v8 → … 마이그레이션 체인 — Database::open 후 schema_version 은 최신값
    // (CURRENT_SCHEMA_VERSION)이어야 한다. v9 에서 도입된 wiki_vectors 는 유지.
    assert_eq!(
        schema_version,
        secall_core::store::schema::CURRENT_SCHEMA_VERSION.to_string()
    );
    assert_eq!(wiki_vectors_exists, 1);
}
