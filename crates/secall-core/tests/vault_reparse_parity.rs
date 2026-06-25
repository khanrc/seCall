// Parity: a session rendered to md then reverse-parsed yields the same
// turn count + role sequence + index sequence as the original turns.
//
// Also covers the re-entrant backfill path introduced in #1021:
// a session row that already exists in the DB but has zero turns gets
// its turns inserted idempotently (INSERT OR IGNORE on UNIQUE(session_id,
// turn_index)) when the reindex loop re-visits it.
use secall_core::ingest::markdown::{render_session, extract_body_text, parse_session_frontmatter};
use secall_core::ingest::parse_turns_from_body;
use secall_core::ingest::types::{AgentKind, Role, Session, TokenUsage, Turn};
use chrono::Utc;

fn turn(index: u32, role: Role, content: &str) -> Turn {
    Turn {
        index,
        role,
        timestamp: None,
        content: content.to_string(),
        actions: Vec::new(),
        tokens: None,
        thinking: None,
        is_sidechain: false,
    }
}

fn sample_session() -> Session {
    Session {
        id: "sess-1".into(),
        agent: AgentKind::ClaudeCode,
        model: Some("claude-opus-4-6".into()),
        project: Some("log".into()),
        cwd: None,
        git_branch: None,
        host: Some("macbook".into()),
        start_time: Utc::now(),
        end_time: None,
        turns: vec![
            turn(0, Role::User, "Question one"),
            turn(1, Role::Assistant, "Answer one"),
            turn(2, Role::Assistant, "Still answering"), // consecutive → ### in md
            turn(3, Role::User, "Question two"),
        ],
        total_tokens: TokenUsage::default(),
        session_type: "interactive".into(),
        archived: false,
        archived_at: None,
    }
}

#[test]
fn test_md_reparse_matches_original_turn_sequence() {
    let session = sample_session();
    // render_session takes a timezone argument (chrono_tz::Tz)
    let md = render_session(&session, chrono_tz::UTC);
    // strip frontmatter the same way reindex_vault does:
    let body = extract_body_text(&md);

    let parsed = parse_turns_from_body(&body, "2026-06-24");

    let orig_roles: Vec<Role> = session.turns.iter().map(|t| t.role).collect();
    let parsed_roles: Vec<Role> = parsed.iter().map(|t| t.role).collect();
    let parsed_indices: Vec<u32> = parsed.iter().map(|t| t.index).collect();

    assert_eq!(parsed.len(), session.turns.len(), "turn count parity");
    assert_eq!(parsed_roles, orig_roles, "role sequence parity");
    assert_eq!(parsed_indices, vec![0, 1, 2, 3], "0-based index recovery");
}

/// Backfill re-entry: simulate a session that was indexed before the #1021 fix
/// (sessions row present, zero turns). Verify that running the turns-insertion
/// loop again (as reindex_vault now does for sessions with incomplete turns)
/// produces the full turn set, and that running it a second time is idempotent
/// (INSERT OR IGNORE on UNIQUE(session_id, turn_index) must not duplicate rows).
#[test]
fn test_reindex_backfill_reentry_idempotent() {
    use secall_core::store::{Database, SessionRepo};

    let session = sample_session();
    let md = render_session(&session, chrono_tz::UTC);
    let body = extract_body_text(&md);

    // Parse frontmatter so we can call insert_session_from_vault.
    let fm = parse_session_frontmatter(&md).expect("frontmatter parse");

    let db = Database::open_memory().expect("in-memory db");

    // First pass: insert the session row (no turns — pre-fix state).
    db.insert_session_from_vault(&fm, &body, "raw/sessions/sess-1.md")
        .expect("insert_session_from_vault");
    assert_eq!(
        db.count_turns_for_session(&session.id).unwrap(),
        0,
        "pre-fix state: sessions row present but zero turns"
    );

    // Second pass: reindex_vault now runs the turns-insertion loop even when
    // the session already exists (because db_turn_count < expected_turns).
    // Simulate that loop here.
    let parsed_turns = parse_turns_from_body(&body, "2026-06-24");
    for turn in &parsed_turns {
        db.insert_turn(&session.id, turn)
            .expect("insert_turn first pass");
    }
    assert_eq!(
        db.count_turns_for_session(&session.id).unwrap(),
        session.turns.len(),
        "after backfill: full turn set present"
    );

    // Third pass: running the same loop again must be idempotent (no duplicates).
    for turn in &parsed_turns {
        db.insert_turn(&session.id, turn)
            .expect("insert_turn second pass (idempotent)");
    }
    assert_eq!(
        db.count_turns_for_session(&session.id).unwrap(),
        session.turns.len(),
        "idempotency: re-inserting already-present turns must not duplicate rows"
    );
}
