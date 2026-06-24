// Parity: a session rendered to md then reverse-parsed yields the same
// turn count + role sequence + index sequence as the original turns.
use secall_core::ingest::markdown::render_session;
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
    let body = secall_core::ingest::markdown::extract_body_text(&md);

    let parsed = parse_turns_from_body(&body, "2026-06-24");

    let orig_roles: Vec<Role> = session.turns.iter().map(|t| t.role).collect();
    let parsed_roles: Vec<Role> = parsed.iter().map(|t| t.role).collect();
    let parsed_indices: Vec<u32> = parsed.iter().map(|t| t.index).collect();

    assert_eq!(parsed.len(), session.turns.len(), "turn count parity");
    assert_eq!(parsed_roles, orig_roles, "role sequence parity");
    assert_eq!(parsed_indices, vec![0, 1, 2, 3], "0-based index recovery");
}
