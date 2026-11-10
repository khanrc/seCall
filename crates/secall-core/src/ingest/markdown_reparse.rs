//! Reverse parser: session markdown body → `Vec<Turn>`.
//!
//! The render side (`markdown.rs`) emits a self-describing format:
//!   `## Turn N — Role (HH:MM)`  for a role change,
//!   `### Turn N (HH:MM)`        for a consecutive same-role turn.
//! Header N is 1-based; stored `turn.index` is 0-based (we subtract 1).
//!
//! Used by `reindex_vault` so vault-pulled sessions gain `turns` rows
//! (and therefore vector embeddings via the `secall embed` backfill).

use crate::ingest::markdown::THINKING_CALLOUT;
use crate::ingest::types::{Role, Turn};
use chrono::{DateTime, NaiveDate, TimeZone, Utc};

/// Revision of the turns [`parse_turns_from_body`] reads from Markdown. Bump it
/// when a change alters the turns read from existing Markdown: `secall sync`
/// then refreshes every other host's session from its Markdown once.
///
/// 1: thinking callouts leave turn content.
pub const REPARSE_REVISION: u32 = 1;

fn role_from_str(s: &str) -> Option<Role> {
    match s {
        "User" => Some(Role::User),
        "Assistant" => Some(Role::Assistant),
        "System" => Some(Role::System),
        _ => None,
    }
}

/// Parse a `## Turn N — Role (HH:MM)` or `### Turn N (HH:MM)` header line.
/// Returns `(index_1based, Some(role)|None_for_h3, Option<HH:MM>)` on a
/// strict match, else `None`. The strict anchor (role enum for `##`,
/// optional `(HH:MM)`) is what guards against literal header text in content.
fn parse_header(line: &str) -> Option<(u32, Option<Role>, Option<String>)> {
    let rest = line
        .strip_prefix("## ")
        .map(|r| (r, true))
        .or_else(|| line.strip_prefix("### ").map(|r| (r, false)))?;
    let (body, is_h2) = rest;
    let after = body.strip_prefix("Turn ")?;

    // time suffix "(HH:MM)" — optional
    let (head, time) = match after.rsplit_once(" (") {
        Some((h, t)) if t.ends_with(')') && t[..t.len() - 1].contains(':') => {
            (h, Some(t[..t.len() - 1].to_string()))
        }
        _ => (after, None),
    };

    if is_h2 {
        // "N — Role"
        let (num_s, role_s) = head.split_once(" — ")?;
        let num: u32 = num_s.trim().parse().ok()?;
        let role = role_from_str(role_s.trim())?; // strict: must be a known Role
        Some((num, Some(role), time))
    } else {
        // "N" (role omitted; inherits previous)
        let num: u32 = head.trim().parse().ok()?;
        Some((num, None, time))
    }
}

fn build_timestamp(date: &str, hhmm: &Option<String>) -> Option<DateTime<Utc>> {
    let hhmm = hhmm.as_ref()?;
    let d = NaiveDate::parse_from_str(date, "%Y-%m-%d").ok()?;
    let (h, m) = hhmm.split_once(':')?;
    let nt = d.and_hms_opt(h.trim().parse().ok()?, m.trim().parse().ok()?, 0)?;
    Some(Utc.from_utc_datetime(&nt))
}

/// A turn's lines as `(thinking, content)`. The renderer puts the thinking
/// callout first; it is not content, which ingest indexes and embeds (#1585).
fn split_thinking(lines: &[String]) -> (Option<String>, String) {
    let first = lines.iter().position(|line| !line.trim().is_empty());
    let Some(callout) = first.filter(|&i| lines[i] == THINKING_CALLOUT) else {
        return (None, lines.join("\n").trim().to_string());
    };
    // The callout runs through its `>` lines; a blank line closes it.
    let body = &lines[callout + 1..];
    let end = body
        .iter()
        .position(|line| !line.starts_with('>'))
        .unwrap_or(body.len());
    let thinking = body[..end]
        .iter()
        .map(|line| line.strip_prefix("> ").unwrap_or(&line[1..]))
        .collect::<Vec<_>>()
        .join("\n");
    let content = body[end..].join("\n").trim().to_string();
    (Some(thinking), content)
}

/// Parse the markdown body (everything after frontmatter) into turns.
/// `date` is the frontmatter `date` (YYYY-MM-DD); pass "" to skip timestamps.
pub fn parse_turns_from_body(body: &str, date: &str) -> Vec<Turn> {
    let mut turns: Vec<Turn> = Vec::new();
    let mut last_role: Option<Role> = None;
    let mut cur: Option<(u32, Role, Option<String>, Vec<String>)> = None;

    let flush = |turns: &mut Vec<Turn>,
                 cur: Option<(u32, Role, Option<String>, Vec<String>)>| {
        if let Some((num, role, time, lines)) = cur {
            let (thinking, content) = split_thinking(&lines);
            turns.push(Turn {
                index: num.saturating_sub(1),
                role,
                timestamp: build_timestamp(date, &time),
                content,
                actions: Vec::new(),
                tokens: None,
                thinking,
                is_sidechain: false,
            });
        }
    };

    for line in body.lines() {
        if let Some((num, role_opt, time)) = parse_header(line) {
            // header → close previous turn, open new
            let role = match role_opt {
                Some(r) => r,
                None => last_role.unwrap_or(Role::Assistant), // h3 inherits
            };
            let prev = cur.take();
            flush(&mut turns, prev);
            last_role = Some(role);
            cur = Some((num, role, time, Vec::new()));
        } else if let Some((_, _, _, lines)) = cur.as_mut() {
            lines.push(line.to_string());
        }
        // lines before the first header (the "# session title" + blockquote
        // meta line) are dropped — they are not turn content.
    }
    flush(&mut turns, cur);
    turns
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::types::Role;

    const SAMPLE: &str = "\
## Turn 1 — User (05:30)

How do I do X?

## Turn 2 — Assistant (05:31)

> [!thinking]- Thinking
> Let me consider.

Here is how.

> [!tool]- Bash
> ```
> ls
> ```

### Turn 3 (05:32)

Continued assistant turn.
";

    #[test]
    fn test_parse_turns_recovers_index_role_and_count() {
        let turns = parse_turns_from_body(SAMPLE, "2026-06-24");
        assert_eq!(turns.len(), 3);
        // 1-based header → 0-based stored index
        assert_eq!(turns[0].index, 0);
        assert_eq!(turns[1].index, 1);
        assert_eq!(turns[2].index, 2);
        assert_eq!(turns[0].role, Role::User);
        assert_eq!(turns[1].role, Role::Assistant);
        // ### form inherits the previous turn's role
        assert_eq!(turns[2].role, Role::Assistant);
    }

    #[test]
    fn test_parse_turns_content_nonempty_and_no_header_bleed() {
        let turns = parse_turns_from_body(SAMPLE, "2026-06-24");
        assert!(turns[0].content.contains("How do I do X?"));
        // turn 1 content must NOT contain turn 2's header text
        assert!(!turns[0].content.contains("Turn 2"));
        // assistant body content is captured (callout lines may be folded into content)
        assert!(turns[1].content.contains("Here is how."));
    }

    #[test]
    fn thinking_callout_is_not_content() {
        let turns = parse_turns_from_body(SAMPLE, "2026-06-24");
        assert_eq!(turns[1].thinking.as_deref(), Some("Let me consider."));
        assert!(turns[1].content.starts_with("Here is how."));
        assert!(!turns[1].content.contains("[!thinking]"));
        assert_eq!(turns[0].thinking, None);
    }

    #[test]
    fn a_turn_of_thinking_alone_has_no_content() {
        let body = "## Turn 1 — Assistant (05:30)\n\n> [!thinking]- Thinking\n\n\
                    ## Turn 2 — Assistant (05:31)\n\n> [!thinking]- Thinking\n> step one\n> \n> step two\n";
        let turns = parse_turns_from_body(body, "2026-06-24");
        assert_eq!(turns[0].content, "");
        assert_eq!(turns[0].thinking.as_deref(), Some(""));
        assert_eq!(turns[1].content, "");
        assert_eq!(turns[1].thinking.as_deref(), Some("step one\n\nstep two"));
    }

    #[test]
    fn content_quoting_a_callout_stays_content() {
        // The callout counts only as the turn's first line; a quote after it
        // and a callout further down are content.
        let body = "## Turn 1 — User (05:30)\n\nSee this:\n\n> [!thinking]- Thinking\n> pasted\n";
        let turns = parse_turns_from_body(body, "2026-06-24");
        assert_eq!(turns[0].thinking, None);
        assert!(turns[0].content.contains("> pasted"));

        let body =
            "## Turn 1 — Assistant (05:30)\n\n> [!thinking]- Thinking\n> plan\n\n> quoted reply\n";
        let turns = parse_turns_from_body(body, "2026-06-24");
        assert_eq!(turns[0].thinking.as_deref(), Some("plan"));
        assert_eq!(turns[0].content, "> quoted reply");
    }

    #[test]
    fn rendered_turns_read_back_without_their_thinking() {
        use crate::ingest::markdown::{extract_body_text, render_session};
        use crate::ingest::{AgentKind, Session, TokenUsage};

        let turn = |index, role, content: &str, thinking: Option<&str>| Turn {
            index,
            role,
            timestamp: None,
            content: content.to_string(),
            actions: Vec::new(),
            tokens: None,
            thinking: thinking.map(str::to_string),
            is_sidechain: false,
        };
        let session = Session {
            id: "s".to_string(),
            agent: AgentKind::ClaudeCode,
            model: None,
            project: None,
            cwd: None,
            git_branch: None,
            host: None,
            start_time: Utc::now(),
            end_time: None,
            turns: vec![
                turn(0, Role::User, "why?", None),
                turn(
                    1,
                    Role::Assistant,
                    "Because.",
                    Some("Weigh it.\n\nThen answer."),
                ),
                turn(2, Role::Assistant, "", Some("")),
            ],
            total_tokens: TokenUsage::default(),
            session_type: "interactive".to_string(),
            archived: false,
            archived_at: None,
        };
        let md = render_session(&session, chrono_tz::Tz::UTC);
        let read: Vec<_> = parse_turns_from_body(&extract_body_text(&md), "")
            .into_iter()
            .map(|t| (t.content, t.thinking))
            .collect();
        assert_eq!(
            read,
            [
                ("why?".to_string(), None),
                (
                    "Because.".to_string(),
                    Some("Weigh it.\n\nThen answer.".to_string())
                ),
                (String::new(), Some(String::new())),
            ]
        );
    }

    #[test]
    fn test_parse_turns_ignores_literal_header_in_content_via_anchor() {
        // A content line that looks like a header but lacks the strict anchor
        // (no valid Role after "—", or appears mid-paragraph) must not split.
        let body = "## Turn 1 — User (05:30)\n\nI typed: ## Turn 99 — Banana\n";
        let turns = parse_turns_from_body(body, "2026-06-24");
        assert_eq!(turns.len(), 1, "non-Role anchor must not split a turn");
    }
}
