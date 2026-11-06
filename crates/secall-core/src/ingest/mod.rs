use std::path::Path;

pub mod chatgpt;
pub mod claude;
pub mod claude_ai;
pub mod codex;
pub mod detect;
pub mod gemini;
pub mod gemini_web;
pub mod lint;
pub mod markdown;
pub mod markdown_reparse;
pub use markdown_reparse::parse_turns_from_body;
pub mod opencode;
pub mod types;

pub use types::{Action, AgentKind, Role, Session, TokenUsage, Turn};

/// P49: secall 이 Claude Code 를 invoke 해 세션 요약을 생성할 때 던지는 프롬프트
/// prefix. 변경 시 wiki 등 생성 코드와 함께 갱신할 것.
const SECALL_SUMMARY_PROMPT_PREFIX: &str =
    "Analyze the following conversation and produce a JSON array of topic-based summaries";

/// P83/P90 (#82, refactoring report [낮음]): codex/claude wiki 호출이 만든
/// subprocess 세션을 ingest 가 self-ingest 노이즈로 식별하기 위한 prompt prefix
/// marker. `is_noise_session` 이 첫 user turn 에서 이 marker 를 검출하면 해당
/// 세션을 skip 한다.
///
/// 소유권은 노이즈 판정 주체인 `ingest` 에 둔다 — wiki 백엔드 (`wiki::{codex,
/// claude}::generate`) 가 prompt 앞에 이 상수를 prepend 하므로, 하위 레이어
/// (ingest) 의 식별 규칙을 상위 레이어 (wiki) 가 참조하는 올바른 의존 방향이다.
/// (이전엔 `wiki::WIKI_INVOCATION_MARKER` 로 정의되어 ingest → wiki 역참조였음.)
pub const WIKI_INVOCATION_MARKER: &str = "<!-- secall:wiki-update -->";

/// Opening of the user turn in which Codex sends the repo's AGENTS.md.
pub const AGENTS_MD_TURN_PREFIX: &str = "# AGENTS.md instructions";

/// Openings of user turns a harness injects ahead of the real task. Codex sends
/// the repo's AGENTS.md, an `<environment_context>` block and, with apps enabled,
/// `<recommended_plugins>` as *user* messages, so whatever keys on "the first user
/// turn" (classification, noise detection, the session summary) must look past
/// them. The codex parser already drops the AGENTS.md turn; it stays listed here
/// so sessions ingested before that skip still classify correctly on backfill.
pub const INJECTED_USER_CONTEXT_PREFIXES: &[&str] = &[
    AGENTS_MD_TURN_PREFIX,
    "<environment_context>",
    "<recommended_plugins>",
];

/// Whether a user turn's content is harness-injected context, not a user message.
///
/// Matches the opening only: Codex often packs several blocks into one turn
/// (`<recommended_plugins>` then the AGENTS.md block, or AGENTS.md then
/// `<environment_context>`), so a whole-turn shape check would miss them.
pub fn is_injected_user_context(content: &str) -> bool {
    let head = content.trim_start();
    INJECTED_USER_CONTEXT_PREFIXES
        .iter()
        .any(|prefix| head.starts_with(prefix))
}

/// The first user turn that is not harness-injected context.
pub fn first_real_user_turn(turns: &[Turn]) -> Option<&Turn> {
    turns
        .iter()
        .find(|t| t.role == Role::User && !is_injected_user_context(&t.content))
}

pub trait SessionParser: Send + Sync {
    /// Check if this parser can handle the given path
    fn can_parse(&self, path: &Path) -> bool;

    /// Parse the session file and return a Session
    fn parse(&self, path: &Path) -> crate::error::Result<Session>;

    /// The agent kind this parser handles
    fn agent_kind(&self) -> AgentKind;

    /// Parse a file that may contain multiple sessions (1:N).
    /// Default: wraps parse() for 1:1 parsers.
    fn parse_all(&self, path: &Path) -> crate::error::Result<Vec<Session>> {
        Ok(vec![self.parse(path)?])
    }
}

/// P49 + P83: 노이즈 세션을 감지한다.
///
/// secall 자체가 Claude Code 또는 codex 를 invoke 해 요약/위키를 생성하는 흐름이
/// `~/.claude/projects/` 또는 `~/.codex/sessions/` 에 또 jsonl 로 남으면서 자기참조
/// ingest 가 발생해 vault 가 거의 동일한 짧은 세션으로 오염됐다. 차단 패턴:
///   1. cwd 가 OS 임시 디렉토리 (`/private/var/folders`, `/var/folders`, `/tmp`)
///   2. 첫 user turn 본문이 secall 의 알려진 summary 프롬프트 prefix 로 시작
///   3. (P83) 첫 user turn 본문이 `WIKI_INVOCATION_MARKER` 를 포함
///      — `secall wiki update` 가 codex/claude 백엔드 subprocess 호출 시 prompt
///      앞에 prefix 로 추가하는 marker. Issue #82 fix.
///
/// 매치 시 사유 문자열을 반환, 정상 세션은 `None`.
pub fn is_noise_session(session: &Session) -> Option<&'static str> {
    if let Some(cwd) = session.cwd.as_ref() {
        if cwd.starts_with("/private/var/folders/")
            || cwd.starts_with("/var/folders/")
            || cwd.starts_with("/tmp/")
        {
            return Some("tmpdir cwd");
        }
    }

    if let Some(first_user) = first_real_user_turn(&session.turns) {
        let content_trimmed = first_user.content.trim_start();
        if content_trimmed.starts_with(SECALL_SUMMARY_PROMPT_PREFIX) {
            return Some("secall summary prompt");
        }
        // P83 marker 는 `contains` — codex/claude 가 system prompt 를 앞에 prepend
        // 하는 경우에도 robust. 변수는 일관성 위해 `content_trimmed` 재사용
        // (trim 결과 marker 자체는 변하지 않음).
        if content_trimmed.contains(WIKI_INVOCATION_MARKER) {
            return Some("secall wiki invocation");
        }
    }

    None
}

/// Reason a parsed session is not an indexable standalone conversation, or `None`.
/// Intrinsic to the parsed content — independent of the `subagents/` path skip in
/// the ingest collector (which is only a fast pre-filter), so it still holds if CC
/// renames that subtree (logan-cha/log#1607):
///
/// - **no turns** — a workflow journal (`subagents/workflows/wf_*/journal.jsonl`)
///   is a `type:started/result` event log that parses to zero turns; lacking a
///   content `sessionId` it would take the filename-stem id "journal" and collide
///   across every workflow.
/// - **subagent sidechain** — `subagents/agent-<hash>.jsonl` (all turns
///   `isSidechain`) carries the PARENT session's `sessionId`, so ingesting it as
///   the parent shrinks the turn count → FullRebuild → the parent's vectors wiped.
pub fn subagent_skip_reason(session: &Session) -> Option<&'static str> {
    if session.turns.is_empty() {
        return Some("no turns");
    }
    if session.turns.iter().all(|t| t.is_sidechain) {
        return Some("subagent sidechain");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn dummy_session(cwd: Option<&str>, first_user_content: Option<&str>) -> Session {
        let turns = if let Some(content) = first_user_content {
            vec![Turn {
                index: 0,
                role: Role::User,
                timestamp: None,
                content: content.to_string(),
                actions: vec![],
                tokens: None,
                thinking: None,
                is_sidechain: false,
            }]
        } else {
            vec![]
        };
        Session {
            id: "t".into(),
            agent: AgentKind::ClaudeCode,
            model: None,
            project: None,
            cwd: cwd.map(std::path::PathBuf::from),
            git_branch: None,
            host: None,
            start_time: Utc::now(),
            end_time: None,
            turns,
            total_tokens: TokenUsage::default(),
            session_type: "interactive".into(),
            archived: false,
            archived_at: None,
        }
    }

    fn user_turn(content: &str) -> Turn {
        Turn {
            index: 0,
            role: Role::User,
            timestamp: None,
            content: content.to_string(),
            actions: vec![],
            tokens: None,
            thinking: None,
            is_sidechain: false,
        }
    }

    #[test]
    fn first_real_user_turn_skips_codex_injected_context() {
        let task = "<today>2026-11-06</today>\n\n<digest>\n</digest>";
        let turns = vec![
            user_turn("<environment_context>\n  <cwd>/home/u/log</cwd>\n</environment_context>"),
            user_turn("\n<recommended_plugins>\n- github\n</recommended_plugins>"),
            user_turn("# AGENTS.md instructions for /home/u/log"),
            user_turn(task),
        ];
        assert_eq!(
            first_real_user_turn(&turns).map(|t| t.content.as_str()),
            Some(task)
        );
    }

    #[test]
    fn first_real_user_turn_is_none_when_only_injected_context() {
        let turns = vec![user_turn("<environment_context></environment_context>")];
        assert!(first_real_user_turn(&turns).is_none());
    }

    #[test]
    fn noise_marker_behind_injected_context_is_detected() {
        let mut s = dummy_session(
            Some("/home/u/proj"),
            Some("<environment_context></environment_context>"),
        );
        s.turns.push(user_turn(&format!(
            "{WIKI_INVOCATION_MARKER}\nupdate the wiki"
        )));
        assert_eq!(is_noise_session(&s), Some("secall wiki invocation"));
    }

    #[test]
    fn is_noise_macos_tmpdir() {
        let s = dummy_session(Some("/private/var/folders/sy/abc/T"), None);
        assert_eq!(is_noise_session(&s), Some("tmpdir cwd"));
    }

    #[test]
    fn is_noise_linux_var_folders() {
        let s = dummy_session(Some("/var/folders/x/y/T"), None);
        assert_eq!(is_noise_session(&s), Some("tmpdir cwd"));
    }

    #[test]
    fn subagent_skip_empty_is_no_turns() {
        let s = dummy_session(Some("/home/u/proj"), None); // zero turns (e.g. journal)
        assert_eq!(subagent_skip_reason(&s), Some("no turns"));
    }

    #[test]
    fn subagent_skip_all_sidechain() {
        let mut s = dummy_session(Some("/home/u/proj"), Some("hi"));
        s.turns[0].is_sidechain = true;
        assert_eq!(subagent_skip_reason(&s), Some("subagent sidechain"));
    }

    #[test]
    fn subagent_skip_normal_session_is_none() {
        let s = dummy_session(Some("/home/u/proj"), Some("hi"));
        assert_eq!(subagent_skip_reason(&s), None);
    }

    #[test]
    fn subagent_skip_mixed_turns_is_none() {
        // A parent whose own file inlines a sidechain turn must NOT be skipped.
        let mut s = dummy_session(Some("/home/u/proj"), Some("hi"));
        s.turns.push(Turn {
            index: 1,
            role: Role::Assistant,
            timestamp: None,
            content: "sub".into(),
            actions: vec![],
            tokens: None,
            thinking: None,
            is_sidechain: true,
        });
        assert_eq!(subagent_skip_reason(&s), None);
    }

    #[test]
    fn is_noise_tmp() {
        let s = dummy_session(Some("/tmp/foo"), None);
        assert_eq!(is_noise_session(&s), Some("tmpdir cwd"));
    }

    #[test]
    fn is_not_noise_normal_cwd() {
        let s = dummy_session(Some("/Users/me/projects/seCall"), Some("Fix the bug"));
        assert_eq!(is_noise_session(&s), None);
    }

    #[test]
    fn is_noise_secall_summary_prompt() {
        let s = dummy_session(
            Some("/Users/me/projects/seCall"),
            Some("Analyze the following conversation and produce a JSON array of topic-based summaries.\n\nEach element..."),
        );
        assert_eq!(is_noise_session(&s), Some("secall summary prompt"));
    }

    #[test]
    fn is_noise_secall_summary_prompt_with_leading_whitespace() {
        let s = dummy_session(
            Some("/Users/me/projects/seCall"),
            Some("   Analyze the following conversation and produce a JSON array of topic-based summaries"),
        );
        assert_eq!(is_noise_session(&s), Some("secall summary prompt"));
    }

    #[test]
    fn is_not_noise_when_no_cwd_no_user_turn() {
        let s = dummy_session(None, None);
        assert_eq!(is_noise_session(&s), None);
    }

    /// P83 (issue #82): codex/claude wiki 호출이 marker prefix 한 prompt 로
    /// 생성한 세션은 self-ingest 루프 차단을 위해 skip.
    #[test]
    fn is_noise_wiki_invocation_marker_at_start() {
        let prompt = format!(
            "{}\n\nUpdate the wiki for the following sessions...",
            WIKI_INVOCATION_MARKER
        );
        let s = dummy_session(Some("/Users/me/projects/seCall"), Some(&prompt));
        assert_eq!(is_noise_session(&s), Some("secall wiki invocation"));
    }

    #[test]
    fn is_noise_wiki_invocation_marker_in_middle() {
        // marker 가 어디에 있든 검출되어야 함 (codex/claude 의 system prompt 가
        // 앞에 붙는 경우에도 robust).
        let prompt = format!("Some preamble...\n{}\nMore content", WIKI_INVOCATION_MARKER);
        let s = dummy_session(Some("/Users/me/projects/seCall"), Some(&prompt));
        assert_eq!(is_noise_session(&s), Some("secall wiki invocation"));
    }

    #[test]
    fn is_not_noise_without_wiki_marker() {
        let s = dummy_session(
            Some("/Users/me/projects/seCall"),
            Some("Just a normal user prompt without any markers"),
        );
        assert_eq!(is_noise_session(&s), None);
    }
}
