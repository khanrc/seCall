use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentKind {
    ClaudeCode,
    ClaudeAi,
    ChatGpt,
    Codex,
    GeminiCli,
    GeminiWeb,
    OpenCode,
}

impl AgentKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            AgentKind::ClaudeCode => "claude-code",
            AgentKind::ClaudeAi => "claude-ai",
            AgentKind::ChatGpt => "chatgpt",
            AgentKind::Codex => "codex",
            AgentKind::GeminiCli => "gemini-cli",
            AgentKind::GeminiWeb => "gemini-web",
            AgentKind::OpenCode => "opencode",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub agent: AgentKind,
    pub model: Option<String>,
    pub project: Option<String>,
    pub cwd: Option<PathBuf>,
    pub git_branch: Option<String>,
    pub host: Option<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub turns: Vec<Turn>,
    pub total_tokens: TokenUsage,
    /// 세션 분류 레이블 (e.g. "interactive", "automated"). 기본값: "interactive"
    pub session_type: String,
    /// P45 — archive 상태. vault frontmatter SSOT 와 동기화.
    pub archived: bool,
    /// archive 된 시각 (archived=true 일 때만 Some)
    pub archived_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub index: u32,
    pub role: Role,
    pub timestamp: Option<DateTime<Utc>>,
    pub content: String,
    pub actions: Vec<Action>,
    pub tokens: Option<TokenUsage>,
    pub thinking: Option<String>,
    pub is_sidechain: bool,
}

impl Turn {
    /// Text used for both vector and BM25 indexing: the turn's content plus a
    /// compact fold of its tool calls (`[Tool: name] input_summary`). Thinking
    /// is intentionally excluded from the index (stored/displayed only), and
    /// tool output is excluded (input summary carries the "what ran" signal at
    /// far lower token cost). Sharing this between the chunker and BM25 keeps
    /// tool-only assistant turns — which have empty `content` — searchable
    /// instead of indexing as empty and being skipped (#1585).
    pub fn index_text(&self) -> String {
        let mut parts = Vec::new();
        if !self.content.is_empty() {
            parts.push(self.content.clone());
        }
        for action in &self.actions {
            if let Action::ToolUse {
                name,
                input_summary,
                ..
            } = action
            {
                parts.push(format!("[Tool: {}] {}", name, input_summary));
            }
        }
        parts.join("\n\n")
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    ToolUse {
        name: String,
        input_summary: String,
        output_summary: String,
        tool_use_id: Option<String>,
    },
    FileEdit {
        path: String,
    },
    Command {
        cmd: String,
        exit_code: Option<i32>,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input: u64,
    pub output: u64,
    pub cached: u64,
}
