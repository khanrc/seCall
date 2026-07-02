use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{anyhow, Result};
use chrono::DateTime;
use serde_json::Value;

use super::types::{Action, AgentKind, Role, Session, TokenUsage, Turn};
use super::SessionParser;

const TOOL_OUTPUT_MAX_CHARS: usize = 1000;

pub struct ClaudeCodeParser;

impl SessionParser for ClaudeCodeParser {
    fn can_parse(&self, path: &Path) -> bool {
        // Match ~/.claude/projects/**/*.jsonl pattern
        let path_str = path.to_string_lossy();
        (path_str.contains("/.claude/projects/") || path_str.contains("\\.claude\\projects\\"))
            && path.extension().map(|e| e == "jsonl").unwrap_or(false)
    }

    fn parse(&self, path: &Path) -> crate::error::Result<Session> {
        parse_claude_jsonl(path).map_err(|e| crate::error::SecallError::Parse {
            path: path.to_string_lossy().into_owned(),
            source: e,
        })
    }

    fn agent_kind(&self) -> AgentKind {
        AgentKind::ClaudeCode
    }
}

pub fn parse_claude_jsonl(path: &Path) -> Result<Session> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    let mut session_id: Option<String> = None;
    let mut model: Option<String> = None;
    let mut cwd: Option<std::path::PathBuf> = None;
    let mut git_branch: Option<String> = None;
    let mut first_timestamp: Option<DateTime<chrono::Utc>> = None;
    let mut last_timestamp: Option<DateTime<chrono::Utc>> = None;
    let mut turns: Vec<Turn> = Vec::new();
    let mut total_tokens = TokenUsage::default();

    // Pending tool_use entries keyed by tool_use_id waiting for tool_result
    let mut pending_tool_uses: HashMap<String, usize> = HashMap::new(); // tool_use_id -> action index in last assistant turn

    // message.id of the last assistant turn, to merge Claude Code's split
    // streaming events (thinking / text / tool_use of one message arrive as
    // separate events sharing an id) into a single Turn (#1585 WS2).
    let mut last_asst_msg_id: Option<String> = None;

    let mut line_count = 0;

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        line_count += 1;

        let value: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "JSON parse error (skipping line)");
                continue;
            }
        };

        let msg_type = match value["type"].as_str() {
            Some(t) => t,
            None => continue,
        };

        // Extract timestamp
        let ts = value["timestamp"]
            .as_str()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        if let Some(t) = ts {
            if first_timestamp.is_none() {
                first_timestamp = Some(t);
            }
            last_timestamp = Some(t);
        }

        match msg_type {
            "user" => {
                // Extract session metadata from first user message
                if session_id.is_none() {
                    session_id = value["sessionId"].as_str().map(String::from);
                }
                if cwd.is_none() {
                    cwd = value["cwd"].as_str().map(std::path::PathBuf::from);
                }
                if git_branch.is_none() {
                    git_branch = value["gitBranch"].as_str().map(String::from);
                }

                let is_sidechain = value["isSidechain"].as_bool().unwrap_or(false);
                let message = &value["message"];
                let content_val = &message["content"];

                // Check if this is a tool_result message
                if content_val.is_array() {
                    let items = content_val.as_array().unwrap();
                    let has_tool_result = items
                        .iter()
                        .any(|item| item["type"].as_str() == Some("tool_result"));

                    if has_tool_result {
                        // Attach tool results to the last assistant turn
                        for item in items {
                            if item["type"].as_str() == Some("tool_result") {
                                let tool_use_id =
                                    item["tool_use_id"].as_str().unwrap_or("").to_string();
                                let output = extract_tool_result_content(&item["content"]);
                                let truncated = truncate_str(&output, TOOL_OUTPUT_MAX_CHARS);

                                // Find the corresponding action in the last assistant turn
                                if let Some(&action_idx) = pending_tool_uses.get(&tool_use_id) {
                                    if let Some(Action::ToolUse { output_summary, .. }) = turns
                                        .last_mut()
                                        .and_then(|turn| turn.actions.get_mut(action_idx))
                                    {
                                        *output_summary = truncated;
                                    }
                                }
                            }
                        }
                        pending_tool_uses.clear();
                        continue;
                    }
                }

                // Regular user message
                let text = extract_user_text(content_val);
                if text.is_empty() {
                    continue;
                }

                let turn = Turn {
                    index: turns.len() as u32,
                    role: Role::User,
                    timestamp: ts,
                    content: text,
                    actions: Vec::new(),
                    tokens: None,
                    thinking: None,
                    is_sidechain,
                };
                turns.push(turn);
            }

            "assistant" => {
                let message = &value["message"];

                // Extract model
                if model.is_none() {
                    model = message["model"].as_str().map(String::from);
                }

                let is_sidechain = value["isSidechain"].as_bool().unwrap_or(false);

                // Parse usage
                let usage = &message["usage"];
                // Session totals are summed per-turn after the loop, not here:
                // Claude Code repeats the same usage on every split event of one
                // message, so accumulating per-event would multiply-count merged
                // messages. Per-turn tokens (deduped by the merge) are the source.
                let tokens = if !usage.is_null() {
                    Some(TokenUsage {
                        input: usage["input_tokens"].as_u64().unwrap_or(0),
                        output: usage["output_tokens"].as_u64().unwrap_or(0),
                        cached: usage["cache_read_input_tokens"].as_u64().unwrap_or(0),
                    })
                } else {
                    None
                };

                // Parse content array
                let mut text_parts: Vec<String> = Vec::new();
                let mut actions: Vec<Action> = Vec::new();
                let mut thinking_parts: Vec<String> = Vec::new();
                let mut new_pending: HashMap<String, usize> = HashMap::new();

                if let Some(content_arr) = message["content"].as_array() {
                    for item in content_arr {
                        match item["type"].as_str() {
                            Some("text") => {
                                if let Some(t) = item["text"].as_str() {
                                    text_parts.push(t.to_string());
                                }
                            }
                            Some("thinking") => {
                                if let Some(t) = item["thinking"].as_str() {
                                    thinking_parts.push(t.to_string());
                                }
                            }
                            Some("tool_use") => {
                                let name = item["name"].as_str().unwrap_or("unknown").to_string();
                                let tool_use_id = item["id"].as_str().unwrap_or("").to_string();
                                let input_summary = summarize_tool_input(&name, &item["input"]);

                                let action_idx = actions.len();
                                if !tool_use_id.is_empty() {
                                    new_pending.insert(tool_use_id.clone(), action_idx);
                                }

                                actions.push(Action::ToolUse {
                                    name,
                                    input_summary,
                                    output_summary: String::new(),
                                    tool_use_id: Some(tool_use_id),
                                });
                            }
                            _ => {}
                        }
                    }
                }

                let content = text_parts.join("\n\n");
                let thinking = if thinking_parts.is_empty() {
                    None
                } else {
                    Some(thinking_parts.join("\n\n"))
                };

                // Merge into the previous turn when it is the same assistant
                // message split across events; otherwise start a new turn. The
                // role check guards against merging across an intervening user
                // turn (message ids are unique per message anyway).
                let msg_id = message["id"].as_str().map(|s| s.to_string());
                let merge = msg_id.is_some()
                    && msg_id == last_asst_msg_id
                    && turns.last().map(|t| t.role == Role::Assistant).unwrap_or(false);

                if merge {
                    let last = turns.last_mut().unwrap();
                    let action_base = last.actions.len();
                    if !content.is_empty() {
                        if last.content.is_empty() {
                            last.content = content;
                        } else {
                            last.content.push_str("\n\n");
                            last.content.push_str(&content);
                        }
                    }
                    if let Some(tk) = thinking {
                        match last.thinking.as_mut() {
                            Some(existing) => {
                                existing.push_str("\n\n");
                                existing.push_str(&tk);
                            }
                            None => last.thinking = Some(tk),
                        }
                    }
                    last.actions.extend(actions);
                    // usage repeats across a message's events; keep the latest.
                    if tokens.is_some() {
                        last.tokens = tokens;
                    }
                    // new tool_uses land after the pre-merge actions — offset
                    // their pending indices so tool_result attaches correctly.
                    for (k, v) in new_pending {
                        pending_tool_uses.insert(k, v + action_base);
                    }
                } else {
                    let turn = Turn {
                        index: turns.len() as u32,
                        role: Role::Assistant,
                        timestamp: ts,
                        content,
                        actions,
                        tokens,
                        thinking,
                        is_sidechain,
                    };
                    turns.push(turn);
                    pending_tool_uses = new_pending;
                }
                last_asst_msg_id = msg_id;
            }

            // Skip non-conversation message types
            "queue-operation" | "attachment" | "last-prompt" => continue,
            _ => continue,
        }
    }

    if line_count == 0 {
        return Err(anyhow!("empty session file"));
    }

    let id = session_id
        .or_else(|| {
            // Derive from filename if not in content
            path.file_stem().and_then(|s| s.to_str()).map(String::from)
        })
        .unwrap_or_else(|| uuid_from_path(path));

    // Derive project from cwd
    let project = cwd
        .as_ref()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .map(String::from);

    let start_time = first_timestamp.unwrap_or_else(chrono::Utc::now);
    let end_time = last_timestamp;

    // Sum session totals from the (merge-deduped) per-turn usage.
    for turn in &turns {
        if let Some(tk) = &turn.tokens {
            total_tokens.input += tk.input;
            total_tokens.output += tk.output;
            total_tokens.cached += tk.cached;
        }
    }

    Ok(Session {
        id,
        agent: AgentKind::ClaudeCode,
        model,
        project,
        cwd,
        git_branch,
        host: Some(gethostname::gethostname().to_string_lossy().to_string()),
        start_time,
        end_time,
        turns,
        total_tokens,
        session_type: "interactive".to_string(),
        archived: false,
        archived_at: None,
    })
}

fn extract_user_text(content: &Value) -> String {
    if content.is_string() {
        return content.as_str().unwrap_or("").to_string();
    }
    if let Some(arr) = content.as_array() {
        let parts: Vec<String> = arr
            .iter()
            .filter_map(|item| {
                if item["type"].as_str() == Some("text") {
                    item["text"].as_str().map(String::from)
                } else {
                    None
                }
            })
            .collect();
        return parts.join("\n");
    }
    String::new()
}

fn extract_tool_result_content(content: &Value) -> String {
    if content.is_string() {
        return content.as_str().unwrap_or("").to_string();
    }
    if let Some(arr) = content.as_array() {
        let parts: Vec<String> = arr
            .iter()
            .filter_map(|item| {
                if item["type"].as_str() == Some("text") {
                    item["text"].as_str().map(String::from)
                } else {
                    None
                }
            })
            .collect();
        return parts.join("\n");
    }
    String::new()
}

fn summarize_tool_input(tool_name: &str, input: &Value) -> String {
    match tool_name {
        "Bash" | "bash" => input["command"].as_str().unwrap_or("").to_string(),
        "Read" | "read" => input["file_path"].as_str().unwrap_or("").to_string(),
        "Edit" | "edit" | "MultiEdit" => input["file_path"].as_str().unwrap_or("").to_string(),
        "Write" | "write" => input["file_path"].as_str().unwrap_or("").to_string(),
        "Grep" | "grep" => {
            let pattern = input["pattern"].as_str().unwrap_or("");
            let path = input["path"].as_str().unwrap_or("");
            format!("{pattern} in {path}")
        }
        "Glob" | "glob" => input["pattern"].as_str().unwrap_or("").to_string(),
        _ => {
            // Generic: show first 200 chars of JSON
            let s = input.to_string();
            truncate_str(&s, 200)
        }
    }
}

fn truncate_str(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        s.to_string()
    } else {
        let truncated: String = chars[..max].iter().collect();
        format!("{}...", truncated)
    }
}

fn uuid_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_jsonl(lines: &[&str]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(f, "{}", line).unwrap();
        }
        f
    }

    #[test]
    fn test_parse_basic_user_assistant() {
        let lines = &[
            r#"{"type":"user","message":{"role":"user","content":"Hello there"},"timestamp":"2026-04-05T10:00:00Z","sessionId":"test-session-123","cwd":"/Users/user/myproject","gitBranch":"main","version":"1.0"}"#,
            r#"{"type":"assistant","message":{"role":"assistant","id":"msg_1","model":"claude-opus-4-6","content":[{"type":"text","text":"Hello! How can I help?"}],"usage":{"input_tokens":5,"output_tokens":10,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}},"timestamp":"2026-04-05T10:00:01Z"}"#,
        ];
        let f = write_jsonl(lines);
        let session = parse_claude_jsonl(f.path()).unwrap();
        assert_eq!(session.id, "test-session-123");
        assert_eq!(session.turns.len(), 2);
        assert_eq!(session.turns[0].role, Role::User);
        assert_eq!(session.turns[1].role, Role::Assistant);
        assert_eq!(session.model.as_deref(), Some("claude-opus-4-6"));
    }

    #[test]
    fn test_merge_split_assistant_message() {
        // Claude Code streams one assistant message's blocks (thinking / text /
        // tool_use) as separate events sharing a message.id; they must collapse
        // into one turn, preserving thinking and attaching the tool_result to
        // the merged turn's action (#1585 WS2).
        let lines = &[
            r#"{"type":"user","message":{"role":"user","content":"run ls"},"timestamp":"2026-04-05T10:00:00Z","sessionId":"s1","cwd":"/p","gitBranch":"main"}"#,
            r#"{"type":"assistant","message":{"role":"assistant","id":"msg_A","model":"claude","content":[{"type":"thinking","thinking":"let me think"}],"usage":{"output_tokens":3}},"timestamp":"2026-04-05T10:00:01Z"}"#,
            r#"{"type":"assistant","message":{"role":"assistant","id":"msg_A","content":[{"type":"text","text":"Listing now"},{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls"}}]},"timestamp":"2026-04-05T10:00:02Z"}"#,
            r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"file1\nfile2"}]},"timestamp":"2026-04-05T10:00:03Z"}"#,
            r#"{"type":"assistant","message":{"role":"assistant","id":"msg_B","content":[{"type":"text","text":"Done"}]},"timestamp":"2026-04-05T10:00:04Z"}"#,
        ];
        let f = write_jsonl(lines);
        let session = parse_claude_jsonl(f.path()).unwrap();

        // user, merged assistant (msg_A), separate assistant (msg_B)
        assert_eq!(session.turns.len(), 3);
        let a = &session.turns[1];
        assert_eq!(a.role, Role::Assistant);
        assert_eq!(a.thinking.as_deref(), Some("let me think"));
        assert_eq!(a.content, "Listing now");
        assert_eq!(a.actions.len(), 1);
        match &a.actions[0] {
            Action::ToolUse {
                name,
                output_summary,
                ..
            } => {
                assert_eq!(name, "Bash");
                assert!(
                    output_summary.contains("file1"),
                    "tool_result must attach to the merged turn's action"
                );
            }
            _ => panic!("expected ToolUse action"),
        }
        // tool text is folded into the indexed text so the turn stays searchable
        assert!(a.index_text().contains("[Tool: Bash]"));
        // a different message.id starts a new turn
        assert_eq!(session.turns[2].content, "Done");
    }

    #[test]
    fn test_merge_interleaved_parallel_tools() {
        // One assistant message issues two tool calls; Claude Code interleaves
        // tool_use / tool_result events, all under one message.id. Both results
        // must attach to the right action in the single merged turn (this is
        // what the pending-index offset exists for).
        let lines = &[
            r#"{"type":"user","message":{"role":"user","content":"do two things"},"timestamp":"2026-04-05T10:00:00Z","sessionId":"s6","cwd":"/p","gitBranch":"main"}"#,
            r#"{"type":"assistant","message":{"role":"assistant","id":"mp","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls"}}]},"timestamp":"2026-04-05T10:00:01Z"}"#,
            r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"OUT_ONE"}]},"timestamp":"2026-04-05T10:00:02Z"}"#,
            r#"{"type":"assistant","message":{"role":"assistant","id":"mp","content":[{"type":"tool_use","id":"t2","name":"Grep","input":{"pattern":"foo"}}]},"timestamp":"2026-04-05T10:00:03Z"}"#,
            r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t2","content":"OUT_TWO"}]},"timestamp":"2026-04-05T10:00:04Z"}"#,
        ];
        let f = write_jsonl(lines);
        let s = parse_claude_jsonl(f.path()).unwrap();
        assert_eq!(s.turns.len(), 2, "user + one merged assistant turn");
        let a = &s.turns[1];
        assert_eq!(a.actions.len(), 2, "both tool calls in the merged turn");
        let outs: Vec<&str> = a
            .actions
            .iter()
            .filter_map(|act| match act {
                Action::ToolUse { output_summary, .. } => Some(output_summary.as_str()),
                _ => None,
            })
            .collect();
        assert!(outs.contains(&"OUT_ONE"), "first tool_result attached");
        assert!(outs.contains(&"OUT_TWO"), "second tool_result attached to offset action");
    }

    #[test]
    fn test_split_message_tokens_counted_once() {
        // One message split across two events repeats the same usage; the session
        // total must count it once, not once per event.
        let lines = &[
            r#"{"type":"user","message":{"role":"user","content":"go"},"timestamp":"2026-04-05T10:00:00Z","sessionId":"s5","cwd":"/tmp","gitBranch":"main"}"#,
            r#"{"type":"assistant","message":{"role":"assistant","id":"mm","content":[{"type":"thinking","thinking":"hmm"}],"usage":{"input_tokens":10,"output_tokens":20,"cache_read_input_tokens":5}},"timestamp":"2026-04-05T10:00:01Z"}"#,
            r#"{"type":"assistant","message":{"role":"assistant","id":"mm","content":[{"type":"text","text":"answer"}],"usage":{"input_tokens":10,"output_tokens":20,"cache_read_input_tokens":5}},"timestamp":"2026-04-05T10:00:02Z"}"#,
        ];
        let f = write_jsonl(lines);
        let s = parse_claude_jsonl(f.path()).unwrap();
        assert_eq!(s.turns.len(), 2, "user + one merged assistant");
        assert_eq!(s.total_tokens.input, 10, "repeated usage counted once");
        assert_eq!(s.total_tokens.output, 20);
        assert_eq!(s.total_tokens.cached, 5);
    }

    #[test]
    fn test_parse_tool_use() {
        let lines = &[
            r#"{"type":"user","message":{"role":"user","content":"Run ls"},"timestamp":"2026-04-05T10:00:00Z","sessionId":"s1","cwd":"/tmp","gitBranch":"main","version":"1.0"}"#,
            r#"{"type":"assistant","message":{"role":"assistant","id":"msg_1","model":"claude","content":[{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"ls -la","description":"List files"}}],"usage":{"input_tokens":5,"output_tokens":3}},"timestamp":"2026-04-05T10:00:01Z"}"#,
            r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"file1.txt\nfile2.txt","is_error":false}]},"timestamp":"2026-04-05T10:00:02Z"}"#,
        ];
        let f = write_jsonl(lines);
        let session = parse_claude_jsonl(f.path()).unwrap();
        // user + assistant (tool_result doesn't create new turn)
        assert_eq!(session.turns.len(), 2);
        assert_eq!(session.turns[1].actions.len(), 1);
        if let Action::ToolUse {
            name,
            output_summary,
            ..
        } = &session.turns[1].actions[0]
        {
            assert_eq!(name, "Bash");
            assert!(output_summary.contains("file1.txt"));
        }
    }

    #[test]
    fn test_parse_thinking_block() {
        let lines = &[
            r#"{"type":"user","message":{"role":"user","content":"Think about this"},"timestamp":"2026-04-05T10:00:00Z","sessionId":"s2","cwd":"/tmp","gitBranch":"main","version":"1.0"}"#,
            r#"{"type":"assistant","message":{"role":"assistant","id":"msg_1","model":"claude","content":[{"type":"thinking","thinking":"Let me reason..."},{"type":"text","text":"Here is my answer"}],"usage":{"input_tokens":5,"output_tokens":8}},"timestamp":"2026-04-05T10:00:01Z"}"#,
        ];
        let f = write_jsonl(lines);
        let session = parse_claude_jsonl(f.path()).unwrap();
        assert_eq!(
            session.turns[1].thinking.as_deref(),
            Some("Let me reason...")
        );
        assert!(session.turns[1].content.contains("Here is my answer"));
    }

    #[test]
    fn test_skip_invalid_lines() {
        let lines = &[
            r#"{"type":"user","message":{"role":"user","content":"Hello"},"timestamp":"2026-04-05T10:00:00Z","sessionId":"s3","cwd":"/tmp","gitBranch":"main","version":"1.0"}"#,
            r#"INVALID JSON LINE"#,
            r#"{"type":"queue-operation","operation":"enqueue","timestamp":"2026-04-05T10:00:01Z"}"#,
        ];
        let f = write_jsonl(lines);
        let session = parse_claude_jsonl(f.path()).unwrap();
        assert_eq!(session.turns.len(), 1); // Only the valid user turn
    }

    #[test]
    fn test_empty_file_returns_err() {
        let f = write_jsonl(&[]);
        let result = parse_claude_jsonl(f.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_token_aggregation() {
        let lines = &[
            r#"{"type":"user","message":{"role":"user","content":"Q1"},"timestamp":"2026-04-05T10:00:00Z","sessionId":"s4","cwd":"/tmp","gitBranch":"main","version":"1.0"}"#,
            r#"{"type":"assistant","message":{"role":"assistant","id":"m1","model":"claude","content":[{"type":"text","text":"A1"}],"usage":{"input_tokens":100,"output_tokens":50,"cache_read_input_tokens":200}},"timestamp":"2026-04-05T10:00:01Z"}"#,
        ];
        let f = write_jsonl(lines);
        let session = parse_claude_jsonl(f.path()).unwrap();
        assert_eq!(session.total_tokens.input, 100);
        assert_eq!(session.total_tokens.output, 50);
        assert_eq!(session.total_tokens.cached, 200);
    }
}
