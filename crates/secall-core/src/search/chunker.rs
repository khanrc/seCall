use std::sync::OnceLock;

use crate::ingest::Session;

// dragonkue/multilingual-e5-small (xlm-roberta) has max_seq_length 512. We split
// on the model tokenizer so a chunk never overflows and gets silently truncated
// (the char-based 3600 cap did exactly that for Korean — ~2 chars/token — losing
// the tail of long turns). 510 leaves room for the 2 special tokens the embedder
// adds; overlap is ~15%.
const MAX_CHUNK_TOKENS: usize = 510;
const OVERLAP_TOKENS: usize = 77;

// Fallback char budget when the model tokenizer isn't on disk (unit tests, or
// before the model is downloaded). Conservative: 1000 chars stays under 510
// tokens even for dense Korean (~2 chars/token).
const FALLBACK_MAX_CHUNK_CHARS: usize = 1000;
const FALLBACK_OVERLAP_CHARS: usize = 150;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub session_id: String,
    pub turn_index: u32,
    pub seq: u32,
    pub text: String,
    pub context: String,
}

pub fn chunk_session(session: &Session, tz: chrono_tz::Tz) -> Vec<Chunk> {
    let mut chunks = Vec::new();

    for turn in &session.turns {
        let context = format!(
            "Session: {} {} {} | Turn {}: {}",
            session.agent.as_str(),
            session.project.as_deref().unwrap_or("unknown"),
            session.start_time.with_timezone(&tz).format("%Y-%m-%d"),
            turn.index + 1,
            turn.role.as_str(),
        );

        // index_text() folds tool-call summaries into the embedded/BM25 text so
        // tool-only assistant turns (empty content) stay searchable (#1585).
        let text = turn.index_text();
        if text.is_empty() {
            continue;
        }

        for (seq, chunk_text) in split_turn_text(&text).into_iter().enumerate() {
            chunks.push(Chunk {
                session_id: session.id.clone(),
                turn_index: turn.index,
                seq: seq as u32,
                text: chunk_text,
                context: context.clone(),
            });
        }
    }

    chunks
}

/// Split one turn's text into chunks that stay within the embedding model's
/// token budget. Uses the model tokenizer when available (token-precise, the
/// dragonkue/e5 path); falls back to a conservative char split otherwise.
fn split_turn_text(text: &str) -> Vec<String> {
    if let Some(tok) = model_tokenizer() {
        return split_by_tokens(text, tok, MAX_CHUNK_TOKENS, OVERLAP_TOKENS);
    }
    split_into_chunks(text, FALLBACK_MAX_CHUNK_CHARS, FALLBACK_OVERLAP_CHARS)
}

/// Lazily load the embedding model's tokenizer from the model dir. `None` when
/// the model hasn't been downloaded (tests, fresh install) → char fallback.
fn model_tokenizer() -> Option<&'static tokenizers::Tokenizer> {
    static TOKENIZER: OnceLock<Option<tokenizers::Tokenizer>> = OnceLock::new();
    TOKENIZER
        .get_or_init(|| {
            let path = crate::search::model_manager::default_model_path().join("tokenizer.json");
            tokenizers::Tokenizer::from_file(&path).ok()
        })
        .as_ref()
}

/// Token-precise windowing. Encodes without special tokens, then slices the
/// original text at the byte offsets of token boundaries so each window holds
/// at most `max_tokens` content tokens, with `overlap` tokens of carry-over.
fn split_by_tokens(
    text: &str,
    tokenizer: &tokenizers::Tokenizer,
    max_tokens: usize,
    overlap: usize,
) -> Vec<String> {
    let encoding = match tokenizer.encode(text, false) {
        Ok(e) => e,
        Err(_) => return split_into_chunks(text, FALLBACK_MAX_CHUNK_CHARS, FALLBACK_OVERLAP_CHARS),
    };
    let offsets = encoding.get_offsets();
    let n = offsets.len();
    if n <= max_tokens {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start_tok = 0;
    while start_tok < n {
        let end_tok = (start_tok + max_tokens).min(n);
        let start_byte = clamp_boundary(text, offsets[start_tok].0);
        let end_byte = if end_tok < n {
            clamp_boundary(text, offsets[end_tok].0)
        } else {
            text.len()
        };
        if end_byte > start_byte {
            chunks.push(text[start_byte..end_byte].to_string());
        }
        if end_tok >= n {
            break;
        }
        start_tok = end_tok.saturating_sub(overlap);
    }
    if chunks.is_empty() {
        chunks.push(text.to_string());
    }
    chunks
}

/// Round a byte index down to the nearest UTF-8 char boundary. Tokenizer
/// offsets are char-aligned in practice, but a normalizer can shift them; this
/// keeps slicing panic-free.
fn clamp_boundary(text: &str, mut idx: usize) -> usize {
    if idx >= text.len() {
        return text.len();
    }
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn split_into_chunks(text: &str, max_size: usize, overlap: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();

    if total <= max_size {
        return vec![text.to_string()];
    }

    let mut start = 0;
    while start < total {
        let end = (start + max_size).min(total);
        let chunk: String = chars[start..end].iter().collect();
        chunks.push(chunk);

        if end >= total {
            break;
        }
        // Advance with overlap
        start = end.saturating_sub(overlap);
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::types::{AgentKind, Role, Session, TokenUsage, Turn};
    use chrono::{TimeZone, Utc};

    fn make_session(turns: Vec<Turn>) -> Session {
        Session {
            id: "test-session".to_string(),
            agent: AgentKind::ClaudeCode,
            model: None,
            project: Some("testproj".to_string()),
            cwd: None,
            git_branch: None,
            host: None,
            start_time: Utc.with_ymd_and_hms(2026, 4, 5, 0, 0, 0).unwrap(),
            end_time: None,
            turns,
            total_tokens: TokenUsage::default(),
            session_type: "interactive".to_string(),
            archived: false,
            archived_at: None,
        }
    }

    #[test]
    fn test_short_turn_single_chunk() {
        let turns = vec![Turn {
            index: 0,
            role: Role::User,
            timestamp: None,
            content: "Short content".to_string(),
            actions: Vec::new(),
            tokens: None,
            thinking: None,
            is_sidechain: false,
        }];
        let session = make_session(turns);
        let chunks = chunk_session(&session, chrono_tz::Tz::UTC);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].seq, 0);
    }

    #[test]
    fn test_long_turn_split() {
        let long_content = "word ".repeat(1000);
        let turns = vec![Turn {
            index: 0,
            role: Role::User,
            timestamp: None,
            content: long_content,
            actions: Vec::new(),
            tokens: None,
            thinking: None,
            is_sidechain: false,
        }];
        let session = make_session(turns);
        let chunks = chunk_session(&session, chrono_tz::Tz::UTC);
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_context_string_correct() {
        let turns = vec![Turn {
            index: 0,
            role: Role::User,
            timestamp: None,
            content: "Test".to_string(),
            actions: Vec::new(),
            tokens: None,
            thinking: None,
            is_sidechain: false,
        }];
        let session = make_session(turns);
        let chunks = chunk_session(&session, chrono_tz::Tz::UTC);
        assert!(chunks[0].context.contains("claude-code"));
        assert!(chunks[0].context.contains("testproj"));
    }

    #[test]
    fn test_overlap_in_long_turn() {
        // Long content that generates multiple chunks
        let content: String = (0..1000).map(|i| format!("word{} ", i)).collect();
        let turns = vec![Turn {
            index: 0,
            role: Role::User,
            timestamp: None,
            content,
            actions: Vec::new(),
            tokens: None,
            thinking: None,
            is_sidechain: false,
        }];
        let session = make_session(turns);
        let chunks = chunk_session(&session, chrono_tz::Tz::UTC);
        if chunks.len() > 1 {
            // Each chunk should have seq increasing
            for (i, chunk) in chunks.iter().enumerate() {
                assert_eq!(chunk.seq, i as u32);
            }
        }
    }
}
