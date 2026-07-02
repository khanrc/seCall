use std::sync::OnceLock;

use crate::ingest::Session;

// We split on the model tokenizer so a chunk never overflows the model's
// max_seq and gets silently truncated (the char-based 3600 cap did exactly that
// for Korean — ~2 chars/token — losing the tail of long turns). The budget is
// derived from the model's own model_max_length minus RESERVE_TOKENS, since the
// embedded sequence is `<e5 prefix> + chunk + 2 special tokens`: ~4 prefix + 2
// specials + margin. So an e5 512-model → 500, bge-m3 8192 → 8180, automatically.
// `embedding.max_chunk_tokens` overrides; DEFAULT is the last-resort fallback
// when model_max_length can't be read. Overlap is ~15%, overridable too.
const RESERVE_TOKENS: usize = 12;
const DEFAULT_MAX_CHUNK_TOKENS: usize = 500;
const DEFAULT_OVERLAP_TOKENS: usize = 75;

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
    let cfg = chunker_config();
    if let Some(tok) = &cfg.tokenizer {
        return split_by_tokens(text, tok, cfg.max_tokens, cfg.overlap);
    }
    split_into_chunks(text, FALLBACK_MAX_CHUNK_CHARS, FALLBACK_OVERLAP_CHARS)
}

struct ChunkerConfig {
    /// The embedding model's tokenizer (`None` → char fallback: model not on
    /// disk, e.g. tests / fresh install).
    tokenizer: Option<tokenizers::Tokenizer>,
    max_tokens: usize,
    overlap: usize,
}

/// Lazily resolve the chunker's config once. The tokenizer is loaded from the
/// same path the embedder uses (`config.embedding.model_path`, else default) so
/// token counts match what the model is actually fed; truncation is forced off
/// so a tokenizer that shipped a truncation setting can't silently cap `encode`
/// and hide the tail-loss this chunker exists to prevent. The token budget /
/// overlap come from config (`embedding.max_chunk_tokens` / `overlap_tokens`),
/// defaulting to the e5-512-safe values.
fn chunker_config() -> &'static ChunkerConfig {
    static CFG: OnceLock<ChunkerConfig> = OnceLock::new();
    CFG.get_or_init(|| {
        let cfg = crate::vault::Config::load_or_default();
        let model_dir = cfg
            .embedding
            .model_path
            .clone()
            .unwrap_or_else(crate::search::model_manager::default_model_path);
        let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .ok()
            .map(|mut t| {
                t.with_truncation(None).ok();
                t
            });
        // Budget: explicit override → else model_max_length − reserve → else the
        // fallback default. Keeps the chunk within the model's real limit
        // automatically, no per-model hardcoding.
        let max_tokens = cfg
            .embedding
            .max_chunk_tokens
            .or_else(|| {
                crate::search::model_manager::read_model_max_length(&model_dir)
                    .map(|m| m.saturating_sub(RESERVE_TOKENS))
            })
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_MAX_CHUNK_TOKENS);
        ChunkerConfig {
            tokenizer,
            max_tokens,
            overlap: cfg.embedding.overlap_tokens.unwrap_or(DEFAULT_OVERLAP_TOKENS),
        }
    })
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

    #[test]
    fn test_split_by_tokens_windows_within_budget_with_overlap() {
        use std::collections::HashMap;
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;
        use tokenizers::Tokenizer;

        // Trivial word-level tokenizer: one token per whitespace-delimited word,
        // so token counts are deterministic (no on-disk model needed).
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for i in 0..40u32 {
            vocab.insert(format!("w{i}"), i);
        }
        vocab.insert("[UNK]".to_string(), 40);
        let model = WordLevel::builder()
            .vocab(vocab.into_iter().collect()) // infers the builder's AHashMap
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(Whitespace::default()));

        let text = (0..20).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        let chunks = split_by_tokens(&text, &tok, 5, 2);

        assert!(chunks.len() > 1, "20 tokens / budget 5 → multiple chunks");
        for c in &chunks {
            let n = tok.encode(c.as_str(), false).unwrap().get_ids().len();
            assert!(n <= 5, "chunk exceeds token budget: {n} tokens in {c:?}");
        }
        // First and last original tokens are covered (no head/tail loss).
        assert!(chunks[0].contains("w0"));
        assert!(chunks.last().unwrap().contains("w19"));
    }
}
