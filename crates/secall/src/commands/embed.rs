use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use futures::stream::{self, StreamExt};
use secall_core::{
    search::vector::VectorIndexer,
    store::{get_default_db_path, Database, ReconcileOutcome},
    vault::Config,
};

/// Exit codes for a `secall embed` run, consumed by the daemon's embed-down
/// alert (log/scheduler.py mirrors these constants). The numeric values are a
/// cross-repo contract — keep them in sync, and avoid reserved codes: 1 (generic
/// anyhow error), 2 (clap usage), 126+ (shell/signal).
pub const EXIT_NO_BACKEND: u8 = 10;
pub const EXIT_RECONCILE_REFUSED: u8 = 11;
pub const EXIT_ALL_FAILED: u8 = 12;

/// Outcome of an embed pass, mapped to a process exit code by `main`.
///
/// Only the states the daemon must distinguish get a dedicated code; genuine
/// errors that propagate via `?` (db open, usage) stay on the anyhow path →
/// exit 1, so "backend down" is never conflated with "db locked / panic".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedOutcome {
    /// All pending chunks embedded, or nothing to do.
    Ok,
    /// No embedding backend could be constructed (ORT runtime / model absent).
    NoBackend,
    /// The store holds a foreign model and the wipe was not authorized.
    ReconcileRefused,
    /// Sessions were attempted but every one failed to embed.
    AllFailed,
}

impl EmbedOutcome {
    pub fn exit_code(self) -> u8 {
        match self {
            Self::Ok => 0,
            Self::NoBackend => EXIT_NO_BACKEND,
            Self::ReconcileRefused => EXIT_RECONCILE_REFUSED,
            Self::AllFailed => EXIT_ALL_FAILED,
        }
    }
}

pub async fn run(
    all: bool,
    batch_size: Option<usize>,
    concurrency: usize,
    allow_model_switch: bool,
) -> Result<EmbedOutcome> {
    let config = Config::load_or_default();
    let db_path = get_default_db_path();
    let db = Database::open(&db_path)?;

    let vector_indexer = secall_core::search::vector::create_embedding_indexer(&config).await;
    let Some(indexer) = vector_indexer else {
        eprintln!("No embedding backend available.");
        eprintln!("  1. Download model: secall model download");
        eprintln!("  2. Check config: [embedding] section in config.toml");
        return Ok(EmbedOutcome::NoBackend);
    };

    let batch_size = batch_size.unwrap_or(32);
    let indexer = Arc::new(indexer.with_batch_size(batch_size));

    // Single-active-model invariant: if the store holds a previous model's
    // vectors, wipe them before embedding. Runs here — single-threaded, before
    // the pre-filter scan (which would otherwise treat old-model chunks as
    // "already embedded" and silently skip) and before the concurrent embed
    // loop (avoiding any wipe/insert race on the shared DB).
    // A wipe destroys every stored vector, so it is authorized only for a
    // deliberate model migration: the operator passed --allow-model-switch AND
    // this is the configured backend (never a degraded fallback, whose model
    // name differs and would otherwise trigger the wipe). Without authorization,
    // a foreign-model store makes reconcile refuse → we abort with a nonzero exit
    // (the daemon's embed-down alert fires) rather than silently corrupt or
    // regress the vector space.
    let allow_wipe = allow_model_switch && !indexer.is_fallback();
    match db.reconcile_vector_model(indexer.model_name(), allow_wipe)? {
        ReconcileOutcome::Wiped => {
            eprintln!(
                "Embedding model changed → cleared stale vectors; performing a full re-embed."
            );
        }
        ReconcileOutcome::Refused => {
            // Branch BOTH the diagnosis and the remedy: in the fallback case
            // --allow-model-switch is deliberately inert (a fallback can never
            // wipe), so pointing the operator at that flag would send them into a
            // no-op retry during exactly the incident this guard defends against.
            let (reason, remedy) = if indexer.is_fallback() {
                (
                    "the configured embedding backend failed to load and this run fell back \
                     to a different model",
                    "Repair the configured embedding backend (ONNX runtime + model load) so it \
                     loads, then re-run `secall embed` — a fallback model can never authorize a \
                     wipe.",
                )
            } else {
                (
                    "the vector store holds a different embedding model",
                    "Re-run `secall embed --allow-model-switch` to intentionally re-embed from \
                     scratch with the active model.",
                )
            };
            // Not an anyhow error: a distinct exit code lets the daemon alert
            // point at the right remedy instead of the generic "backend down".
            // Print the diagnosis to stderr so the alert's output tail carries it.
            eprintln!(
                "refusing to embed: {reason}. The store still holds the previous model's \
                 vectors; embedding the active model now would corrupt the single-model \
                 invariant. {remedy}"
            );
            return Ok(EmbedOutcome::ReconcileRefused);
        }
        ReconcileOutcome::Clean => {}
    }

    let model = indexer.model_name().to_string();
    let dimensions = indexer.dimensions();
    let tz = config.timezone();
    let candidate_ids: Vec<String> = db.list_all_session_ids()?;

    // Pre-filter pass — sessions whose chunks are all already embedded (or
    // whose every turn is chunker-skip) are dropped, so [i/N] progress reflects
    // actual work. Only ids are kept: holding every pending Session until its
    // turn in the embed pass grows with the backlog, and the worker reloads one
    // session at a time instead.
    //
    // `--all` skips the pre-filter — the worker deletes each session's vectors
    // before re-embedding it.
    let work: Vec<String> = if all {
        candidate_ids
    } else {
        let scan_start = Instant::now();
        let total_candidates = candidate_ids.len();
        eprintln!("Scanning {total_candidates} session(s) for pending chunks...");
        let mut pending: Vec<String> = Vec::new();
        for sid in candidate_ids {
            // A session that fails to load or check stays queued, so the embed
            // pass reports the failure.
            let needs_embedding = match db.get_session_for_embedding(&sid) {
                Ok(session) => indexer
                    .has_pending_chunks(&db, &session, tz)
                    .unwrap_or(true),
                Err(_) => true,
            };
            if needs_embedding {
                pending.push(sid);
            }
        }
        eprintln!(
            "  Scan: {} session(s) need embedding, {} skipped no-op (in {:.2}s)",
            pending.len(),
            total_candidates - pending.len(),
            scan_start.elapsed().as_secs_f64(),
        );
        pending
    };

    let outcome = if work.is_empty() {
        drop(indexer);
        println!("All sessions already embedded.");
        EmbedOutcome::Ok
    } else {
        embed_sessions(indexer, work, &db_path, all, tz, batch_size, concurrency).await
    };

    // The pass wrote vectors without touching the vector file, so the embedder
    // and the file are never in memory together. Until this sync, search
    // scores the run's new vectors from the database. A sync failure leaves
    // search on the BLOB scan, slower but complete, so it does not fail the run.
    #[cfg(not(target_os = "windows"))]
    sync_vector_file(&db, &model, dimensions);

    Ok(outcome)
}

/// Embed `work`'s pending chunks, `concurrency` sessions at a time. Takes the
/// indexer by value so the embedder is released when the pass ends.
async fn embed_sessions(
    indexer: Arc<VectorIndexer>,
    work: Vec<String>,
    db_path: &std::path::Path,
    all: bool,
    tz: chrono_tz::Tz,
    batch_size: usize,
    concurrency: usize,
) -> EmbedOutcome {
    let total = work.len();
    eprintln!(
        "Embedding {} session(s) [batch_size={}, concurrency={}]...",
        total, batch_size, concurrency
    );
    let db_path: Arc<PathBuf> = Arc::new(db_path.to_path_buf());
    let counter = Arc::new(AtomicUsize::new(0));
    // AllFailed keys on embedding-level outcomes only: `embedded` counts sessions
    // that actually wrote chunks (chunks_embedded > 0), `embed_failed` counts
    // index_session errors. Infra errors (db open / load / delete) and chunk-empty
    // no-ops touch neither — they must not mask nor fabricate an all-failed verdict.
    let embedded = Arc::new(AtomicUsize::new(0));
    let embed_failed = Arc::new(AtomicUsize::new(0));
    let total_chunks = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    stream::iter(work)
        .map(|sid| {
            let indexer = Arc::clone(&indexer);
            let db_path = Arc::clone(&db_path);
            let counter = Arc::clone(&counter);
            let embedded = Arc::clone(&embedded);
            let embed_failed = Arc::clone(&embed_failed);
            let total_chunks = Arc::clone(&total_chunks);
            async move {
                let short = &sid[..sid.len().min(8)];
                let db = match Database::open(db_path.as_path()) {
                    Ok(d) => d,
                    Err(e) => {
                        let i = counter.fetch_add(1, Ordering::Relaxed) + 1;
                        eprintln!("  [{i}/{total}] {short} — db open failed: {e}");
                        return;
                    }
                };
                if all {
                    if let Err(e) = db.delete_session_vectors(&sid) {
                        let i = counter.fetch_add(1, Ordering::Relaxed) + 1;
                        eprintln!("  [{i}/{total}] {short} — delete-before-rebuild failed: {e}");
                        return;
                    }
                }
                let session = match db.get_session_for_embedding(&sid) {
                    Ok(s) => s,
                    Err(e) => {
                        let i = counter.fetch_add(1, Ordering::Relaxed) + 1;
                        eprintln!("  [{i}/{total}] {short} — load failed: {e}");
                        return;
                    }
                };
                match indexer.index_session(&db, &session, tz).await {
                    Ok(stats) => {
                        // Only count a genuine embed; a chunk-empty no-op
                        // (Ok(default)) is neither success nor failure.
                        if stats.chunks_embedded > 0 {
                            embedded.fetch_add(1, Ordering::Relaxed);
                        }
                        let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
                        let chunks_done = total_chunks
                            .fetch_add(stats.chunks_embedded, Ordering::Relaxed)
                            + stats.chunks_embedded;
                        let elapsed = start.elapsed().as_secs_f64();
                        let rate = if elapsed > 0.0 {
                            chunks_done as f64 / elapsed
                        } else {
                            0.0
                        };
                        let remaining = total - done;
                        let eta_secs = if done > 0 && elapsed > 0.0 {
                            remaining as f64 / (done as f64 / elapsed)
                        } else {
                            0.0
                        };
                        let eta_min = (eta_secs / 60.0).ceil() as u64;
                        eprintln!(
                            "  [{done}/{total}] {short} — {} chunks ({:.1} chunks/s, ETA ~{eta_min}m)",
                            stats.chunks_embedded,
                            rate,
                        );
                    }
                    Err(e) => {
                        embed_failed.fetch_add(1, Ordering::Relaxed);
                        let i = counter.fetch_add(1, Ordering::Relaxed) + 1;
                        eprintln!("  [{i}/{total}] {short} — embedding failed: {e}");
                    }
                }
            }
        })
        .buffer_unordered(concurrency)
        .collect::<()>()
        .await;

    let elapsed = start.elapsed();
    let mins = elapsed.as_secs() / 60;
    let secs = elapsed.as_secs() % 60;
    let total_c = total_chunks.load(Ordering::Relaxed);
    let embedded_n = embedded.load(Ordering::Relaxed);
    let failed_n = embed_failed.load(Ordering::Relaxed);
    eprintln!(
        "\nDone: {}/{} sessions embedded, {} chunks in {}m {}s ({:.1} chunks/s)",
        embedded_n,
        total,
        total_c,
        mins,
        secs,
        total_c as f64 / elapsed.as_secs_f64().max(0.001),
    );

    // Backend loaded but every session that reached the embedder failed (dylib
    // drift, OOM, per-session ORT errors) — the silent-degradation mode the daemon
    // alert exists for. Gate on embedding-level outcomes only: at least one real
    // failure AND zero genuine embeds. A single success, an infra-only failure
    // (db/load), or an all-no-op run does not qualify.
    if failed_n > 0 && embedded_n == 0 {
        return EmbedOutcome::AllFailed;
    }
    EmbedOutcome::Ok
}

/// Bring the vector file in line with the stored vectors.
#[cfg(not(target_os = "windows"))]
fn sync_vector_file(db: &Database, model: &str, dimensions: usize) {
    use secall_core::search::vector_file::{self, vector_file_path, VectorFileSync};

    if dimensions == 0 {
        return;
    }
    let path = vector_file_path(model, dimensions);
    let start = Instant::now();
    match vector_file::sync_vector_file(db, model, dimensions, &path) {
        Ok(VectorFileSync::Unchanged { vectors }) => {
            eprintln!("Vector file up to date ({vectors} vectors).");
        }
        Ok(VectorFileSync::Updated {
            vectors,
            added,
            removed,
        }) => eprintln!(
            "Vector file updated: {added} added, {removed} removed, {vectors} vectors ({:.1}s).",
            start.elapsed().as_secs_f64()
        ),
        Ok(VectorFileSync::Rebuilt { vectors }) => eprintln!(
            "Vector file rebuilt: {vectors} vectors ({:.1}s).",
            start.elapsed().as_secs_f64()
        ),
        Err(e) => eprintln!("Warning: vector file sync failed; search uses the BLOB scan: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_are_stable_and_distinct() {
        // Cross-repo contract with log/scheduler.py — changing these silently
        // breaks the daemon's embed-down alert branching.
        assert_eq!(EmbedOutcome::Ok.exit_code(), 0);
        assert_eq!(EmbedOutcome::NoBackend.exit_code(), 10);
        assert_eq!(EmbedOutcome::ReconcileRefused.exit_code(), 11);
        assert_eq!(EmbedOutcome::AllFailed.exit_code(), 12);
    }

    #[test]
    fn nonzero_codes_avoid_reserved_values() {
        // 1 = generic anyhow error, 2 = clap usage, 126+ = shell/signal.
        for code in [EXIT_NO_BACKEND, EXIT_RECONCILE_REFUSED, EXIT_ALL_FAILED] {
            assert!(code > 2 && code < 126, "reserved exit code {code}");
        }
    }
}
