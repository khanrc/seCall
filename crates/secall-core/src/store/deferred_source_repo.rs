//! Deferred inputs belong to source paths, never to a parent session's identity.
use super::Database;
use rusqlite::OptionalExtension;
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};

pub const CREATE_DEFERRED_SOURCES: &str = "
CREATE TABLE IF NOT EXISTS deferred_sources (
    path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    parser_revision TEXT NOT NULL,
    reason TEXT NOT NULL
);";

pub struct SourceFingerprint {
    path: String,
    content_hash: String,
    parser_revision: String,
}

impl SourceFingerprint {
    pub fn read(path: &Path, parser_revision: &str) -> std::io::Result<Self> {
        let canonical = path.canonicalize()?;
        let mut file = std::fs::File::open(&canonical)?;
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
        }
        Ok(Self {
            path: canonical.to_string_lossy().into_owned(),
            content_hash: format!("{:x}", hash.finalize()),
            parser_revision: parser_revision.to_owned(),
        })
    }

    pub fn still_matches(&self, path: &Path) -> std::io::Result<bool> {
        Ok(Self::read(path, &self.parser_revision)?.content_hash == self.content_hash)
    }
}

impl Database {
    pub fn is_deferred_source(&self, source: &SourceFingerprint) -> crate::error::Result<bool> {
        let stored = self
            .conn()
            .query_row(
                "SELECT content_hash,parser_revision FROM deferred_sources WHERE path=?1",
                [&source.path],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
            )
            .optional()?;
        Ok(stored == Some((source.content_hash.clone(), source.parser_revision.clone())))
    }

    pub fn defer_source(
        &self,
        source: &SourceFingerprint,
        reason: &str,
    ) -> crate::error::Result<()> {
        self.conn().execute(
            "INSERT INTO deferred_sources(path,content_hash,parser_revision,reason) VALUES (?1,?2,?3,?4)
             ON CONFLICT(path) DO UPDATE SET content_hash=excluded.content_hash,parser_revision=excluded.parser_revision,reason=excluded.reason",
            rusqlite::params![source.path,source.content_hash,source.parser_revision,reason])?;
        Ok(())
    }

    pub fn clear_deferred_source(&self, source: &SourceFingerprint) -> crate::error::Result<()> {
        self.conn()
            .execute("DELETE FROM deferred_sources WHERE path=?1", [&source.path])?;
        Ok(())
    }
}
