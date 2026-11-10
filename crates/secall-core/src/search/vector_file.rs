//! Flat file of one embedding model's `turn_vectors`, scanned exhaustively for
//! the nearest neighbors of a query.
//!
//! Only `secall embed` writes the file: after its embedding pass,
//! [`sync_vector_file`] rewrites it to hold exactly the stored vectors and
//! replaces it by rename. Search processes hold a [`VectorFileReader`], which
//! memory-maps the file and maps it again when it is replaced, so the vectors
//! sit in the page cache instead of every search process's heap.
//!
//! Search compares the query with every vector. The HNSW graph this file
//! replaced found about half of the true top 10 on the deployment's 861k
//! vectors (many exact duplicates, e5 query embeddings against passage
//! embeddings); the exhaustive pass is exact at about 0.1s per query.
//!
//! Layout, little-endian:
//! - header, 64 bytes: magic `SECALLV1`, dimensions (u32), zero (u32),
//!   vector count (u64), zero padding
//! - vectors: `count × dimensions` f32, each scaled to unit length so a dot
//!   product is the cosine similarity
//! - keys: `count` u64 vector ids, strictly ascending, one per vector in the
//!   same order

#[cfg(not(target_os = "windows"))]
mod inner {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;
    use std::fs::{self, File};
    use std::io::{BufWriter, Seek, SeekFrom, Write};
    use std::path::{Path, PathBuf};
    use std::sync::RwLock;
    use std::time::{Duration, SystemTime};

    use anyhow::{bail, Result};
    use memmap2::Mmap;

    use crate::store::db::Database;

    // The mapped bytes are read in place as f32 and written straight from f32.
    const _: () = assert!(
        cfg!(target_endian = "little"),
        "the vector file format is little-endian"
    );

    const MAGIC: &[u8; 8] = b"SECALLV1";
    const HEADER_LEN: usize = 64;
    /// Rows read per database page while writing the file.
    const PAGE: usize = 8192;
    /// A temporary file this old belongs to a run that died while writing.
    const STALE_TEMP_AGE: Duration = Duration::from_secs(3600);
    /// Threads one search splits the scan across: enough to keep a query near
    /// 0.1s, few enough to leave cores for the host's other work.
    const MAX_SCAN_THREADS: usize = 4;
    /// Below this many vectors per thread, a thread costs more than it saves.
    const MIN_VECTORS_PER_THREAD: usize = 65_536;

    /// Where the vector file for an embedding model lives: next to the index
    /// database it mirrors, so a database elsewhere (`SECALL_DB_PATH`) gets its
    /// own file. Keyed by model and dimensions, so a model switch starts a new
    /// file.
    pub fn vector_file_path(model_name: &str, dimensions: usize) -> PathBuf {
        let model = model_name.replace(['/', ':'], "_");
        crate::store::get_default_db_path()
            .with_file_name(format!("vectors_{model}_{dimensions}.f32"))
    }

    /// `vector` scaled to unit length. A zero vector stays zero, so every
    /// distance to it is 1, as in the database scan.
    fn unit(vector: &[f32]) -> Vec<f32> {
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return vec![0.0; vector.len()];
        }
        vector.iter().map(|x| x / norm).collect()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        // Eight independent sums let the compiler vectorize the loop; one
        // running sum is a dependency chain it has to keep in order.
        let mut sums = [0.0f32; 8];
        let a_chunks = a.chunks_exact(8);
        let b_chunks = b.chunks_exact(8);
        let rest: f32 = a_chunks
            .remainder()
            .iter()
            .zip(b_chunks.remainder())
            .map(|(x, y)| x * y)
            .sum();
        for (x, y) in a_chunks.zip(b_chunks) {
            for i in 0..8 {
                sums[i] += x[i] * y[i];
            }
        }
        sums.iter().sum::<f32>() + rest
    }

    fn as_bytes(floats: &[f32]) -> &[u8] {
        // SAFETY: f32 has no padding bytes, and u8 has no alignment requirement.
        unsafe {
            std::slice::from_raw_parts(floats.as_ptr().cast::<u8>(), std::mem::size_of_val(floats))
        }
    }

    /// A vector file mapped read-only.
    struct VectorFile {
        map: Mmap,
        dimensions: usize,
        count: usize,
    }

    impl VectorFile {
        fn open(path: &Path, dimensions: usize) -> Result<Self> {
            let file = File::open(path)?;
            // SAFETY: a published file is never modified. Writers write a
            // temporary file and rename it over the path, which leaves the
            // mapped inode as it was for as long as this map holds it.
            let map = unsafe { Mmap::map(&file)? };
            if map.len() < HEADER_LEN || &map[..MAGIC.len()] != MAGIC {
                bail!("{} is not a seCall vector file", path.display());
            }
            let stored = u32::from_le_bytes(map[8..12].try_into()?) as usize;
            if stored != dimensions {
                bail!(
                    "vector file {} has {stored} dimensions, expected {dimensions}",
                    path.display()
                );
            }
            let count = usize::try_from(u64::from_le_bytes(map[16..24].try_into()?))?;
            let expected_len = count
                .checked_mul(dimensions * 4 + 8)
                .and_then(|body| body.checked_add(HEADER_LEN));
            if expected_len != Some(map.len()) {
                bail!("vector file {} is truncated or corrupt", path.display());
            }
            let file = Self {
                map,
                dimensions,
                count,
            };
            if (1..count).any(|i| file.key(i - 1) >= file.key(i)) {
                bail!("vector file {} has unordered keys", path.display());
            }
            Ok(file)
        }

        fn vectors(&self) -> &[f32] {
            let bytes = &self.map[HEADER_LEN..HEADER_LEN + self.count * self.dimensions * 4];
            // SAFETY: every bit pattern is a valid f32. The map starts on a page
            // boundary and the header is 64 bytes, so the slice is aligned; the
            // assert below holds that rather than trusting it.
            let (head, floats, tail) = unsafe { bytes.align_to::<f32>() };
            assert!(head.is_empty() && tail.is_empty(), "misaligned vector data");
            floats
        }

        fn vector(&self, position: usize) -> &[f32] {
            &self.vectors()[position * self.dimensions..(position + 1) * self.dimensions]
        }

        fn key(&self, position: usize) -> u64 {
            let at = HEADER_LEN + self.count * self.dimensions * 4 + position * 8;
            u64::from_le_bytes(self.map[at..at + 8].try_into().expect("8-byte key"))
        }

        /// The highest key, or 0 when the file is empty.
        fn newest_key(&self) -> u64 {
            self.count.checked_sub(1).map_or(0, |last| self.key(last))
        }

        /// The `limit` vectors most similar to the unit-length `query`, as
        /// `(key, cosine distance)`, nearest first.
        fn nearest(&self, query: &[f32], limit: usize) -> Vec<(u64, f32)> {
            if limit == 0 || self.count == 0 {
                return Vec::new();
            }
            let threads = std::thread::available_parallelism()
                .map_or(1, |n| n.get())
                .min(MAX_SCAN_THREADS)
                .min(self.count.div_ceil(MIN_VECTORS_PER_THREAD));
            let per_thread = self.count.div_ceil(threads);
            let (vectors, dimensions) = (self.vectors(), self.dimensions);
            let mut best: Vec<Scored> = std::thread::scope(|scope| {
                let workers: Vec<_> = (0..threads)
                    .map(|t| {
                        let start = t * per_thread;
                        let end = (start + per_thread).min(self.count);
                        scope.spawn(move || {
                            top_scores(query, vectors, dimensions, start..end, limit)
                        })
                    })
                    .collect();
                workers
                    .into_iter()
                    .flat_map(|worker| worker.join().expect("vector scan thread panicked"))
                    .collect()
            });
            best.sort_by(|a, b| b.score.total_cmp(&a.score));
            best.truncate(limit);
            best.into_iter()
                .map(|s| (self.key(s.position), 1.0 - s.score))
                .collect()
        }
    }

    /// A vector's similarity to the query. Ordered so a max-heap keeps the
    /// least similar on top, ready to be evicted.
    #[derive(Clone, Copy)]
    struct Scored {
        score: f32,
        position: usize,
    }

    impl Ord for Scored {
        fn cmp(&self, other: &Self) -> Ordering {
            other.score.total_cmp(&self.score)
        }
    }

    impl PartialOrd for Scored {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl PartialEq for Scored {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == Ordering::Equal
        }
    }

    impl Eq for Scored {}

    fn top_scores(
        query: &[f32],
        vectors: &[f32],
        dimensions: usize,
        positions: std::ops::Range<usize>,
        limit: usize,
    ) -> Vec<Scored> {
        let mut heap = BinaryHeap::with_capacity(limit + 1);
        for position in positions {
            let vector = &vectors[position * dimensions..(position + 1) * dimensions];
            let candidate = Scored {
                score: dot(query, vector),
                position,
            };
            if heap.len() < limit {
                heap.push(candidate);
            } else if let Some(mut least) = heap.peek_mut() {
                if candidate.score > least.score {
                    *least = candidate;
                }
            }
        }
        heap.into_vec()
    }

    /// Writes a vector file. Keys must arrive in ascending order.
    struct Writer {
        out: BufWriter<File>,
        dimensions: usize,
        keys: Vec<u64>,
    }

    impl Writer {
        fn create(path: &Path, dimensions: usize) -> Result<Self> {
            let mut out = BufWriter::with_capacity(1 << 20, File::create(path)?);
            let mut header = [0u8; HEADER_LEN];
            header[..MAGIC.len()].copy_from_slice(MAGIC);
            header[8..12].copy_from_slice(&u32::try_from(dimensions)?.to_le_bytes());
            out.write_all(&header)?;
            Ok(Self {
                out,
                dimensions,
                keys: Vec::new(),
            })
        }

        fn push(&mut self, key: u64, unit_vector: &[f32]) -> Result<()> {
            debug_assert_eq!(unit_vector.len(), self.dimensions);
            debug_assert!(self.keys.last().map_or(true, |&last| last < key));
            self.out.write_all(as_bytes(unit_vector))?;
            self.keys.push(key);
            Ok(())
        }

        /// Append the keys, record the count in the header, and return it.
        fn finish(mut self) -> Result<usize> {
            for key in &self.keys {
                self.out.write_all(&key.to_le_bytes())?;
            }
            let mut file = self.out.into_inner().map_err(|e| e.into_error())?;
            file.seek(SeekFrom::Start(16))?;
            file.write_all(&(self.keys.len() as u64).to_le_bytes())?;
            Ok(self.keys.len())
        }
    }

    /// The temporary name carries the pid: nothing stops two embed runs from
    /// overlapping, and a shared name would let one rename the other's
    /// half-written file into place.
    fn temp_path(path: &Path) -> PathBuf {
        let mut name = path.file_name().unwrap_or_default().to_os_string();
        name.push(format!(".{}.tmp", std::process::id()));
        path.with_file_name(name)
    }

    /// Remove temporary files that runs killed mid-write left next to `path`.
    /// Each carries its writer's pid, so no later write overwrites them.
    fn remove_stale_temps(path: &Path) {
        let (Some(dir), Some(name)) = (path.parent(), path.file_name().and_then(|n| n.to_str()))
        else {
            return;
        };
        let prefix = format!("{name}.");
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let file = entry.file_name();
            let Some(file) = file.to_str() else {
                continue;
            };
            if !file.starts_with(&prefix) || !file.ends_with(".tmp") {
                continue;
            }
            let stale = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|modified| modified.elapsed().ok())
                .is_some_and(|age| age > STALE_TEMP_AGE);
            if stale {
                let _ = fs::remove_file(entry.path());
            }
        }
    }

    /// Identity of the file on disk. A rename gives it a new inode, so a
    /// replacement is seen even within the same mtime second.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Stamp {
        len: u64,
        modified: Option<SystemTime>,
        #[cfg(unix)]
        inode: u64,
    }

    fn stamp(path: &Path) -> Option<Stamp> {
        let meta = fs::metadata(path).ok()?;
        Some(Stamp {
            len: meta.len(),
            modified: meta.modified().ok(),
            #[cfg(unix)]
            inode: std::os::unix::fs::MetadataExt::ino(&meta),
        })
    }

    struct Mapped {
        stamp: Option<Stamp>,
        file: Option<VectorFile>,
    }

    /// The nearest vectors the file holds.
    pub struct FileHits {
        /// `(vector id, cosine distance)`, nearest first.
        pub hits: Vec<(u64, f32)>,
        /// The highest vector id in the file, 0 when it is empty. Vector ids
        /// only grow, so vectors with a higher id were stored after the file
        /// was written and are not among `hits`.
        pub newest_key: u64,
    }

    /// Read-only, memory-mapped vector file for search processes.
    pub struct VectorFileReader {
        path: PathBuf,
        dimensions: usize,
        mapped: RwLock<Mapped>,
    }

    impl VectorFileReader {
        /// Never fails: a missing or unreadable file only means no file search
        /// until a usable file appears at `path`.
        pub fn open(path: &Path, dimensions: usize) -> Self {
            let reader = Self {
                path: path.to_path_buf(),
                dimensions,
                mapped: RwLock::new(Mapped {
                    stamp: None,
                    file: None,
                }),
            };
            reader.refresh();
            reader
        }

        fn refresh(&self) {
            let current = stamp(&self.path);
            {
                let mapped = self.mapped.read().unwrap_or_else(|e| e.into_inner());
                if mapped.stamp == current {
                    return;
                }
            }
            let file = current.and_then(|_| match VectorFile::open(&self.path, self.dimensions) {
                Ok(file) => {
                    tracing::info!(
                        path = %self.path.display(),
                        vectors = file.count,
                        "vector file mapped"
                    );
                    Some(file)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "vector file unavailable, using BLOB scan");
                    None
                }
            });
            let mut mapped = self.mapped.write().unwrap_or_else(|e| e.into_inner());
            *mapped = Mapped {
                stamp: current,
                file,
            };
        }

        /// The `limit` vectors in the file nearest to `query`, or `None` when
        /// no usable file is mapped. Keys of vectors deleted since the file
        /// was written are still returned; skipping them is the caller's job.
        pub fn search(&self, query: &[f32], limit: usize) -> Option<FileHits> {
            if query.len() != self.dimensions {
                tracing::warn!(
                    query = query.len(),
                    file = self.dimensions,
                    "query dimensions differ from the vector file"
                );
                return None;
            }
            self.refresh();
            let mapped = self.mapped.read().unwrap_or_else(|e| e.into_inner());
            let file = mapped.file.as_ref()?;
            Some(FileHits {
                hits: file.nearest(&unit(query), limit),
                newest_key: file.newest_key(),
            })
        }
    }

    /// What [`sync_vector_file`] did to the file.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum VectorFileSync {
        /// The file already held exactly the stored vectors; it is untouched.
        Unchanged { vectors: usize },
        /// Rewritten from the previous file, without the deleted vectors and
        /// with the new ones read from the database.
        Updated {
            vectors: usize,
            added: usize,
            removed: usize,
        },
        /// Written from the database alone, for want of a usable file.
        Rebuilt { vectors: usize },
    }

    /// Bring the file at `path` in line with `model`'s rows in `turn_vectors`.
    ///
    /// Vectors the current file holds are copied from it; only the others are
    /// read from the database. The new file replaces the old one by rename.
    pub fn sync_vector_file(
        db: &Database,
        model: &str,
        dimensions: usize,
        path: &Path,
    ) -> Result<VectorFileSync> {
        remove_stale_temps(path);
        let ids = db.vector_ids(model)?;
        let existing = if path.exists() {
            match VectorFile::open(path, dimensions) {
                Ok(file) => Some(file),
                Err(e) => {
                    tracing::warn!(error = %e, "vector file unreadable, rebuilding");
                    None
                }
            }
        } else {
            None
        };
        if let Some(file) = &existing {
            let same = file.count == ids.len()
                && ids
                    .iter()
                    .enumerate()
                    .all(|(position, &id)| file.key(position) == id as u64);
            if same {
                return Ok(VectorFileSync::Unchanged {
                    vectors: file.count,
                });
            }
        }

        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let tmp = temp_path(path);
        let written = match write(db, model, dimensions, &ids, existing.as_ref(), &tmp) {
            Ok(written) => written,
            Err(e) => {
                let _ = fs::remove_file(&tmp);
                return Err(e);
            }
        };
        let rebuilt = existing.is_none();
        drop(existing);
        fs::rename(&tmp, path)?;
        Ok(if rebuilt {
            VectorFileSync::Rebuilt {
                vectors: written.vectors,
            }
        } else {
            VectorFileSync::Updated {
                vectors: written.vectors,
                added: written.added,
                removed: written.removed,
            }
        })
    }

    struct Written {
        vectors: usize,
        added: usize,
        removed: usize,
    }

    fn write(
        db: &Database,
        model: &str,
        dimensions: usize,
        ids: &[i64],
        existing: Option<&VectorFile>,
        tmp: &Path,
    ) -> Result<Written> {
        let mut writer = Writer::create(tmp, dimensions)?;
        let (mut added, mut removed) = (0, 0);
        // Ids up to `copied_through` come from the existing file, or by id
        // from the database when the file lacks them.
        let mut copied_through = 0i64;
        if let Some(file) = existing {
            // Only through the newest id both snapshots cover: an overlapping
            // sync can publish `existing` after `ids` was read, and the ids it
            // holds past `ids` come from the database tail below.
            copied_through = (file.newest_key() as i64).min(ids.last().copied().unwrap_or(0));
            let mut position = 0;
            for &id in ids.iter().take_while(|&&id| id <= copied_through) {
                while position < file.count && (file.key(position) as i64) < id {
                    removed += 1;
                    position += 1;
                }
                if position < file.count && file.key(position) as i64 == id {
                    writer.push(id as u64, file.vector(position))?;
                    position += 1;
                } else if let Some(vector) = db
                    .vector_by_id(id)?
                    .filter(|vector| vector.len() == dimensions)
                {
                    writer.push(id as u64, &unit(&vector))?;
                    added += 1;
                }
            }
            removed += file.count - position;
        }
        // Everything newer, in id order. Pages may reach past `ids` while
        // another embed run inserts; those vectors belong in the file too.
        let mut after = copied_through;
        loop {
            let page = db.vectors_after(model, after, PAGE)?;
            let Some(&(last, _)) = page.last() else {
                break;
            };
            after = last;
            for (id, vector) in page {
                // A vector of another dimension cannot be compared with the
                // query; the database scan scores it as unrelated too.
                if vector.len() == dimensions {
                    writer.push(id as u64, &unit(&vector))?;
                    added += 1;
                }
            }
        }
        let vectors = writer.finish()?;
        Ok(Written {
            vectors,
            added,
            removed,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::store::VectorRepo;
        use tempfile::tempdir;

        const DIM: usize = 3;

        fn db_with(vectors: &[[f32; DIM]]) -> (Database, Vec<i64>) {
            let db = Database::open_memory().unwrap();
            db.init_vector_table().unwrap();
            let ids = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    db.insert_vector(v, &format!("s{i}"), 0, 0, "m")
                        .unwrap()
                        .unwrap()
                })
                .collect();
            (db, ids)
        }

        fn keys(path: &Path) -> Vec<i64> {
            let file = VectorFile::open(path, DIM).unwrap();
            (0..file.count).map(|i| file.key(i) as i64).collect()
        }

        #[test]
        fn sync_builds_then_leaves_a_matching_file_alone() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("vectors.f32");
            let (db, ids) = db_with(&[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);

            assert_eq!(
                sync_vector_file(&db, "m", DIM, &path).unwrap(),
                VectorFileSync::Rebuilt { vectors: 2 }
            );
            assert_eq!(keys(&path), ids);
            let file = VectorFile::open(&path, DIM).unwrap();
            assert_eq!(file.vector(1), &[0.0, 1.0, 0.0], "stored at unit length");

            let before = stamp(&path);
            assert_eq!(
                sync_vector_file(&db, "m", DIM, &path).unwrap(),
                VectorFileSync::Unchanged { vectors: 2 }
            );
            assert_eq!(stamp(&path), before, "an unchanged file is not rewritten");
        }

        #[test]
        fn sync_drops_deleted_vectors_and_adds_new_ones() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("vectors.f32");
            let many: Vec<[f32; DIM]> = (0..20).map(|i| [1.0, i as f32, 0.0]).collect();
            let (db, ids) = db_with(&many);
            sync_vector_file(&db, "m", DIM, &path).unwrap();

            db.delete_session_vectors("s0").unwrap();
            db.delete_session_vectors("s7").unwrap();
            let new_id = db
                .insert_vector(&[0.0, 0.0, 1.0], "s-new", 0, 0, "m")
                .unwrap()
                .unwrap();

            assert_eq!(
                sync_vector_file(&db, "m", DIM, &path).unwrap(),
                VectorFileSync::Updated {
                    vectors: 19,
                    added: 1,
                    removed: 2
                }
            );
            let mut expected: Vec<i64> = ids
                .iter()
                .copied()
                .filter(|&id| id != ids[0] && id != ids[7])
                .collect();
            expected.push(new_id);
            assert_eq!(keys(&path), expected);
            let file = VectorFile::open(&path, DIM).unwrap();
            assert_eq!(file.vector(18), &[0.0, 0.0, 1.0]);
        }

        #[test]
        fn sync_keeps_vectors_a_newer_file_holds_past_the_id_snapshot() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("vectors.f32");
            let (db, mut ids) = db_with(&[[0.0, 1.0, 0.0]]);
            let snapshot = db.vector_ids("m").unwrap();
            // Another sync publishes a file after this one read its ids, then
            // another vector arrives before this one reads the tail.
            ids.push(
                db.insert_vector(&[1.0, 0.0, 0.0], "s1", 0, 0, "m")
                    .unwrap()
                    .unwrap(),
            );
            sync_vector_file(&db, "m", DIM, &path).unwrap();
            ids.push(
                db.insert_vector(&[-1.0, 0.0, 0.0], "s2", 0, 0, "m")
                    .unwrap()
                    .unwrap(),
            );

            let newer = VectorFile::open(&path, DIM).unwrap();
            let tmp = dir.path().join("vectors.tmp");
            write(&db, "m", DIM, &snapshot, Some(&newer), &tmp).unwrap();
            assert_eq!(keys(&tmp), ids);
        }

        #[test]
        fn sync_rebuilds_a_file_of_other_dimensions() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("vectors.f32");
            let (db, ids) = db_with(&[[1.0, 0.0, 0.0]]);
            fs::write(&path, b"not a vector file").unwrap();

            assert_eq!(
                sync_vector_file(&db, "m", DIM, &path).unwrap(),
                VectorFileSync::Rebuilt { vectors: 1 }
            );
            assert_eq!(keys(&path), ids);
            assert!(VectorFile::open(&path, DIM + 1).is_err());
        }

        #[test]
        fn reader_maps_a_file_that_appears_and_follows_its_replacement() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("vectors.f32");
            let reader = VectorFileReader::open(&path, DIM);
            assert!(reader.search(&[1.0, 0.0, 0.0], 1).is_none());

            let (db, ids) = db_with(&[[1.0, 0.0, 0.0]]);
            sync_vector_file(&db, "m", DIM, &path).unwrap();
            let found = reader.search(&[1.0, 0.0, 0.0], 1).unwrap();
            assert_eq!(found.hits[0].0, ids[0] as u64);
            assert_eq!(found.newest_key, ids[0] as u64);

            let new_id = db
                .insert_vector(&[0.0, 1.0, 0.0], "s-new", 0, 0, "m")
                .unwrap()
                .unwrap() as u64;
            let found = reader.search(&[0.0, 1.0, 0.0], 1).unwrap();
            assert_ne!(found.hits[0].0, new_id, "not in the file until a sync");
            assert!(found.newest_key < new_id);

            sync_vector_file(&db, "m", DIM, &path).unwrap();
            let found = reader.search(&[0.0, 1.0, 0.0], 1).unwrap();
            assert_eq!(found.hits[0].0, new_id);
            assert_eq!(found.newest_key, new_id);
        }

        #[test]
        fn reader_returns_the_true_nearest_neighbors() {
            // Enough vectors for the scan to split across threads.
            const D: usize = 16;
            let mut state = 0x9e37_79b9_7f4a_7c15_u64;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            };
            let mut random_vector = || (0..D).map(|_| next()).collect::<Vec<f32>>();
            let vectors: Vec<Vec<f32>> = (0..150_000).map(|_| random_vector()).collect();
            let db = Database::open_memory().unwrap();
            db.init_vector_table().unwrap();
            let ids: Vec<u64> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    db.insert_vector(v, &format!("s{i}"), 0, 0, "m")
                        .unwrap()
                        .unwrap() as u64
                })
                .collect();
            let dir = tempdir().unwrap();
            let path = dir.path().join("vectors.f32");
            sync_vector_file(&db, "m", D, &path).unwrap();
            let reader = VectorFileReader::open(&path, D);

            let cosine = |a: &[f32], b: &[f32]| {
                let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
                dot / (norm(a) * norm(b))
            };
            for _ in 0..10 {
                let query = random_vector();
                let mut ranked: Vec<(f32, u64)> = vectors
                    .iter()
                    .zip(&ids)
                    .map(|(v, &id)| (cosine(&query, v), id))
                    .collect();
                ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
                let found = reader.search(&query, 10).unwrap().hits;
                let expected: Vec<u64> = ranked[..10].iter().map(|&(_, id)| id).collect();
                let keys: Vec<u64> = found.iter().map(|&(key, _)| key).collect();
                assert_eq!(keys, expected, "nearest first");
                for (&(_, distance), &(similarity, _)) in found.iter().zip(&ranked) {
                    assert!((distance - (1.0 - similarity)).abs() < 1e-5);
                }
            }
        }

        #[test]
        fn sync_removes_temp_files_left_by_killed_runs() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("vectors.f32");
            let stale = dir.path().join("vectors.f32.4242.tmp");
            let fresh = dir.path().join("vectors.f32.4343.tmp");
            for tmp in [&stale, &fresh] {
                fs::write(tmp, b"partial").unwrap();
            }
            fs::File::options()
                .write(true)
                .open(&stale)
                .unwrap()
                .set_modified(SystemTime::now() - 2 * STALE_TEMP_AGE)
                .unwrap();

            let (db, _) = db_with(&[[1.0, 0.0, 0.0]]);
            sync_vector_file(&db, "m", DIM, &path).unwrap();

            assert!(!stale.exists(), "a temp file past the age limit is removed");
            assert!(
                fresh.exists(),
                "a temp file another run may be writing stays"
            );
        }
    }
}

#[cfg(not(target_os = "windows"))]
pub use inner::{sync_vector_file, vector_file_path, FileHits, VectorFileReader, VectorFileSync};
