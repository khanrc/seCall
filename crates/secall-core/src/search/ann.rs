//! HNSW ANN index over `turn_vectors`, one file per embedding model.
//!
//! Only `secall embed` writes the file: after its embedding pass, [`sync_ann`]
//! brings the index in line with the stored vectors and replaces the file by
//! rename. Search processes hold an [`AnnReader`], which memory-maps the file
//! and maps it again when the file is replaced, so they neither copy the index
//! into their heap nor keep serving the one they started with.

#[cfg(not(target_os = "windows"))]
mod inner {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::RwLock;
    use std::time::{Duration, SystemTime};

    use anyhow::{anyhow, bail, Result};
    use usearch::{new_index, Index, IndexOptions, MetricKind, ScalarKind};

    use crate::store::db::Database;

    /// Rows read per page while building, and vectors added per parallel round.
    const BUILD_PAGE: usize = 8192;
    /// Rebuild instead of patching in place once dead slots (keys of deleted
    /// vectors, plus slots already removed and never reclaimed) exceed this
    /// fraction of the stored vectors. usearch's `remove` leaves the slot in
    /// memory and in the file, so only a rebuild gives the space back.
    const REBUILD_DEAD_FRACTION: f64 = 0.25;
    /// A temporary index file this old belongs to a run that died while saving.
    const STALE_TEMP_AGE: Duration = Duration::from_secs(3600);

    fn options(dimensions: usize) -> IndexOptions {
        IndexOptions {
            dimensions,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: 0,
            expansion_add: 0,
            expansion_search: 0,
            multi: false,
        }
    }

    fn empty_index(dimensions: usize) -> Result<Index> {
        new_index(&options(dimensions)).map_err(|e| anyhow!("{e}"))
    }

    fn path_str(path: &Path) -> Result<&str> {
        path.to_str()
            .ok_or_else(|| anyhow!("non-UTF-8 ANN index path: {}", path.display()))
    }

    fn check_dimensions(index: &Index, dimensions: usize, path: &Path) -> Result<()> {
        if index.dimensions() != dimensions {
            bail!(
                "ANN index {} has {} dimensions, expected {dimensions}",
                path.display(),
                index.dimensions()
            );
        }
        Ok(())
    }

    /// Memory-map an index file read-only.
    fn view(path: &Path, dimensions: usize) -> Result<Index> {
        let index = empty_index(dimensions)?;
        index.view(path_str(path)?).map_err(|e| anyhow!("{e}"))?;
        check_dimensions(&index, dimensions, path)?;
        Ok(index)
    }

    /// Write next to `path`, then rename over it: a process that has the old
    /// file mapped keeps reading the old inode instead of a half-written one.
    /// The temporary name carries the pid because nothing stops two embed runs
    /// from overlapping, and a shared name would let one rename the other's
    /// half-written file into place.
    fn save_replacing(index: &Index, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension(format!("usearch.{}.tmp", std::process::id()));
        if let Err(e) = index.save(path_str(&tmp)?) {
            let _ = fs::remove_file(&tmp);
            bail!("{e}");
        }
        fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Remove temporary files that runs killed mid-save left next to `path`.
    /// Each carries its writer's pid, so no later save overwrites them.
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

    /// Where the index for an embedding model lives. Keyed by model and
    /// dimensions, so a model switch starts a fresh file.
    pub fn ann_path(model_name: &str, dimensions: usize) -> PathBuf {
        let model = model_name.replace(['/', ':'], "_");
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("secall")
            .join(format!("ann_{model}_{dimensions}.usearch"))
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
        index: Option<Index>,
    }

    /// Read-only, memory-mapped ANN index for search processes.
    pub struct AnnReader {
        path: PathBuf,
        dimensions: usize,
        mapped: RwLock<Mapped>,
    }

    impl AnnReader {
        /// Never fails: a missing or unreadable file only means no ANN until a
        /// usable file appears at `path`.
        pub fn open(path: &Path, dimensions: usize) -> Self {
            let reader = Self {
                path: path.to_path_buf(),
                dimensions,
                mapped: RwLock::new(Mapped {
                    stamp: None,
                    index: None,
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
            let index = current.and_then(|_| match view(&self.path, self.dimensions) {
                Ok(index) => {
                    tracing::info!(
                        path = %self.path.display(),
                        vectors = index.size(),
                        "ANN index mapped"
                    );
                    Some(index)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "ANN index unavailable, using BLOB scan");
                    None
                }
            });
            let mut mapped = self.mapped.write().unwrap_or_else(|e| e.into_inner());
            *mapped = Mapped {
                stamp: current,
                index,
            };
        }

        /// The `limit` nearest keys with their distances, or `None` when no
        /// index is mapped or it lacks `newest`, the highest stored vector id.
        ///
        /// Vector ids only grow, so an index holding the newest vector holds
        /// every vector that existed when it was synced. Keys of vectors deleted
        /// since then are still returned; skipping them is the caller's job.
        pub fn search(
            &self,
            query: &[f32],
            limit: usize,
            newest: u64,
        ) -> Result<Option<Vec<(u64, f32)>>> {
            self.refresh();
            let mapped = self.mapped.read().unwrap_or_else(|e| e.into_inner());
            let Some(index) = mapped.index.as_ref() else {
                return Ok(None);
            };
            if !index.contains(newest) {
                tracing::info!(
                    newest,
                    "ANN index predates the newest stored vector, using BLOB scan"
                );
                return Ok(None);
            }
            let found = index.search(query, limit).map_err(|e| anyhow!("{e}"))?;
            Ok(Some(found.keys.into_iter().zip(found.distances).collect()))
        }
    }

    /// What [`sync_ann`] did to the index file.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum AnnSync {
        /// The index already held exactly the stored vectors; the file is untouched.
        Unchanged { vectors: usize },
        /// Keys of deleted vectors removed and missing vectors added in place.
        Updated {
            vectors: usize,
            added: usize,
            removed: usize,
        },
        /// Built afresh from the stored vectors.
        Rebuilt { vectors: usize },
    }

    /// Bring the index at `path` in line with `model`'s rows in `turn_vectors`.
    ///
    /// Patches the index in place when it is close, and rebuilds it from the
    /// stored vectors when there is no usable file or too many dead slots.
    pub fn sync_ann(db: &Database, model: &str, dimensions: usize, path: &Path) -> Result<AnnSync> {
        remove_stale_temps(path);
        let ids = db.vector_ids(model)?;
        let existing = if path.exists() {
            match view(path, dimensions) {
                Ok(index) => Some(index),
                Err(e) => {
                    tracing::warn!(error = %e, "ANN index unreadable, rebuilding");
                    None
                }
            }
        } else {
            None
        };
        let Some(existing) = existing else {
            return rebuild(db, model, dimensions, path, ids.len());
        };

        let present = ids
            .iter()
            .filter(|&&id| existing.contains(id as u64))
            .count();
        let live = existing.size();
        if present == ids.len() && live == present {
            return Ok(AnnSync::Unchanged { vectors: live });
        }
        // Right after a view, capacity is the slot count, removed slots included.
        let dead = existing.capacity().saturating_sub(live) + live.saturating_sub(present);
        drop(existing);
        if dead as f64 > REBUILD_DEAD_FRACTION * ids.len() as f64 {
            return rebuild(db, model, dimensions, path, ids.len());
        }

        let index = empty_index(dimensions)?;
        index.load(path_str(path)?).map_err(|e| anyhow!("{e}"))?;
        check_dimensions(&index, dimensions, path)?;
        let mut removed = 0;
        for key in 1..=db.vector_id_ceiling()?.max(0) as u64 {
            if index.contains(key) && ids.binary_search(&(key as i64)).is_err() {
                index.remove(key).map_err(|e| anyhow!("{e}"))?;
                removed += 1;
            }
        }
        let missing: Vec<i64> = ids
            .iter()
            .copied()
            .filter(|&id| !index.contains(id as u64))
            .collect();
        index
            .reserve(index.capacity() + missing.len())
            .map_err(|e| anyhow!("{e}"))?;
        let mut added = 0;
        // Ids deleted since `vector_ids` read them, or vectors of another
        // dimension, which a rebuild would skip as well.
        let mut unindexable = 0;
        for id in missing {
            match db.vector_by_id(id)? {
                Some(vector) if vector.len() == dimensions => {
                    index.add(id as u64, &vector).map_err(|e| anyhow!("{e}"))?;
                    added += 1;
                }
                _ => unindexable += 1,
            }
        }
        // Keys above the id ceiling (an index from another database) cannot be
        // found by the scan above; a rebuild is the only way to drop them.
        if index.size() + unindexable != ids.len() {
            drop(index);
            return rebuild(db, model, dimensions, path, ids.len());
        }
        if added == 0 && removed == 0 {
            return Ok(AnnSync::Unchanged {
                vectors: index.size(),
            });
        }
        save_replacing(&index, path)?;
        Ok(AnnSync::Updated {
            vectors: index.size(),
            added,
            removed,
        })
    }

    fn rebuild(
        db: &Database,
        model: &str,
        dimensions: usize,
        path: &Path,
        expected: usize,
    ) -> Result<AnnSync> {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(8);
        let index = empty_index(dimensions)?;
        index
            .reserve_capacity_and_threads(expected.max(1), threads)
            .map_err(|e| anyhow!("{e}"))?;
        let mut after = 0;
        loop {
            let page = db.vectors_after(model, after, BUILD_PAGE)?;
            let Some(&(last, _)) = page.last() else {
                break;
            };
            after = last;
            let page: Vec<(i64, Vec<f32>)> = page
                .into_iter()
                .filter(|(_, v)| v.len() == dimensions)
                .collect();
            if index.size() + page.len() > index.capacity() {
                index
                    .reserve_capacity_and_threads(index.size() + page.len(), threads)
                    .map_err(|e| anyhow!("{e}"))?;
            }
            let index = &index;
            std::thread::scope(|scope| -> Result<()> {
                let workers: Vec<_> = page
                    .chunks(page.len().div_ceil(threads).max(1))
                    .map(|chunk| {
                        scope.spawn(move || -> Result<()> {
                            for (id, vector) in chunk {
                                index.add(*id as u64, vector).map_err(|e| anyhow!("{e}"))?;
                            }
                            Ok(())
                        })
                    })
                    .collect();
                for worker in workers {
                    worker
                        .join()
                        .map_err(|_| anyhow!("ANN build thread panicked"))??;
                }
                Ok(())
            })?;
        }
        save_replacing(&index, path)?;
        Ok(AnnSync::Rebuilt {
            vectors: index.size(),
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

        fn keys(path: &Path, ids: &[i64]) -> Vec<i64> {
            let index = view(path, DIM).unwrap();
            ids.iter()
                .copied()
                .filter(|&id| index.contains(id as u64))
                .collect()
        }

        #[test]
        fn sync_builds_then_leaves_a_matching_index_alone() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("ann.usearch");
            let (db, ids) = db_with(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);

            assert_eq!(
                sync_ann(&db, "m", DIM, &path).unwrap(),
                AnnSync::Rebuilt { vectors: 2 }
            );
            assert_eq!(keys(&path, &ids), ids);

            let before = stamp(&path);
            assert_eq!(
                sync_ann(&db, "m", DIM, &path).unwrap(),
                AnnSync::Unchanged { vectors: 2 }
            );
            assert_eq!(stamp(&path), before, "an unchanged index is not rewritten");
        }

        #[test]
        fn sync_patches_deleted_and_new_vectors_in_place() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("ann.usearch");
            let many: Vec<[f32; DIM]> = (0..20).map(|i| [1.0, i as f32, 0.0]).collect();
            let (db, ids) = db_with(&many);
            sync_ann(&db, "m", DIM, &path).unwrap();

            db.delete_session_vectors("s0").unwrap();
            let new_id = db
                .insert_vector(&[0.0, 0.0, 1.0], "s-new", 0, 0, "m")
                .unwrap()
                .unwrap();

            assert_eq!(
                sync_ann(&db, "m", DIM, &path).unwrap(),
                AnnSync::Updated {
                    vectors: 20,
                    added: 1,
                    removed: 1
                }
            );
            let mut expected: Vec<i64> = ids[1..].to_vec();
            expected.push(new_id);
            let mut all = ids.clone();
            all.push(new_id);
            assert_eq!(keys(&path, &all), expected);
        }

        #[test]
        fn sync_rebuilds_once_dead_slots_pass_the_threshold() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("ann.usearch");
            let many: Vec<[f32; DIM]> = (0..8).map(|i| [1.0, i as f32, 0.0]).collect();
            let (db, ids) = db_with(&many);
            sync_ann(&db, "m", DIM, &path).unwrap();

            for i in 0..4 {
                db.delete_session_vectors(&format!("s{i}")).unwrap();
            }

            assert_eq!(
                sync_ann(&db, "m", DIM, &path).unwrap(),
                AnnSync::Rebuilt { vectors: 4 }
            );
            let index = view(&path, DIM).unwrap();
            assert_eq!(
                index.capacity(),
                index.size(),
                "a rebuild leaves no dead slots"
            );
            assert_eq!(keys(&path, &ids), ids[4..].to_vec());
        }

        #[test]
        fn reader_maps_an_index_that_appears_and_follows_its_replacement() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("ann.usearch");
            let reader = AnnReader::open(&path, DIM);
            assert!(reader.search(&[1.0, 0.0, 0.0], 1, 1).unwrap().is_none());

            let (db, ids) = db_with(&[[1.0, 0.0, 0.0]]);
            let first_id = ids[0] as u64;
            sync_ann(&db, "m", DIM, &path).unwrap();
            let hits = reader
                .search(&[1.0, 0.0, 0.0], 1, first_id)
                .unwrap()
                .unwrap();
            assert_eq!(hits[0].0, first_id);

            let new_id = db
                .insert_vector(&[0.0, 1.0, 0.0], "s-new", 0, 0, "m")
                .unwrap()
                .unwrap() as u64;
            assert!(
                reader
                    .search(&[0.0, 1.0, 0.0], 1, new_id)
                    .unwrap()
                    .is_none(),
                "an index synced before the newest vector declines"
            );
            sync_ann(&db, "m", DIM, &path).unwrap();
            let hits = reader.search(&[0.0, 1.0, 0.0], 1, new_id).unwrap().unwrap();
            assert_eq!(hits[0].0, new_id);
        }

        #[test]
        fn sync_removes_temp_files_left_by_killed_runs() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("ann.usearch");
            let stale = dir.path().join("ann.usearch.4242.tmp");
            let fresh = dir.path().join("ann.usearch.4343.tmp");
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
            sync_ann(&db, "m", DIM, &path).unwrap();

            assert!(!stale.exists(), "a temp file past the age limit is removed");
            assert!(
                fresh.exists(),
                "a temp file another run may be writing stays"
            );
        }
    }
}

#[cfg(not(target_os = "windows"))]
pub use inner::{ann_path, sync_ann, AnnReader, AnnSync};
