//! Hot-reload via inotify (behind `hot-reload` feature).

#[cfg(feature = "hot-reload")]
use std::path::PathBuf;

#[cfg(feature = "hot-reload")]
use std::sync::mpsc;

/// Watches asset files for changes and triggers re-parse.
#[cfg(feature = "hot-reload")]
pub(crate) struct AssetWatcher {
    _watcher: notify::RecommendedWatcher,
    rx: mpsc::Receiver<PathBuf>,
}

#[cfg(feature = "hot-reload")]
impl AssetWatcher {
    pub fn new() -> Option<Self> {
        use notify::Watcher;

        let (tx, rx) = mpsc::channel();
        let watcher = notify::recommended_watcher(move |res: Result<notify::Event, _>| {
            if let Ok(event) = res {
                if matches!(
                    event.kind,
                    notify::EventKind::Modify(_) | notify::EventKind::Create(_)
                ) {
                    for path in event.paths {
                        let _ = tx.send(path);
                    }
                }
            }
        })
        .ok()?;
        Some(Self {
            _watcher: watcher,
            rx,
        })
    }

    /// Drain changed paths since last poll.
    pub fn poll_changes(&self) -> Vec<PathBuf> {
        let mut changed = Vec::new();
        while let Ok(path) = self.rx.try_recv() {
            changed.push(path);
        }
        changed
    }
}
