//! Background asset loading — parse thread + result channel.

use std::path::PathBuf;
use std::sync::mpsc;

use crate::assets::AssetId;

/// Result from a background parse operation.
pub(crate) enum ParseResult {
    /// glTF scene loaded and ready for GPU upload.
    GltfScene {
        id: AssetId,
        scene: esox_gfx::mesh3d::GltfScene,
    },
    /// Parse failed.
    Error {
        #[allow(dead_code)]
        id: AssetId,
        path: PathBuf,
        error: String,
    },
}

/// Spawn a background thread to parse a glTF file.
pub(crate) fn spawn_gltf_parse(tx: mpsc::Sender<ParseResult>, id: AssetId, path: PathBuf) {
    std::thread::spawn(move || {
        let result = match esox_gfx::mesh3d::GltfScene::load(&path) {
            Ok(scene) => ParseResult::GltfScene { id, scene },
            Err(e) => ParseResult::Error {
                id,
                path,
                error: e.to_string(),
            },
        };
        let _ = tx.send(result);
    });
}
