//! Shader hot-reload library for 3D renderer shaders.
//!
//! With the `hot-reload` feature enabled, shaders are loaded from `.wgsl` files
//! on disk and watched for changes via `notify`. Without the feature, this is a
//! zero-cost wrapper that returns the embedded `const &str` shader sources.

use std::path::PathBuf;

use super::material::MaterialType;
use super::shaders_embedded::{SHADER_PREAMBLE, FS_UNLIT, FS_LIT, FS_PBR, FS_TOON, COMPOSITE_SHADER_3D};
use super::shadow::SHADOW_VERTEX_SHADER;
use super::ssao::{SSAO_SHADER, SSAO_BLUR_SHADER};
use super::depth_resolve::DEPTH_RESOLVE_SHADER;
use super::skinning::SKINNING_SHADER;
use crate::bloom;

/// Identifies a shader source slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderSlot {
    Preamble,
    FsUnlit,
    FsLit,
    FsPbr,
    FsToon,
    Composite,
    ShadowVertex,
    Ssao,
    SsaoBlur,
    Skinning,
    DepthResolve,
    BloomDownsample,
    BloomUpsample,
}

impl ShaderSlot {
    /// All shader slots.
    #[cfg_attr(not(feature = "hot-reload"), allow(dead_code))]
    pub const ALL: &[ShaderSlot] = &[
        ShaderSlot::Preamble,
        ShaderSlot::FsUnlit,
        ShaderSlot::FsLit,
        ShaderSlot::FsPbr,
        ShaderSlot::FsToon,
        ShaderSlot::Composite,
        ShaderSlot::ShadowVertex,
        ShaderSlot::Ssao,
        ShaderSlot::SsaoBlur,
        ShaderSlot::Skinning,
        ShaderSlot::DepthResolve,
        ShaderSlot::BloomDownsample,
        ShaderSlot::BloomUpsample,
    ];

    /// The file stem (without `.wgsl` extension) for this slot.
    #[cfg_attr(not(feature = "hot-reload"), allow(dead_code))]
    fn file_stem(&self) -> &'static str {
        match self {
            ShaderSlot::Preamble => "preamble",
            ShaderSlot::FsUnlit => "fs_unlit",
            ShaderSlot::FsLit => "fs_lit",
            ShaderSlot::FsPbr => "fs_pbr",
            ShaderSlot::FsToon => "fs_toon",
            ShaderSlot::Composite => "composite",
            ShaderSlot::ShadowVertex => "shadow_vertex",
            ShaderSlot::Ssao => "ssao",
            ShaderSlot::SsaoBlur => "ssao_blur",
            ShaderSlot::Skinning => "skinning",
            ShaderSlot::DepthResolve => "depth_resolve",
            ShaderSlot::BloomDownsample => "bloom_downsample",
            ShaderSlot::BloomUpsample => "bloom_upsample",
        }
    }

    /// The embedded const fallback source for this slot.
    fn fallback_source(&self) -> &'static str {
        match self {
            ShaderSlot::Preamble => SHADER_PREAMBLE,
            ShaderSlot::FsUnlit => FS_UNLIT,
            ShaderSlot::FsLit => FS_LIT,
            ShaderSlot::FsPbr => FS_PBR,
            ShaderSlot::FsToon => FS_TOON,
            ShaderSlot::Composite => COMPOSITE_SHADER_3D,
            ShaderSlot::ShadowVertex => SHADOW_VERTEX_SHADER,
            ShaderSlot::Ssao => SSAO_SHADER,
            ShaderSlot::SsaoBlur => SSAO_BLUR_SHADER,
            ShaderSlot::Skinning => SKINNING_SHADER,
            ShaderSlot::DepthResolve => DEPTH_RESOLVE_SHADER,
            // Bloom slots: compose from parts as fallback.
            ShaderSlot::BloomDownsample | ShaderSlot::BloomUpsample => {
                // Can't return a dynamic string from &'static str, handled
                // separately in get() for the non-hot-reload path.
                ""
            }
        }
    }

    /// Map a file stem to a slot, if known.
    #[cfg_attr(not(feature = "hot-reload"), allow(dead_code))]
    fn from_file_stem(stem: &str) -> Option<ShaderSlot> {
        ShaderSlot::ALL.iter().find(|s| s.file_stem() == stem).copied()
    }
}

// ── Non-hot-reload implementation (release builds) ──

#[cfg(not(feature = "hot-reload"))]
pub struct ShaderLibrary {
    bloom_downsample_cache: String,
    bloom_upsample_cache: String,
}

#[cfg(not(feature = "hot-reload"))]
impl ShaderLibrary {
    pub fn new(_shader_dir: Option<PathBuf>) -> Self {
        Self {
            bloom_downsample_cache: bloom::downsample_shader_source(),
            bloom_upsample_cache: bloom::upsample_shader_source(),
        }
    }

    pub fn get(&self, slot: ShaderSlot) -> &str {
        match slot {
            ShaderSlot::BloomDownsample => &self.bloom_downsample_cache,
            ShaderSlot::BloomUpsample => &self.bloom_upsample_cache,
            other => other.fallback_source(),
        }
    }

    pub fn compose_material_shader(&self, material_type: MaterialType) -> String {
        let preamble = self.get(ShaderSlot::Preamble);
        let fragment = match material_type {
            MaterialType::Unlit => self.get(ShaderSlot::FsUnlit),
            MaterialType::Lit => self.get(ShaderSlot::FsLit),
            MaterialType::PBR => self.get(ShaderSlot::FsPbr),
            MaterialType::Toon => self.get(ShaderSlot::FsToon),
        };
        format!("{preamble}\n{fragment}")
    }

    pub fn poll_changes(&mut self) -> Vec<ShaderSlot> {
        Vec::new()
    }
}

// ── Hot-reload implementation (dev builds) ──

#[cfg(feature = "hot-reload")]
use std::collections::HashMap;
#[cfg(feature = "hot-reload")]
use std::sync::mpsc;

#[cfg(feature = "hot-reload")]
pub struct ShaderLibrary {
    sources: HashMap<ShaderSlot, String>,
    #[allow(dead_code)]
    watcher: Option<notify::RecommendedWatcher>,
    rx: Option<mpsc::Receiver<PathBuf>>,
    _shader_dir: Option<PathBuf>,
}

#[cfg(feature = "hot-reload")]
impl ShaderLibrary {
    pub fn new(shader_dir: Option<PathBuf>) -> Self {
        let mut sources = HashMap::new();

        // Pre-populate bloom caches from composed sources.
        sources.insert(ShaderSlot::BloomDownsample, bloom::downsample_shader_source());
        sources.insert(ShaderSlot::BloomUpsample, bloom::upsample_shader_source());

        let (watcher, rx) = if let Some(ref dir) = shader_dir {
            // Load shader files from disk, falling back to embedded sources.
            for &slot in ShaderSlot::ALL {
                let path = dir.join(format!("{}.wgsl", slot.file_stem()));
                if let Ok(content) = std::fs::read_to_string(&path) {
                    tracing::info!("Loaded shader from disk: {}", path.display());
                    sources.insert(slot, content);
                }
                // If file missing, fallback will be used via get().
            }

            // Start watching the shader directory.
            match Self::start_watcher(dir) {
                Ok((w, r)) => (Some(w), Some(r)),
                Err(e) => {
                    tracing::warn!("Failed to start shader watcher: {e}");
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        Self {
            sources,
            watcher,
            rx,
            _shader_dir: shader_dir,
        }
    }

    fn start_watcher(
        dir: &PathBuf,
    ) -> Result<(notify::RecommendedWatcher, mpsc::Receiver<PathBuf>), notify::Error> {
        use notify::{Watcher, RecursiveMode, Event, EventKind};

        let (tx, rx) = mpsc::channel();
        let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            if let Ok(event) = res {
                match event.kind {
                    EventKind::Modify(_) | EventKind::Create(_) => {
                        for path in event.paths {
                            let _ = tx.send(path);
                        }
                    }
                    _ => {}
                }
            }
        })?;
        watcher.watch(dir.as_ref(), RecursiveMode::NonRecursive)?;
        tracing::info!("Shader hot-reload watcher started on {}", dir.display());

        Ok((watcher, rx))
    }

    pub fn get(&self, slot: ShaderSlot) -> &str {
        if let Some(src) = self.sources.get(&slot) {
            return src;
        }
        // Fallback to embedded const.
        match slot {
            ShaderSlot::BloomDownsample | ShaderSlot::BloomUpsample => {
                // Should be in sources from new(), but just in case:
                ""
            }
            other => other.fallback_source(),
        }
    }

    pub fn compose_material_shader(&self, material_type: MaterialType) -> String {
        let preamble = self.get(ShaderSlot::Preamble);
        let fragment = match material_type {
            MaterialType::Unlit => self.get(ShaderSlot::FsUnlit),
            MaterialType::Lit => self.get(ShaderSlot::FsLit),
            MaterialType::PBR => self.get(ShaderSlot::FsPbr),
            MaterialType::Toon => self.get(ShaderSlot::FsToon),
        };
        format!("{preamble}\n{fragment}")
    }

    pub fn poll_changes(&mut self) -> Vec<ShaderSlot> {
        let rx = match &self.rx {
            Some(rx) => rx,
            None => return Vec::new(),
        };

        // Drain all pending file-change notifications.
        let mut changed_paths: Vec<PathBuf> = Vec::new();
        while let Ok(path) = rx.try_recv() {
            if !changed_paths.contains(&path) {
                changed_paths.push(path);
            }
        }

        let mut updated_slots = Vec::new();

        for path in changed_paths {
            let stem = match path.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s,
                None => continue,
            };

            // Only process .wgsl files.
            match path.extension().and_then(|e| e.to_str()) {
                Some("wgsl") => {}
                _ => continue,
            }

            let slot = match ShaderSlot::from_file_stem(stem) {
                Some(s) => s,
                None => continue,
            };

            // Re-read the file.
            let content = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("Failed to read shader file {}: {e}", path.display());
                    continue;
                }
            };

            // Validate with naga before accepting.
            let validate_src = match slot {
                // Material shaders need preamble prepended for validation.
                ShaderSlot::FsUnlit | ShaderSlot::FsLit | ShaderSlot::FsPbr | ShaderSlot::FsToon => {
                    let preamble = self.get(ShaderSlot::Preamble);
                    format!("{preamble}\n{content}")
                }
                ShaderSlot::Preamble => {
                    // Validate preamble + existing FsUnlit as a representative combo.
                    let fragment = self.get(ShaderSlot::FsUnlit);
                    format!("{content}\n{fragment}")
                }
                _ => content.clone(),
            };

            if let Err(e) = validate_wgsl(&validate_src) {
                tracing::warn!(
                    "Shader validation failed for {}.wgsl, keeping old version: {e}",
                    stem
                );
                continue;
            }

            tracing::info!("Hot-reloaded shader: {}.wgsl", stem);
            self.sources.insert(slot, content);
            updated_slots.push(slot);
        }

        updated_slots
    }
}

/// Validate WGSL source using naga.
#[cfg_attr(not(feature = "hot-reload"), allow(dead_code))]
fn validate_wgsl(source: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|e| format!("parse error: {e}"))?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .map_err(|e| format!("validation error: {e}"))?;
    Ok(())
}
