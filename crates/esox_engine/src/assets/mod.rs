//! Asset management — handles, deduplication, and background loading.

pub(crate) mod loader;
pub(crate) mod registry;
pub(crate) mod watcher;

use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use esox_gfx::mesh3d::{MaterialHandle, MeshData, MeshHandle, Renderer3D, TextureHandle};
use esox_gfx::GpuContext;

use registry::AssetRegistry;

/// Lightweight handle to an asset. 4 bytes, Copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetHandle<T> {
    id: AssetId,
    _marker: PhantomData<T>,
}

/// Generational index: 24-bit slot + 8-bit generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetId(u32);

impl AssetId {
    fn new(index: u32, generation: u8) -> Self {
        Self((generation as u32) << 24 | (index & 0x00FF_FFFF))
    }

    pub fn index(self) -> u32 {
        self.0 & 0x00FF_FFFF
    }

    pub fn generation(self) -> u8 {
        (self.0 >> 24) as u8
    }
}

/// Marker types for AssetHandle.
pub struct MeshAsset;
pub struct TextureAsset;
pub struct MaterialAsset;

/// Manages asset loading, deduplication, and GPU handle resolution.
pub struct AssetManager {
    meshes: AssetRegistry<MeshHandle>,
    textures: AssetRegistry<TextureHandle>,
    materials: AssetRegistry<MaterialHandle>,
    path_map: HashMap<PathBuf, AssetId>,
    /// Reverse map from asset ID to a string name (path or user-provided name like `"@cube"`).
    reverse_map: HashMap<AssetId, String>,
    parse_tx: mpsc::Sender<loader::ParseResult>,
    parse_rx: mpsc::Receiver<loader::ParseResult>,
}

impl AssetManager {
    pub(crate) fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            meshes: AssetRegistry::new(),
            textures: AssetRegistry::new(),
            materials: AssetRegistry::new(),
            path_map: HashMap::new(),
            reverse_map: HashMap::new(),
            parse_tx: tx,
            parse_rx: rx,
        }
    }

    /// Register a pre-uploaded mesh handle (e.g., from procedural generation).
    pub fn register_mesh(&mut self, handle: MeshHandle) -> AssetHandle<MeshAsset> {
        let id = self.meshes.insert(handle);
        AssetHandle {
            id,
            _marker: PhantomData,
        }
    }

    /// Register a pre-uploaded texture handle.
    pub fn register_texture(&mut self, handle: TextureHandle) -> AssetHandle<TextureAsset> {
        let id = self.textures.insert(handle);
        AssetHandle {
            id,
            _marker: PhantomData,
        }
    }

    /// Register a pre-uploaded material handle.
    pub fn register_material(&mut self, handle: MaterialHandle) -> AssetHandle<MaterialAsset> {
        let id = self.materials.insert(handle);
        AssetHandle {
            id,
            _marker: PhantomData,
        }
    }

    /// Register a pre-uploaded mesh handle with a name (for serialization).
    pub fn register_mesh_named(
        &mut self,
        name: impl Into<String>,
        handle: MeshHandle,
    ) -> AssetHandle<MeshAsset> {
        let id = self.meshes.insert(handle);
        self.reverse_map.insert(id, name.into());
        AssetHandle {
            id,
            _marker: PhantomData,
        }
    }

    /// Register a pre-uploaded material handle with a name (for serialization).
    pub fn register_material_named(
        &mut self,
        name: impl Into<String>,
        handle: MaterialHandle,
    ) -> AssetHandle<MaterialAsset> {
        let id = self.materials.insert(handle);
        self.reverse_map.insert(id, name.into());
        AssetHandle {
            id,
            _marker: PhantomData,
        }
    }

    /// Look up the name/path for a mesh asset handle (for serialization).
    pub fn mesh_name(&self, handle: AssetHandle<MeshAsset>) -> Option<&str> {
        self.reverse_map.get(&handle.id).map(|s| s.as_str())
    }

    /// Look up the name/path for a material asset handle (for serialization).
    pub fn material_name(&self, handle: AssetHandle<MaterialAsset>) -> Option<&str> {
        self.reverse_map.get(&handle.id).map(|s| s.as_str())
    }

    /// Find a GPU mesh handle by name (for deserialization).
    pub fn find_mesh_by_name(&self, name: &str) -> Option<MeshHandle> {
        self.reverse_map
            .iter()
            .find(|(_, v)| v.as_str() == name)
            .and_then(|(id, _)| self.meshes.get(*id))
    }

    /// Find a GPU material handle by name (for deserialization).
    pub fn find_material_by_name(&self, name: &str) -> Option<MaterialHandle> {
        self.reverse_map
            .iter()
            .find(|(_, v)| v.as_str() == name)
            .and_then(|(id, _)| self.materials.get(*id))
    }

    /// Look up the name for a GPU mesh handle (searches all registered meshes).
    pub fn name_for_gpu_mesh(&self, gpu_handle: MeshHandle) -> Option<&str> {
        self.meshes.find_id_by_value(gpu_handle)
            .and_then(|id| self.reverse_map.get(&id))
            .map(|s| s.as_str())
    }

    /// Look up the name for a GPU material handle (searches all registered materials).
    pub fn name_for_gpu_material(&self, gpu_handle: MaterialHandle) -> Option<&str> {
        self.materials.find_id_by_value(gpu_handle)
            .and_then(|id| self.reverse_map.get(&id))
            .map(|s| s.as_str())
    }

    /// Load a mesh from MeshData synchronously (upload immediately).
    pub fn load_mesh_sync(
        &mut self,
        gpu: &GpuContext,
        renderer: &mut Renderer3D,
        data: &MeshData,
    ) -> AssetHandle<MeshAsset> {
        let gpu_handle = renderer.upload_mesh(gpu, data);
        self.register_mesh(gpu_handle)
    }

    /// Load a glTF scene asynchronously. Returns an AssetId for tracking.
    pub fn load_gltf_async(&mut self, path: impl AsRef<Path>) -> AssetId {
        let path = path.as_ref().to_owned();
        if let Some(&id) = self.path_map.get(&path) {
            return id;
        }
        let id = self.meshes.allocate(); // Placeholder slot.
        self.reverse_map.insert(id, path.to_string_lossy().into_owned());
        self.path_map.insert(path.clone(), id);
        loader::spawn_gltf_parse(self.parse_tx.clone(), id, path);
        id
    }

    /// Process completed background loads, uploading to GPU.
    pub fn process_uploads(&mut self, gpu: &GpuContext, renderer: &mut Renderer3D) {
        while let Ok(result) = self.parse_rx.try_recv() {
            match result {
                loader::ParseResult::GltfScene { id, scene } => {
                    let handles = renderer.upload_gltf_scene(gpu, scene);
                    // Fill the placeholder slot with the first mesh handle.
                    if let Some(&first) = handles.meshes.first() {
                        self.meshes.set(id, first);
                    }
                    // Register remaining mesh handles as additional slots.
                    for &mh in handles.meshes.iter().skip(1) {
                        self.meshes.insert(mh);
                    }
                    tracing::info!("uploaded glTF scene ({} meshes)", handles.meshes.len());
                }
                loader::ParseResult::Error { path, error, .. } => {
                    tracing::error!("asset load failed: {}: {error}", path.display());
                }
            }
        }
    }

    /// Resolve a mesh asset handle to a GPU handle.
    pub fn get_mesh(&self, handle: AssetHandle<MeshAsset>) -> Option<MeshHandle> {
        self.meshes.get(handle.id)
    }

    /// Resolve a texture asset handle to a GPU handle.
    pub fn get_texture(&self, handle: AssetHandle<TextureAsset>) -> Option<TextureHandle> {
        self.textures.get(handle.id)
    }

    /// Resolve a material asset handle to a GPU handle.
    pub fn get_material(&self, handle: AssetHandle<MaterialAsset>) -> Option<MaterialHandle> {
        self.materials.get(handle.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_id_roundtrip() {
        let id = AssetId::new(12345, 7);
        assert_eq!(id.index(), 12345);
        assert_eq!(id.generation(), 7);
    }

    #[test]
    fn asset_id_max_index() {
        let id = AssetId::new(0x00FF_FFFF, 255);
        assert_eq!(id.index(), 0x00FF_FFFF);
        assert_eq!(id.generation(), 255);
    }
}
