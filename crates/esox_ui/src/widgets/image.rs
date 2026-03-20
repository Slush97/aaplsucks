//! Image widget — display decoded PNG/JPEG images in the UI.

use std::collections::HashMap;

use esox_gfx::{AtlasManager, AtlasId, GpuContext};

use crate::response::Response;
use crate::state::WidgetKind;
use crate::Ui;

/// Handle to a loaded image in the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageHandle(pub u64);

/// Cached image entry.
struct CachedImage {
    /// UV rect in the atlas.
    uv: esox_gfx::UvRect,
    /// Layer in the atlas.
    layer: u32,
    /// Original pixel dimensions.
    #[allow(dead_code)]
    width: u32,
    #[allow(dead_code)]
    height: u32,
}

/// Image cache backed by a GPU atlas. Decodes and uploads RGBA8 images.
pub struct ImageCache {
    atlas: AtlasManager,
    images: HashMap<u64, CachedImage>,
    dirty: bool,
}

impl ImageCache {
    /// Create a new image cache with a 512x512 initial atlas.
    pub fn new(gpu: &GpuContext) -> Self {
        let atlas = AtlasManager::new(
            &gpu.device,
            AtlasId(99), // distinct from glyph atlas
            512,
            512,
            "image_atlas",
        );
        Self {
            atlas,
            images: HashMap::new(),
            dirty: true,
        }
    }

    /// Load an image from raw bytes (PNG or JPEG). Returns a handle for rendering.
    /// Returns `None` if decoding fails.
    pub fn load_from_bytes(&mut self, data: &[u8], gpu: &GpuContext) -> Option<ImageHandle> {
        // Hash the data for dedup.
        let hash = fnv1a_hash(data);
        if self.images.contains_key(&hash) {
            return Some(ImageHandle(hash));
        }

        let img = image::load_from_memory(data).ok()?.into_rgba8();
        let (w, h) = img.dimensions();
        self.upload(hash, w, h, &img, gpu)
    }

    /// Load an image from a file path. Returns `None` if loading or decoding fails.
    pub fn load_from_path(
        &mut self,
        path: &std::path::Path,
        gpu: &GpuContext,
    ) -> Option<ImageHandle> {
        let data = std::fs::read(path).ok()?;
        self.load_from_bytes(&data, gpu)
    }

    fn upload(
        &mut self,
        key: u64,
        w: u32,
        h: u32,
        rgba: &image::RgbaImage,
        gpu: &GpuContext,
    ) -> Option<ImageHandle> {
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("image_atlas_upload") },
        );
        let (_alloc_id, region) = self
            .atlas
            .allocate(&gpu.device, &mut encoder, w, h)
            .ok()?;
        gpu.queue.submit(Some(encoder.finish()));

        self.atlas.upload_region(&gpu.queue, &region, rgba.as_raw());

        let atlas_w = self.atlas.size().0;
        let atlas_h = self.atlas.size().1;
        let uv = region.to_uv_rect(atlas_w, atlas_h);

        self.images.insert(key, CachedImage {
            uv,
            layer: region.layer,
            width: w,
            height: h,
        });
        self.dirty = true;
        Some(ImageHandle(key))
    }

    /// Get the atlas texture view (for binding).
    pub fn view(&self) -> &wgpu::TextureView {
        self.atlas.view()
    }

    /// Whether the atlas has been modified since the last bind.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark the atlas as bound (resets dirty flag).
    pub fn mark_bound(&mut self) {
        self.dirty = false;
    }

    fn get(&self, handle: ImageHandle) -> Option<&CachedImage> {
        self.images.get(&handle.0)
    }
}

/// Simple FNV-1a hash for content dedup.
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

impl<'f> Ui<'f> {
    /// Draw an image widget at the specified size. Returns a Response.
    ///
    /// The image is sampled from the image atlas using `flags.y = 2.0`.
    pub fn image(
        &mut self,
        id: u64,
        cache: &ImageCache,
        handle: ImageHandle,
        width: f32,
        height: f32,
    ) -> Response {
        let rect = self.allocate_rect_keyed(id, width, height);
        self.register_widget(id, rect, WidgetKind::Button);
        let response = self.widget_response(id, rect);

        if let Some(img) = cache.get(handle) {
            self.frame.push(esox_gfx::QuadInstance {
                rect: [rect.x, rect.y, rect.w, rect.h],
                uv: [img.uv.u0, img.uv.v0, img.uv.u1, img.uv.v1],
                color: [1.0, 1.0, 1.0, 1.0],
                border_radius: [0.0; 4],
                sdf_params: [0.0; 4],
                flags: [0.0, 2.0, img.layer as f32, 0.0], // flags.y=2.0 triggers image sampling
                clip_rect: [0.0; 4],
                color2: [0.0; 4],
                extra: [0.0; 4],
            });
        }

        response
    }
}
