use std::collections::HashMap;

use crate::error::Error;
use crate::primitive::UvRect;

/// Identifier for a texture atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AtlasId(pub u32);

/// Handle for a single allocation within an atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllocationId(pub u64);

/// A rectangular region within a texture atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtlasRegion {
    /// Which atlas this region belongs to.
    pub id: AtlasId,
    /// Array layer (for array textures).
    pub layer: u32,
    /// X offset in texels.
    pub x: u32,
    /// Y offset in texels.
    pub y: u32,
    /// Width in texels.
    pub w: u32,
    /// Height in texels.
    pub h: u32,
}

impl AtlasRegion {
    /// Convert this region to UV coordinates for texture sampling.
    pub fn to_uv_rect(&self, atlas_width: u32, atlas_height: u32) -> UvRect {
        let w = atlas_width as f32;
        let h = atlas_height as f32;
        UvRect {
            u0: self.x as f32 / w,
            v0: self.y as f32 / h,
            u1: (self.x + self.w) as f32 / w,
            v1: (self.y + self.h) as f32 / h,
        }
    }
}

/// Multi-layer atlas manager backed by a 2D array texture.
///
/// Manages a stack of [`ShelfAllocator`]s — one per array layer. When the
/// current layer is full, a new layer is added (up to the GPU limit) by
/// recreating the texture with one additional layer and copying existing data.
pub struct AtlasManager {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    layers: Vec<ShelfAllocator>,
    atlas_id: AtlasId,
    max_layers: u32,
    next_alloc_id: u64,
}

impl AtlasManager {
    /// Create a new atlas manager with a single initial layer.
    pub fn new(
        device: &wgpu::Device,
        atlas_id: AtlasId,
        width: u32,
        height: u32,
        label: &str,
    ) -> Self {
        let max_layers = device.limits().max_texture_array_layers;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let allocator = ShelfAllocator::new(atlas_id, width, height);
        Self {
            texture,
            view,
            width,
            height,
            layers: vec![allocator],
            atlas_id,
            max_layers,
            next_alloc_id: 0,
        }
    }

    /// Allocate a region, growing to a new layer if the current one is full.
    pub fn allocate(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        w: u32,
        h: u32,
    ) -> Result<(AllocationId, AtlasRegion), Error> {
        let layer_count = self.layers.len();
        // Try current (last) layer first.
        if let Some(allocator) = self.layers.last_mut() {
            match allocator.allocate(w, h) {
                Ok((_, mut region)) => {
                    region.layer = (layer_count - 1) as u32;
                    let alloc_id = AllocationId(self.next_alloc_id);
                    self.next_alloc_id = self.next_alloc_id.saturating_add(1);
                    return Ok((alloc_id, region));
                }
                Err(Error::AtlasFull) => {
                    // Fall through to grow.
                }
                Err(e) => return Err(e),
            }
        }

        // Grow to a new layer.
        self.grow(device, encoder)?;
        let new_layer = self.layers.len() - 1;
        let allocator = self
            .layers
            .last_mut()
            .expect("grow() just pushed a new layer");
        let (_, mut region) = allocator.allocate(w, h)?;
        region.layer = new_layer as u32;
        let alloc_id = AllocationId(self.next_alloc_id);
        self.next_alloc_id = self.next_alloc_id.saturating_add(1);
        Ok((alloc_id, region))
    }

    /// Add a new layer to the atlas, copying existing data to the new texture.
    fn grow(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), Error> {
        let new_layer_count = self.layers.len() as u32 + 1;
        if new_layer_count > self.max_layers {
            return Err(Error::AtlasFull);
        }

        let new_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atlas_array_texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: new_layer_count,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // Copy existing layers.
        for layer in 0..self.layers.len() as u32 {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &new_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        let new_view = new_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        self.texture = new_texture;
        self.view = new_view;
        self.layers
            .push(ShelfAllocator::new(self.atlas_id, self.width, self.height));

        tracing::info!(layers = new_layer_count, "atlas grew to new layer");
        Ok(())
    }

    /// Upload RGBA8 pixel data to a region (including layer).
    pub fn upload_region(&self, queue: &wgpu::Queue, region: &AtlasRegion, data: &[u8]) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: region.x,
                    y: region.y,
                    z: region.layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * region.w),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: region.w,
                height: region.h,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Get the texture view (D2Array dimension) for bind group binding.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Get the atlas dimensions `(width, height)`.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Number of layers currently allocated.
    pub fn layer_count(&self) -> u32 {
        self.layers.len() as u32
    }
}

/// A GPU texture used as a glyph atlas, supporting sub-region uploads.
pub struct AtlasTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl AtlasTexture {
    /// Create a new atlas texture with the given dimensions.
    ///
    /// The texture view is created with `D2Array` dimension so it is compatible
    /// with the bind group layout used by the instanced quad pipeline.
    pub fn new(device: &wgpu::Device, width: u32, height: u32, label: &str) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        Self {
            texture,
            view,
            width,
            height,
        }
    }

    /// Upload R8 pixel data to a sub-region of the atlas.
    pub fn upload_region(
        &self,
        queue: &wgpu::Queue,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        data: &[u8],
    ) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width), // R8 = 1 byte per pixel
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Get the texture view for bind group binding.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Get the atlas dimensions `(width, height)`.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// A strategy for allocating rectangular regions in a texture atlas.
pub trait AtlasAllocator {
    /// Allocate a region of the given size. Returns `Error::AtlasFull` if no space.
    fn allocate(&mut self, width: u32, height: u32) -> Result<(AllocationId, AtlasRegion), Error>;

    /// Free a previously allocated region.
    fn deallocate(&mut self, id: AllocationId);

    /// Clear all allocations.
    fn clear(&mut self);

    /// Fraction of the atlas that is currently allocated (0.0–1.0).
    fn utilization(&self) -> f32;

    /// Total atlas dimensions.
    fn size(&self) -> (u32, u32);

    /// Update the LRU timestamp for an allocation. No-op by default.
    fn touch(&mut self, _id: AllocationId) {}

    /// Advance the frame generation counter. No-op by default.
    fn advance_generation(&mut self) {}

    /// Evict the least-recently-used entries, freeing at least `min_area` texels
    /// or `min_count` entries (whichever comes first). Returns evicted IDs.
    fn evict_lru(&mut self, _min_area: u64, _min_count: usize) -> Vec<AllocationId> {
        Vec::new()
    }

    /// Ratio of freed-but-unusable space (fragmentation). 0.0 by default.
    fn fragmentation(&self) -> f32 {
        0.0
    }
}

/// Row (level) in the shelf allocator.
struct Shelf {
    y: u32,
    height: u32,
    cursor_x: u32,
}

/// A simple shelf-packing atlas allocator. Allocates rows of fixed height and
/// packs rectangles left-to-right within each row.
pub struct ShelfAllocator {
    atlas_id: AtlasId,
    width: u32,
    height: u32,
    shelves: Vec<Shelf>,
    next_y: u32,
    allocated_area: u64,
    next_alloc_id: u64,
}

impl ShelfAllocator {
    /// Create a new shelf allocator for an atlas of the given dimensions.
    pub fn new(atlas_id: AtlasId, width: u32, height: u32) -> Self {
        Self {
            atlas_id,
            width,
            height,
            shelves: Vec::new(),
            next_y: 0,
            allocated_area: 0,
            next_alloc_id: 0,
        }
    }
}

impl AtlasAllocator for ShelfAllocator {
    fn allocate(&mut self, w: u32, h: u32) -> Result<(AllocationId, AtlasRegion), Error> {
        // Allocate with 1px padding to prevent texture bleeding between glyphs.
        let padded_w = w + 1;
        let padded_h = h + 1;

        // Try to fit in an existing shelf.
        for shelf in &mut self.shelves {
            if padded_h <= shelf.height && shelf.cursor_x + padded_w <= self.width {
                let region = AtlasRegion {
                    id: self.atlas_id,
                    layer: 0,
                    x: shelf.cursor_x,
                    y: shelf.y,
                    w,
                    h,
                };
                shelf.cursor_x += padded_w;
                self.allocated_area += u64::from(w) * u64::from(h);
                let alloc_id = AllocationId(self.next_alloc_id);
                self.next_alloc_id = self.next_alloc_id.saturating_add(1);
                return Ok((alloc_id, region));
            }
        }

        // Open a new shelf.
        if self.next_y + padded_h > self.height {
            return Err(Error::AtlasFull);
        }
        let shelf_y = self.next_y;
        self.next_y += padded_h;
        self.shelves.push(Shelf {
            y: shelf_y,
            height: padded_h,
            cursor_x: padded_w,
        });
        let region = AtlasRegion {
            id: self.atlas_id,
            layer: 0,
            x: 0,
            y: shelf_y,
            w,
            h,
        };
        self.allocated_area += u64::from(w) * u64::from(h);
        let alloc_id = AllocationId(self.next_alloc_id);
        self.next_alloc_id = self.next_alloc_id.saturating_add(1);
        Ok((alloc_id, region))
    }

    fn deallocate(&mut self, _id: AllocationId) {
        // Shelf packing doesn't support individual deallocation; use `clear()`.
    }

    fn clear(&mut self) {
        self.shelves.clear();
        self.next_y = 0;
        self.allocated_area = 0;
    }

    fn utilization(&self) -> f32 {
        let total = u64::from(self.width) * u64::from(self.height);
        if total == 0 {
            return 0.0;
        }
        self.allocated_area as f32 / total as f32
    }

    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// An individual allocation tracked by the slab allocator.
struct SlabEntry {
    region: AtlasRegion,
    row_index: usize,
    last_used: u64,
}

/// A row in the slab allocator, similar to a shelf but with free-span tracking.
struct SlabRow {
    y: u32,
    height: u32,
    free_spans: Vec<(u32, u32)>,
    cursor_x: u32,
}

/// A shelf-style atlas allocator that tracks individual allocations for
/// deallocation and LRU eviction.
///
/// Uses the same row-based packing strategy as [`ShelfAllocator`] but maintains
/// per-entry metadata so that individual allocations can be freed and their space
/// reused via free-span tracking.
pub struct SlabAllocator {
    atlas_id: AtlasId,
    width: u32,
    height: u32,
    entries: HashMap<AllocationId, SlabEntry>,
    rows: Vec<SlabRow>,
    next_y: u32,
    next_alloc_id: u64,
    allocated_area: u64,
    freed_area: u64,
    generation: u64,
}

impl SlabAllocator {
    /// Create a new slab allocator for an atlas of the given dimensions.
    pub fn new(atlas_id: AtlasId, width: u32, height: u32) -> Self {
        Self {
            atlas_id,
            width,
            height,
            entries: HashMap::new(),
            rows: Vec::new(),
            next_y: 0,
            next_alloc_id: 0,
            allocated_area: 0,
            freed_area: 0,
            generation: 0,
        }
    }

    /// Try to allocate from free spans in existing rows.
    fn try_free_span(&mut self, padded_w: u32, padded_h: u32) -> Option<(usize, u32)> {
        for (row_idx, row) in self.rows.iter_mut().enumerate() {
            if padded_h > row.height {
                continue;
            }
            // Search free spans for a fit (first-fit).
            for i in 0..row.free_spans.len() {
                let (span_x, span_w) = row.free_spans[i];
                if padded_w <= span_w {
                    let alloc_x = span_x;
                    // Shrink or remove the span.
                    if padded_w == span_w {
                        row.free_spans.remove(i);
                    } else {
                        row.free_spans[i] = (span_x + padded_w, span_w - padded_w);
                    }
                    return Some((row_idx, alloc_x));
                }
            }
        }
        None
    }
}

impl AtlasAllocator for SlabAllocator {
    fn allocate(&mut self, w: u32, h: u32) -> Result<(AllocationId, AtlasRegion), Error> {
        let padded_w = w + 1;
        let padded_h = h + 1;

        // Try free spans first (reclaimed space from deallocation).
        if let Some((row_idx, alloc_x)) = self.try_free_span(padded_w, padded_h) {
            let row = &self.rows[row_idx];
            let region = AtlasRegion {
                id: self.atlas_id,
                layer: 0,
                x: alloc_x,
                y: row.y,
                w,
                h,
            };
            let alloc_id = AllocationId(self.next_alloc_id);
            self.next_alloc_id = self.next_alloc_id.saturating_add(1);
            let area = u64::from(w) * u64::from(h);
            self.allocated_area += area;
            self.freed_area = self.freed_area.saturating_sub(area);
            self.entries.insert(
                alloc_id,
                SlabEntry {
                    region,
                    row_index: row_idx,
                    last_used: self.generation,
                },
            );
            return Ok((alloc_id, region));
        }

        // Try cursor-advance in existing rows.
        for (row_idx, row) in self.rows.iter_mut().enumerate() {
            if padded_h <= row.height && row.cursor_x + padded_w <= self.width {
                let region = AtlasRegion {
                    id: self.atlas_id,
                    layer: 0,
                    x: row.cursor_x,
                    y: row.y,
                    w,
                    h,
                };
                row.cursor_x += padded_w;
                self.allocated_area += u64::from(w) * u64::from(h);
                let alloc_id = AllocationId(self.next_alloc_id);
                self.next_alloc_id = self.next_alloc_id.saturating_add(1);
                self.entries.insert(
                    alloc_id,
                    SlabEntry {
                        region,
                        row_index: row_idx,
                        last_used: self.generation,
                    },
                );
                return Ok((alloc_id, region));
            }
        }

        // Open a new row.
        if self.next_y + padded_h > self.height {
            return Err(Error::AtlasFull);
        }
        let row_y = self.next_y;
        self.next_y += padded_h;
        let row_idx = self.rows.len();
        self.rows.push(SlabRow {
            y: row_y,
            height: padded_h,
            cursor_x: padded_w,
            free_spans: Vec::new(),
        });
        let region = AtlasRegion {
            id: self.atlas_id,
            layer: 0,
            x: 0,
            y: row_y,
            w,
            h,
        };
        self.allocated_area += u64::from(w) * u64::from(h);
        let alloc_id = AllocationId(self.next_alloc_id);
        self.next_alloc_id = self.next_alloc_id.saturating_add(1);
        self.entries.insert(
            alloc_id,
            SlabEntry {
                region,
                row_index: row_idx,
                last_used: self.generation,
            },
        );
        Ok((alloc_id, region))
    }

    fn deallocate(&mut self, id: AllocationId) {
        if let Some(entry) = self.entries.remove(&id) {
            let area = u64::from(entry.region.w) * u64::from(entry.region.h);
            self.allocated_area = self.allocated_area.saturating_sub(area);
            self.freed_area += area;
            // Return padded span to the row's free list.
            let padded_w = entry.region.w + 1;
            if let Some(row) = self.rows.get_mut(entry.row_index) {
                row.free_spans.push((entry.region.x, padded_w));
            }
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.rows.clear();
        self.next_y = 0;
        self.allocated_area = 0;
        self.freed_area = 0;
    }

    fn utilization(&self) -> f32 {
        let total = u64::from(self.width) * u64::from(self.height);
        if total == 0 {
            return 0.0;
        }
        self.allocated_area as f32 / total as f32
    }

    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn touch(&mut self, id: AllocationId) {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.last_used = self.generation;
        }
    }

    fn advance_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    fn evict_lru(&mut self, min_area: u64, min_count: usize) -> Vec<AllocationId> {
        // Collect entries sorted by last_used (oldest first).
        let mut sorted: Vec<(AllocationId, u64, u64)> = self
            .entries
            .iter()
            .map(|(&id, e)| {
                (
                    id,
                    e.last_used,
                    u64::from(e.region.w) * u64::from(e.region.h),
                )
            })
            .collect();
        sorted.sort_by_key(|&(_, frame_gen, _)| frame_gen);

        let mut evicted = Vec::new();
        let mut freed = 0u64;
        for (id, _, area) in sorted {
            if freed >= min_area && evicted.len() >= min_count {
                break;
            }
            self.deallocate(id);
            freed += area;
            evicted.push(id);
        }
        evicted
    }

    fn fragmentation(&self) -> f32 {
        let total = self.allocated_area + self.freed_area;
        if total == 0 {
            return 0.0;
        }
        self.freed_area as f32 / total as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alloc(w: u32, h: u32) -> ShelfAllocator {
        ShelfAllocator::new(AtlasId(0), w, h)
    }

    #[test]
    fn basic_allocation() {
        let mut a = alloc(256, 256);
        let (id, region) = a.allocate(32, 32).unwrap();
        assert_eq!(id, AllocationId(0));
        assert_eq!(region.x, 0);
        assert_eq!(region.y, 0);
        assert_eq!(region.w, 32);
        assert_eq!(region.h, 32);
    }

    #[test]
    fn sequential_ids() {
        let mut a = alloc(256, 256);
        let (id1, _) = a.allocate(10, 10).unwrap();
        let (id2, _) = a.allocate(10, 10).unwrap();
        let (id3, _) = a.allocate(10, 10).unwrap();
        assert_eq!(id1, AllocationId(0));
        assert_eq!(id2, AllocationId(1));
        assert_eq!(id3, AllocationId(2));
    }

    #[test]
    fn packs_same_shelf() {
        let mut a = alloc(100, 100);
        let (_, r1) = a.allocate(30, 20).unwrap();
        let (_, r2) = a.allocate(30, 20).unwrap();
        // Same shelf → same y, adjacent x (with 1px padding gutter).
        assert_eq!(r1.y, r2.y);
        assert_eq!(r2.x, 31);
    }

    #[test]
    fn shorter_fits_existing_shelf() {
        let mut a = alloc(100, 100);
        // First alloc creates a shelf of height 21 (20 + 1px padding).
        a.allocate(50, 20).unwrap();
        // Second alloc is shorter (10+1=11 high) — fits in the same shelf (21 >= 11).
        let (_, r) = a.allocate(30, 10).unwrap();
        assert_eq!(r.y, 0);
        assert_eq!(r.x, 51);
    }

    #[test]
    fn taller_opens_new_shelf() {
        let mut a = alloc(100, 100);
        a.allocate(50, 10).unwrap();
        // This is taller (20+1=21) than the first shelf (10+1=11), so it opens a new one.
        let (_, r) = a.allocate(50, 20).unwrap();
        assert_eq!(r.y, 11);
        assert_eq!(r.x, 0);
    }

    #[test]
    fn atlas_full_vertically() {
        let mut a = alloc(100, 50);
        a.allocate(99, 28).unwrap();
        // Shelf used 29 (28+1), only 21 remain; requesting 21+1=22 should fail.
        assert!(a.allocate(10, 21).is_err());
    }

    #[test]
    fn atlas_full_horizontally_opens_new_shelf() {
        let mut a = alloc(100, 100);
        // Fill the first shelf (99+1=100 padded width fills 100-wide atlas).
        a.allocate(99, 10).unwrap();
        // No horizontal room left → opens a new shelf.
        let (_, r) = a.allocate(50, 10).unwrap();
        assert_eq!(r.y, 11);
    }

    #[test]
    fn utilization_tracking() {
        let mut a = alloc(100, 100);
        assert_eq!(a.utilization(), 0.0);
        a.allocate(50, 50).unwrap(); // 2500 out of 10000
        assert!((a.utilization() - 0.25).abs() < 1e-5);
    }

    #[test]
    fn utilization_zero_size_atlas() {
        let a = alloc(0, 0);
        assert_eq!(a.utilization(), 0.0);
    }

    #[test]
    fn clear_resets_everything() {
        let mut a = alloc(100, 100);
        a.allocate(40, 40).unwrap();
        a.allocate(40, 40).unwrap();
        a.clear();
        assert_eq!(a.utilization(), 0.0);
        // After clear, can allocate from the beginning again.
        let (_, r) = a.allocate(10, 10).unwrap();
        assert_eq!(r.x, 0);
        assert_eq!(r.y, 0);
    }

    #[test]
    fn size_returns_dimensions() {
        let a = alloc(512, 1024);
        assert_eq!(a.size(), (512, 1024));
    }

    #[test]
    fn exact_fit() {
        // With 1px padding, a 31×31 glyph needs 32×32 padded space in a 32×32 atlas.
        let mut a = alloc(32, 32);
        a.allocate(31, 31).unwrap();
        assert!((a.utilization() - (31.0 * 31.0) / (32.0 * 32.0)).abs() < 1e-5);
        // Now it's completely full (no room for even 1+1=2 padded).
        assert!(a.allocate(1, 1).is_err());
    }

    #[test]
    fn to_uv_rect_math() {
        let region = AtlasRegion {
            id: AtlasId(0),
            layer: 0,
            x: 64,
            y: 128,
            w: 32,
            h: 16,
        };
        let uv = region.to_uv_rect(256, 256);
        assert!((uv.u0 - 0.25).abs() < 1e-6);
        assert!((uv.v0 - 0.5).abs() < 1e-6);
        assert!((uv.u1 - 0.375).abs() < 1e-6);
        assert!((uv.v1 - 0.5625).abs() < 1e-6);
    }

    #[test]
    fn to_uv_rect_full_atlas() {
        let region = AtlasRegion {
            id: AtlasId(0),
            layer: 0,
            x: 0,
            y: 0,
            w: 512,
            h: 512,
        };
        let uv = region.to_uv_rect(512, 512);
        assert!((uv.u0).abs() < 1e-6);
        assert!((uv.v0).abs() < 1e-6);
        assert!((uv.u1 - 1.0).abs() < 1e-6);
        assert!((uv.v1 - 1.0).abs() < 1e-6);
    }

    // ── SlabAllocator tests ──

    fn slab(w: u32, h: u32) -> SlabAllocator {
        SlabAllocator::new(AtlasId(0), w, h)
    }

    #[test]
    fn slab_basic_allocation() {
        let mut a = slab(256, 256);
        let (id, region) = a.allocate(32, 32).unwrap();
        assert_eq!(id, AllocationId(0));
        assert_eq!(region.x, 0);
        assert_eq!(region.y, 0);
        assert_eq!(region.w, 32);
        assert_eq!(region.h, 32);
    }

    #[test]
    fn slab_deallocation_frees_space() {
        let mut a = slab(100, 100);
        let (id1, r1) = a.allocate(40, 40).unwrap();
        let util_before = a.utilization();
        a.deallocate(id1);
        assert!(a.utilization() < util_before);
        // Can allocate in freed space.
        let (_, r2) = a.allocate(30, 30).unwrap();
        // Should reuse the freed span (same row, same x).
        assert_eq!(r2.x, r1.x);
        assert_eq!(r2.y, r1.y);
    }

    #[test]
    fn slab_lru_eviction() {
        let mut a = slab(100, 100);
        // Allocate several entries.
        let (id1, _) = a.allocate(20, 20).unwrap();
        a.advance_generation();
        let (id2, _) = a.allocate(20, 20).unwrap();
        a.advance_generation();
        let (_id3, _) = a.allocate(20, 20).unwrap();

        // Touch id2 to make it recent.
        a.touch(id2);

        // Evict oldest (id1 should be evicted first).
        let evicted = a.evict_lru(0, 1);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0], id1);
    }

    #[test]
    fn slab_fragmentation_tracking() {
        let mut a = slab(100, 100);
        let (id1, _) = a.allocate(40, 40).unwrap();
        let (_, _) = a.allocate(40, 40).unwrap();
        assert!((a.fragmentation() - 0.0).abs() < 1e-5);
        a.deallocate(id1);
        assert!(a.fragmentation() > 0.0);
    }

    #[test]
    fn slab_clear_resets_everything() {
        let mut a = slab(100, 100);
        a.allocate(40, 40).unwrap();
        a.allocate(40, 40).unwrap();
        a.clear();
        assert_eq!(a.utilization(), 0.0);
        assert!((a.fragmentation() - 0.0).abs() < 1e-5);
        let (_, r) = a.allocate(10, 10).unwrap();
        assert_eq!(r.x, 0);
        assert_eq!(r.y, 0);
    }

    #[test]
    fn slab_atlas_full() {
        let mut a = slab(100, 50);
        a.allocate(99, 28).unwrap();
        assert!(a.allocate(10, 21).is_err());
    }
}
