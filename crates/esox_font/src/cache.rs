//! Glyph cache backed by an atlas allocator.

use std::collections::HashMap;

use esox_gfx::{AtlasAllocator, AtlasRegion};

use crate::face::FontFace;
use crate::rasterizer::GlyphRasterizer;
use crate::{Error, GlyphKey};

/// A cached glyph with its atlas region and bearing offsets.
#[derive(Debug, Clone, Copy)]
pub struct CachedGlyph {
    /// Region in the atlas texture (zero-size for glyphs like space).
    pub region: AtlasRegion,
    /// Horizontal bearing (offset from origin to left edge of glyph).
    pub bearing_x: f32,
    /// Vertical bearing (offset from baseline to top edge of glyph).
    pub bearing_y: f32,
    /// Atlas allocation ID for LRU tracking and eviction.
    pub alloc_id: esox_gfx::AllocationId,
    /// Whether this glyph was rasterized from a color source (COLR/CBDT/sbix).
    pub is_color: bool,
}

/// Cache of rasterized glyphs, backed by an atlas allocator.
pub struct GlyphCache {
    entries: HashMap<GlyphKey, CachedGlyph>,
    pending_uploads: Vec<(AtlasRegion, Vec<u8>)>,
    reverse_map: HashMap<esox_gfx::AllocationId, GlyphKey>,
}

impl GlyphCache {
    /// Create a new empty glyph cache.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            pending_uploads: Vec::new(),
            reverse_map: HashMap::new(),
        }
    }

    /// Look up a cached glyph by key.
    pub fn get(&self, key: &GlyphKey) -> Option<&CachedGlyph> {
        self.entries.get(key)
    }

    /// Look up or rasterize a glyph, returning its cached entry.
    ///
    /// On cache miss: rasterizes the glyph, allocates an atlas region,
    /// queues the pixel data for upload, and returns the cached glyph.
    /// Zero-size glyphs (space) skip atlas allocation. On cache hit,
    /// touches the allocation for LRU tracking.
    ///
    /// When `color` is true, attempts color rasterization (COLR/CBDT/sbix)
    /// first, falling back to monochrome. The color request is encoded in
    /// style bit 2 of the key to prevent cache collisions.
    pub fn get_or_insert(
        &mut self,
        key: GlyphKey,
        face: &FontFace,
        rasterizer: &mut GlyphRasterizer,
        allocator: &mut dyn AtlasAllocator,
        size_px: f32,
        color: bool,
    ) -> Result<CachedGlyph, Error> {
        if let Some(cached) = self.entries.get(&key) {
            allocator.touch(cached.alloc_id);
            return Ok(*cached);
        }

        let rasterized = if color {
            rasterizer.rasterize_color(face, key.glyph_id, size_px, key.style)?
        } else {
            rasterizer.rasterize(face, key.glyph_id, size_px, key.style)?
        };

        let (alloc_id, region) = if rasterized.width > 0 && rasterized.height > 0 {
            let (alloc_id, region) = allocator.allocate(rasterized.width, rasterized.height)?;
            self.pending_uploads.push((region, rasterized.data));
            (alloc_id, region)
        } else {
            // Zero-size glyph (space, etc.) — no atlas allocation needed.
            let region = AtlasRegion {
                id: esox_gfx::AtlasId(0),
                layer: 0,
                x: 0,
                y: 0,
                w: 0,
                h: 0,
            };
            (esox_gfx::AllocationId(u64::MAX), region)
        };

        let cached = CachedGlyph {
            region,
            bearing_x: rasterized.bearing_x,
            bearing_y: rasterized.bearing_y,
            alloc_id,
            is_color: rasterized.is_color,
        };

        self.entries.insert(key, cached);
        if alloc_id.0 != u64::MAX {
            self.reverse_map.insert(alloc_id, key);
        }
        Ok(cached)
    }

    /// Drain pending uploads. The caller should write these to `AtlasTexture`.
    pub fn drain_uploads(&mut self) -> Vec<(AtlasRegion, Vec<u8>)> {
        std::mem::take(&mut self.pending_uploads)
    }

    /// Clear all cached entries and pending uploads (for atlas-full recovery).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.pending_uploads.clear();
        self.reverse_map.clear();
    }

    /// Remove cache entries for evicted allocation IDs.
    ///
    /// Called after the allocator evicts LRU entries to keep the cache in sync.
    pub fn invalidate(&mut self, evicted: &[esox_gfx::AllocationId]) {
        for &id in evicted {
            if let Some(key) = self.reverse_map.remove(&id) {
                self.entries.remove(&key);
            }
        }
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for GlyphCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FontId;
    use esox_gfx::{AtlasId, ShelfAllocator};

    fn test_face() -> FontFace {
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test-data/JetBrainsMono-Regular.ttf"
        ))
        .expect("test font not found");
        FontFace::from_bytes(FontId(0), data).unwrap()
    }

    fn glyph_key(face: &FontFace, c: char) -> GlyphKey {
        let font_ref = face.as_swash_ref();
        let glyph_id = u32::from(font_ref.charmap().map(c));
        GlyphKey {
            font_id: face.id(),
            glyph_id,
            size_tenths: 160,
            style: 0,
        }
    }

    #[test]
    fn get_miss_returns_none() {
        let cache = GlyphCache::new();
        let key = GlyphKey {
            font_id: FontId(0),
            glyph_id: 1,
            size_tenths: 160,
            style: 0,
        };
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn get_or_insert_then_get() {
        let face = test_face();
        let mut cache = GlyphCache::new();
        let mut rasterizer = GlyphRasterizer::new();
        let mut alloc = ShelfAllocator::new(AtlasId(0), 512, 512);
        let key = glyph_key(&face, 'A');

        let cached = cache
            .get_or_insert(key, &face, &mut rasterizer, &mut alloc, 16.0, false)
            .unwrap();
        assert!(cached.region.w > 0);

        // Now get should return the same result.
        let got = cache.get(&key).unwrap();
        assert_eq!(got.region.w, cached.region.w);
        assert_eq!(got.region.h, cached.region.h);
    }

    #[test]
    fn duplicate_insert_uses_cache() {
        let face = test_face();
        let mut cache = GlyphCache::new();
        let mut rasterizer = GlyphRasterizer::new();
        let mut alloc = ShelfAllocator::new(AtlasId(0), 512, 512);
        let key = glyph_key(&face, 'B');

        cache
            .get_or_insert(key, &face, &mut rasterizer, &mut alloc, 16.0, false)
            .unwrap();
        let before_len = cache.len();

        cache
            .get_or_insert(key, &face, &mut rasterizer, &mut alloc, 16.0, false)
            .unwrap();
        assert_eq!(cache.len(), before_len);
    }

    #[test]
    fn drain_uploads_returns_data_then_empty() {
        let face = test_face();
        let mut cache = GlyphCache::new();
        let mut rasterizer = GlyphRasterizer::new();
        let mut alloc = ShelfAllocator::new(AtlasId(0), 512, 512);
        let key = glyph_key(&face, 'C');

        cache
            .get_or_insert(key, &face, &mut rasterizer, &mut alloc, 16.0, false)
            .unwrap();

        let uploads = cache.drain_uploads();
        assert!(!uploads.is_empty());
        assert!(!uploads[0].1.is_empty());

        // Second drain should be empty.
        let uploads2 = cache.drain_uploads();
        assert!(uploads2.is_empty());
    }

    #[test]
    fn clear_empties_all() {
        let face = test_face();
        let mut cache = GlyphCache::new();
        let mut rasterizer = GlyphRasterizer::new();
        let mut alloc = ShelfAllocator::new(AtlasId(0), 512, 512);
        let key = glyph_key(&face, 'D');

        cache
            .get_or_insert(key, &face, &mut rasterizer, &mut alloc, 16.0, false)
            .unwrap();
        assert!(!cache.is_empty());

        cache.clear();
        assert!(cache.is_empty());
        assert!(cache.get(&key).is_none());
        assert!(cache.drain_uploads().is_empty());
    }

    #[test]
    fn color_true_on_monochrome_font_falls_back() {
        let face = test_face();
        let mut cache = GlyphCache::new();
        let mut rasterizer = GlyphRasterizer::new();
        let mut alloc = ShelfAllocator::new(AtlasId(0), 512, 512);
        let key = glyph_key(&face, 'A');

        let cached = cache
            .get_or_insert(key, &face, &mut rasterizer, &mut alloc, 16.0, true)
            .unwrap();
        // JetBrains Mono has no color tables — is_color should be false.
        assert!(!cached.is_color);
        assert!(cached.region.w > 0);
    }

    #[test]
    fn color_and_mono_keys_dont_collide() {
        let face = test_face();
        let mut cache = GlyphCache::new();
        let mut rasterizer = GlyphRasterizer::new();
        let mut alloc = ShelfAllocator::new(AtlasId(0), 512, 512);

        let mono_key = glyph_key(&face, 'A');
        let color_key = GlyphKey {
            style: mono_key.style | 4,
            ..mono_key
        };

        cache
            .get_or_insert(mono_key, &face, &mut rasterizer, &mut alloc, 16.0, false)
            .unwrap();
        cache
            .get_or_insert(color_key, &face, &mut rasterizer, &mut alloc, 16.0, true)
            .unwrap();

        // Both should be cached as distinct entries.
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&mono_key).is_some());
        assert!(cache.get(&color_key).is_some());
    }
}
