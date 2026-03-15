/// A rectangular damage region in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DamageRect {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width.
    pub width: f32,
    /// Height.
    pub height: f32,
}

impl DamageRect {
    /// Create a new damage rectangle.
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Compute the union of two damage rectangles (bounding box).
    pub fn union(self, other: Self) -> Self {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let right = (self.x + self.width).max(other.x + other.width);
        let bottom = (self.y + self.height).max(other.y + other.height);
        Self {
            x,
            y,
            width: right - x,
            height: bottom - y,
        }
    }

    /// Compute the intersection of two damage rectangles, if any.
    pub fn intersect(self, other: Self) -> Option<Self> {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let right = (self.x + self.width).min(other.x + other.width);
        let bottom = (self.y + self.height).min(other.y + other.height);
        if right > x && bottom > y {
            Some(Self {
                x,
                y,
                width: right - x,
                height: bottom - y,
            })
        } else {
            None
        }
    }

    /// Check whether a point lies inside this rectangle.
    pub fn contains_point(self, px: f32, py: f32) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }
}

/// Tracks dirty screen regions across frames.
pub struct DamageTracker {
    regions: Vec<DamageRect>,
    full_invalidation: bool,
}

impl DamageTracker {
    /// Create a new damage tracker.
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
            // Start with full invalidation so the first frame always renders.
            full_invalidation: true,
        }
    }

    /// Mark the entire surface as needing redraw.
    pub fn invalidate_all(&mut self) {
        self.full_invalidation = true;
        self.regions.clear();
    }

    /// Add a damaged region.
    pub fn add(&mut self, rect: DamageRect) {
        if !self.full_invalidation {
            self.regions.push(rect);
        }
    }

    /// Get the current damage regions. Returns `None` if the full surface is invalidated.
    pub fn regions(&self) -> Option<&[DamageRect]> {
        if self.full_invalidation {
            None
        } else {
            Some(&self.regions)
        }
    }

    /// Whether the full surface needs redraw.
    pub fn is_full_invalidation(&self) -> bool {
        self.full_invalidation
    }

    /// Reset damage state for the next frame.
    pub fn reset(&mut self) {
        self.regions.clear();
        self.full_invalidation = false;
    }
}

impl Default for DamageTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tile-based instance caching ──

/// Tile size in pixels for partial redraw caching.
pub const TILE_SIZE: u32 = 128;

/// Index of a tile in the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileIndex(pub u16);

/// Cached instances for a single tile.
struct TileCache {
    instances: Vec<crate::primitive::QuadInstance>,
    generation: u64,
}

/// A grid of tiles covering the viewport. Each tile caches its quad instances
/// from the previous frame. Dirty tiles get fresh instances; clean tiles replay
/// their cache.
pub struct TileGrid {
    cols: u16,
    rows: u16,
    dirty: Vec<bool>,
    cache: Vec<TileCache>,
    generation: u64,
    viewport_w: u32,
    viewport_h: u32,
}

impl TileGrid {
    /// Create a new tile grid for the given viewport dimensions.
    pub fn new(viewport_w: u32, viewport_h: u32) -> Self {
        let cols = ((viewport_w + TILE_SIZE - 1) / TILE_SIZE) as u16;
        let rows = ((viewport_h + TILE_SIZE - 1) / TILE_SIZE) as u16;
        let count = cols as usize * rows as usize;
        Self {
            cols,
            rows,
            dirty: vec![true; count],
            cache: (0..count)
                .map(|_| TileCache {
                    instances: Vec::new(),
                    generation: 0,
                })
                .collect(),
            generation: 1,
            viewport_w,
            viewport_h,
        }
    }

    /// Number of columns in the grid.
    pub fn cols(&self) -> u16 {
        self.cols
    }

    /// Number of rows in the grid.
    pub fn rows(&self) -> u16 {
        self.rows
    }

    /// Current generation counter.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Whether the given tile is dirty this frame.
    pub fn is_dirty(&self, idx: TileIndex) -> bool {
        self.dirty.get(idx.0 as usize).copied().unwrap_or(true)
    }

    /// Total number of tiles.
    pub fn tile_count(&self) -> usize {
        self.cols as usize * self.rows as usize
    }

    /// Resize the grid if the viewport changed. Marks all tiles dirty.
    pub fn resize(&mut self, viewport_w: u32, viewport_h: u32) {
        if viewport_w == self.viewport_w && viewport_h == self.viewport_h {
            return;
        }
        *self = Self::new(viewport_w, viewport_h);
    }

    /// Mark all tiles as dirty (full redraw).
    pub fn invalidate_all(&mut self) {
        self.dirty.fill(true);
    }

    /// Mark tiles overlapping a damage rect as dirty.
    pub fn mark_damage(&mut self, rect: &DamageRect) {
        let col_start = (rect.x.max(0.0) as u32 / TILE_SIZE) as u16;
        let col_end = (((rect.x + rect.width).ceil() as u32 + TILE_SIZE - 1) / TILE_SIZE).min(self.cols as u32) as u16;
        let row_start = (rect.y.max(0.0) as u32 / TILE_SIZE) as u16;
        let row_end = (((rect.y + rect.height).ceil() as u32 + TILE_SIZE - 1) / TILE_SIZE).min(self.rows as u32) as u16;

        for row in row_start..row_end {
            for col in col_start..col_end {
                self.dirty[row as usize * self.cols as usize + col as usize] = true;
            }
        }
    }

    /// Mark tiles dirty from a `DamageTracker`. Full invalidation marks all tiles dirty.
    pub fn apply_damage(&mut self, damage: &DamageTracker) {
        if damage.is_full_invalidation() {
            self.invalidate_all();
        } else if let Some(regions) = damage.regions() {
            for r in regions {
                self.mark_damage(r);
            }
        }
    }

    /// Compute which tiles a quad instance overlaps, returning tile indices.
    pub fn tiles_for_rect(&self, x: f32, y: f32, w: f32, h: f32) -> impl Iterator<Item = TileIndex> {
        let col_start = (x.max(0.0) as u32 / TILE_SIZE).min(self.cols as u32 - 1) as u16;
        let col_end = (((x + w).ceil() as u32 + TILE_SIZE - 1) / TILE_SIZE).min(self.cols as u32) as u16;
        let row_start = (y.max(0.0) as u32 / TILE_SIZE).min(self.rows.saturating_sub(1) as u32) as u16;
        let row_end = (((y + h).ceil() as u32 + TILE_SIZE - 1) / TILE_SIZE).min(self.rows as u32) as u16;
        let cols = self.cols;

        (row_start..row_end).flat_map(move |row| {
            (col_start..col_end).map(move |col| TileIndex(row * cols + col))
        })
    }

    /// Begin a new frame: reset dirty flags to clean, then apply damage.
    pub fn begin_frame(&mut self, damage: &DamageTracker) {
        self.dirty.fill(false);
        self.apply_damage(damage);
        self.generation += 1;
    }

    /// Store fresh instances for a dirty tile.
    pub fn store_tile(&mut self, idx: TileIndex, instances: Vec<crate::primitive::QuadInstance>) {
        if let Some(cache) = self.cache.get_mut(idx.0 as usize) {
            cache.instances = instances;
            cache.generation = self.generation;
        }
    }

    /// Get cached instances for a clean tile.
    pub fn cached_instances(&self, idx: TileIndex) -> &[crate::primitive::QuadInstance] {
        self.cache
            .get(idx.0 as usize)
            .map(|c| c.instances.as_slice())
            .unwrap_or(&[])
    }

    /// Count dirty tiles this frame.
    pub fn dirty_count(&self) -> usize {
        self.dirty.iter().filter(|&&d| d).count()
    }

    /// Finalize: merge dirty tile buckets into the tile cache, replay clean tiles,
    /// and return the merged instance list.
    pub fn finalize(
        &mut self,
        tile_buckets: &mut [Vec<crate::primitive::QuadInstance>],
        overlay_instances: &[crate::primitive::QuadInstance],
    ) -> Vec<crate::primitive::QuadInstance> {
        let mut merged = Vec::new();

        for idx in 0..self.tile_count() {
            if self.dirty[idx] {
                // Dirty tile: take fresh instances from bucket, store in cache.
                let fresh = std::mem::take(&mut tile_buckets[idx]);
                merged.extend_from_slice(&fresh);
                self.cache[idx].instances = fresh;
                self.cache[idx].generation = self.generation;
            } else {
                // Clean tile: replay cached instances.
                merged.extend_from_slice(&self.cache[idx].instances);
            }
        }

        // Overlay instances are always appended (bypass tile routing).
        merged.extend_from_slice(overlay_instances);

        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── DamageRect ──

    #[test]
    fn union_of_disjoint_rects() {
        let a = DamageRect::new(0.0, 0.0, 10.0, 10.0);
        let b = DamageRect::new(20.0, 20.0, 5.0, 5.0);
        let u = a.union(b);
        assert_eq!(u.x, 0.0);
        assert_eq!(u.y, 0.0);
        assert_eq!(u.width, 25.0);
        assert_eq!(u.height, 25.0);
    }

    #[test]
    fn union_of_overlapping_rects() {
        let a = DamageRect::new(0.0, 0.0, 10.0, 10.0);
        let b = DamageRect::new(5.0, 5.0, 10.0, 10.0);
        let u = a.union(b);
        assert_eq!(u.x, 0.0);
        assert_eq!(u.y, 0.0);
        assert_eq!(u.width, 15.0);
        assert_eq!(u.height, 15.0);
    }

    #[test]
    fn union_is_commutative() {
        let a = DamageRect::new(1.0, 2.0, 3.0, 4.0);
        let b = DamageRect::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(a.union(b), b.union(a));
    }

    #[test]
    fn intersect_overlapping() {
        let a = DamageRect::new(0.0, 0.0, 10.0, 10.0);
        let b = DamageRect::new(5.0, 5.0, 10.0, 10.0);
        let i = a.intersect(b).unwrap();
        assert_eq!(i.x, 5.0);
        assert_eq!(i.y, 5.0);
        assert_eq!(i.width, 5.0);
        assert_eq!(i.height, 5.0);
    }

    #[test]
    fn intersect_disjoint_returns_none() {
        let a = DamageRect::new(0.0, 0.0, 10.0, 10.0);
        let b = DamageRect::new(20.0, 20.0, 5.0, 5.0);
        assert!(a.intersect(b).is_none());
    }

    #[test]
    fn intersect_touching_edge_returns_none() {
        let a = DamageRect::new(0.0, 0.0, 10.0, 10.0);
        let b = DamageRect::new(10.0, 0.0, 10.0, 10.0);
        assert!(a.intersect(b).is_none());
    }

    #[test]
    fn intersect_contained() {
        let outer = DamageRect::new(0.0, 0.0, 100.0, 100.0);
        let inner = DamageRect::new(10.0, 10.0, 5.0, 5.0);
        let i = outer.intersect(inner).unwrap();
        assert_eq!(i, inner);
    }

    #[test]
    fn intersect_is_commutative() {
        let a = DamageRect::new(0.0, 0.0, 10.0, 10.0);
        let b = DamageRect::new(5.0, 5.0, 10.0, 10.0);
        assert_eq!(a.intersect(b), b.intersect(a));
    }

    #[test]
    fn contains_point_inside() {
        let r = DamageRect::new(10.0, 10.0, 20.0, 20.0);
        assert!(r.contains_point(15.0, 15.0));
        assert!(r.contains_point(10.0, 10.0)); // top-left corner inclusive
    }

    #[test]
    fn contains_point_outside() {
        let r = DamageRect::new(10.0, 10.0, 20.0, 20.0);
        assert!(!r.contains_point(5.0, 15.0));
        assert!(!r.contains_point(15.0, 5.0));
        assert!(!r.contains_point(35.0, 15.0));
        assert!(!r.contains_point(15.0, 35.0));
    }

    #[test]
    fn contains_point_right_bottom_edge_exclusive() {
        let r = DamageRect::new(0.0, 0.0, 10.0, 10.0);
        // Right and bottom edges are exclusive (standard half-open rectangle).
        assert!(!r.contains_point(10.0, 5.0));
        assert!(!r.contains_point(5.0, 10.0));
    }

    // ── DamageTracker ──

    #[test]
    fn tracker_starts_with_full_invalidation() {
        let t = DamageTracker::new();
        assert!(t.is_full_invalidation());
    }

    #[test]
    fn tracker_empty_after_reset() {
        let mut t = DamageTracker::new();
        t.reset();
        assert!(!t.is_full_invalidation());
        assert_eq!(t.regions().unwrap().len(), 0);
    }

    #[test]
    fn tracker_add_regions() {
        let mut t = DamageTracker::new();
        t.reset(); // clear initial full invalidation
        t.add(DamageRect::new(0.0, 0.0, 10.0, 10.0));
        t.add(DamageRect::new(5.0, 5.0, 10.0, 10.0));
        assert_eq!(t.regions().unwrap().len(), 2);
    }

    #[test]
    fn tracker_full_invalidation() {
        let mut t = DamageTracker::new();
        t.add(DamageRect::new(0.0, 0.0, 10.0, 10.0));
        t.invalidate_all();
        assert!(t.is_full_invalidation());
        assert!(t.regions().is_none());
    }

    #[test]
    fn tracker_add_after_invalidation_is_ignored() {
        let mut t = DamageTracker::new();
        t.invalidate_all();
        t.add(DamageRect::new(0.0, 0.0, 10.0, 10.0));
        // Region was not added because we're in full invalidation.
        assert!(t.regions().is_none());
    }

    #[test]
    fn tracker_reset_clears_state() {
        let mut t = DamageTracker::new();
        t.invalidate_all();
        t.reset();
        assert!(!t.is_full_invalidation());
        assert_eq!(t.regions().unwrap().len(), 0);
    }

    // ── TileGrid ──

    #[test]
    fn tile_grid_dimensions() {
        let grid = TileGrid::new(1920, 1080);
        assert_eq!(grid.cols(), 15); // 1920/128 = 15
        assert_eq!(grid.rows(), 9);  // ceil(1080/128) = 9
        assert_eq!(grid.tile_count(), 135);
    }

    #[test]
    fn tile_grid_starts_all_dirty() {
        let grid = TileGrid::new(256, 256);
        assert_eq!(grid.dirty_count(), 4); // 2x2
    }

    #[test]
    fn tile_grid_begin_frame_clears_then_applies_damage() {
        let mut grid = TileGrid::new(512, 512);
        let mut damage = DamageTracker::new();
        damage.reset(); // clear initial full invalidation
        damage.add(DamageRect::new(0.0, 0.0, 10.0, 10.0)); // touches tile (0,0)
        grid.begin_frame(&damage);
        assert_eq!(grid.dirty_count(), 1);
        assert!(grid.is_dirty(TileIndex(0)));
        assert!(!grid.is_dirty(TileIndex(1)));
    }

    #[test]
    fn tile_grid_full_invalidation_marks_all_dirty() {
        let mut grid = TileGrid::new(256, 256);
        let damage = DamageTracker::new(); // starts with full invalidation
        grid.begin_frame(&damage);
        assert_eq!(grid.dirty_count(), 4);
    }

    #[test]
    fn tile_grid_mark_damage_spans_tiles() {
        let mut grid = TileGrid::new(512, 512);
        // Damage spanning from tile (0,0) into tile (1,1)
        grid.dirty.fill(false);
        grid.mark_damage(&DamageRect::new(100.0, 100.0, 60.0, 60.0));
        // 100..160 spans tiles 0..1 in both x and y (128px tiles)
        assert!(grid.is_dirty(TileIndex(0))); // (0,0)
        assert!(grid.is_dirty(TileIndex(1))); // (1,0)
        assert!(grid.is_dirty(TileIndex(4))); // (0,1)
        assert!(grid.is_dirty(TileIndex(5))); // (1,1)
        assert!(!grid.is_dirty(TileIndex(2))); // (2,0)
    }

    #[test]
    fn tile_grid_resize_resets() {
        let mut grid = TileGrid::new(256, 256);
        assert_eq!(grid.tile_count(), 4);
        grid.resize(512, 512);
        assert_eq!(grid.tile_count(), 16);
        // After resize, all tiles should be dirty.
        assert_eq!(grid.dirty_count(), 16);
    }
}
