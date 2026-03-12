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
            full_invalidation: false,
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
    fn tracker_starts_empty() {
        let t = DamageTracker::new();
        assert!(!t.is_full_invalidation());
        assert_eq!(t.regions().unwrap().len(), 0);
    }

    #[test]
    fn tracker_add_regions() {
        let mut t = DamageTracker::new();
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
}
