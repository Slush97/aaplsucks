//! Layout primitives — Rect, cursor-based layout, direction, constraints.

/// Layout constraints — min/max sizes and aspect ratio.
#[derive(Debug, Clone, Copy, Default)]
pub struct Constraints {
    pub min_width: Option<f32>,
    pub max_width: Option<f32>,
    pub min_height: Option<f32>,
    pub max_height: Option<f32>,
    pub aspect_ratio: Option<f32>, // w/h
}

impl Constraints {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn min_width(mut self, v: f32) -> Self {
        self.min_width = Some(v);
        self
    }

    pub fn max_width(mut self, v: f32) -> Self {
        self.max_width = Some(v);
        self
    }

    pub fn min_height(mut self, v: f32) -> Self {
        self.min_height = Some(v);
        self
    }

    pub fn max_height(mut self, v: f32) -> Self {
        self.max_height = Some(v);
        self
    }

    pub fn aspect_ratio(mut self, ratio: f32) -> Self {
        self.aspect_ratio = Some(ratio);
        self
    }

    /// Clamp dimensions, then enforce aspect ratio (hard constraints win).
    pub fn apply(&self, w: f32, h: f32) -> (f32, f32) {
        // Apply hard clamps (min wins over max on conflict).
        let mut cw = w;
        if let Some(max) = self.max_width {
            cw = cw.min(max);
        }
        if let Some(min) = self.min_width {
            cw = cw.max(min);
        }

        let mut ch = h;
        if let Some(max) = self.max_height {
            ch = ch.min(max);
        }
        if let Some(min) = self.min_height {
            ch = ch.max(min);
        }

        // Best-effort aspect ratio after hard clamps.
        if let Some(ratio) = self.aspect_ratio {
            if ratio > 0.0 {
                let desired_h = cw / ratio;
                let desired_w = ch * ratio;
                // Try adjusting height first, then width.
                let new_h = desired_h;
                let new_h = if let Some(max) = self.max_height {
                    new_h.min(max)
                } else {
                    new_h
                };
                let new_h = if let Some(min) = self.min_height {
                    new_h.max(min)
                } else {
                    new_h
                };
                if (new_h - desired_h).abs() < 0.5 {
                    ch = new_h;
                } else {
                    // Adjust width instead.
                    let new_w = desired_w;
                    let new_w = if let Some(max) = self.max_width {
                        new_w.min(max)
                    } else {
                        new_w
                    };
                    let new_w = if let Some(min) = self.min_width {
                        new_w.max(min)
                    } else {
                        new_w
                    };
                    cw = new_w;
                    ch = cw / ratio;
                }
            }
        }

        (cw, ch)
    }
}

/// A positioned rectangle in logical pixels.
#[derive(Debug, Clone, Copy, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    /// Whether the point (px, py) is inside this rect.
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px < self.x + self.w && py >= self.y && py < self.y + self.h
    }

    /// Intersect two rects. Returns `None` if they don't overlap.
    pub fn intersect(&self, other: &Rect) -> Option<Rect> {
        let x0 = self.x.max(other.x);
        let y0 = self.y.max(other.y);
        let x1 = (self.x + self.w).min(other.x + other.w);
        let y1 = (self.y + self.h).min(other.y + other.h);
        if x1 > x0 && y1 > y0 {
            Some(Rect::new(x0, y0, x1 - x0, y1 - y0))
        } else {
            None
        }
    }

    /// Convert to `[x, y, w, h]` float array (for GPU clip rects).
    pub fn to_clip_array(&self) -> [f32; 4] {
        [self.x, self.y, self.w, self.h]
    }
}

/// 2D position / offset.
#[derive(Debug, Clone, Copy, Default)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

/// Layout direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Vertical,
    Horizontal,
}

/// Cross-axis alignment for flex layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Start,
    Center,
    End,
    Stretch,
}

/// Main-axis justification for flex layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Justify {
    Start,
    Center,
    End,
    SpaceBetween,
}

/// Flex item properties for per-child grow/shrink/alignment.
#[derive(Debug, Clone, Copy)]
pub struct FlexItem {
    /// Grow factor: how much extra space this child should absorb. Default 0.0.
    pub grow: f32,
    /// Shrink factor: how much this child should shrink when space is tight. Default 1.0.
    pub shrink: f32,
    /// Explicit basis size. `None` = use natural size from previous frame.
    pub basis: Option<f32>,
    /// Per-child cross-axis alignment override.
    pub align_self: Option<Align>,
    /// Per-child margin.
    pub margin: Spacing,
}

impl Default for FlexItem {
    fn default() -> Self {
        Self {
            grow: 0.0,
            shrink: 1.0,
            basis: None,
            align_self: None,
            margin: Spacing::default(),
        }
    }
}

impl FlexItem {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn grow(mut self, v: f32) -> Self {
        self.grow = v;
        self
    }

    pub fn shrink(mut self, v: f32) -> Self {
        self.shrink = v;
        self
    }

    pub fn basis(mut self, v: f32) -> Self {
        self.basis = Some(v);
        self
    }

    pub fn align_self(mut self, a: Align) -> Self {
        self.align_self = Some(a);
        self
    }

    pub fn margin(mut self, m: Spacing) -> Self {
        self.margin = m;
        self
    }
}

/// Spacing on four sides (margin or padding).
#[derive(Debug, Clone, Copy, Default)]
pub struct Spacing {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

impl Spacing {
    pub fn all(v: f32) -> Self {
        Self { top: v, right: v, bottom: v, left: v }
    }

    pub fn symmetric(horizontal: f32, vertical: f32) -> Self {
        Self { top: vertical, right: horizontal, bottom: vertical, left: horizontal }
    }

    pub fn horizontal(&self) -> f32 {
        self.left + self.right
    }

    pub fn vertical(&self) -> f32 {
        self.top + self.bottom
    }
}

/// Flex wrap mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlexWrap {
    /// No wrapping — all items on one line (default).
    NoWrap,
    /// Wrap items to next line when exceeding main axis extent.
    Wrap,
    /// Wrap in reverse order.
    WrapReverse,
}

/// Saved layout context for nested row/padding/scroll.
#[allow(dead_code)]
pub(crate) struct LayoutContext {
    pub direction: Direction,
    pub origin: Vec2,
    pub region: Rect,
    pub saved_cursor: Vec2,
    pub spacing: f32,
    /// For HStack: tallest child seen so far.
    pub max_cross: f32,
    pub clip_rect: Option<[f32; 4]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Rect::contains ──

    #[test]
    fn rect_contains_point_inside() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(r.contains(50.0, 40.0));
    }

    #[test]
    fn rect_contains_left_top_edge_included() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(r.contains(10.0, 20.0)); // exact top-left corner
    }

    #[test]
    fn rect_contains_right_edge_excluded() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        // x=110 is x+w, should be excluded (px < self.x + self.w)
        assert!(!r.contains(110.0, 40.0));
    }

    #[test]
    fn rect_contains_bottom_edge_excluded() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        // y=70 is y+h, should be excluded
        assert!(!r.contains(50.0, 70.0));
    }

    #[test]
    fn rect_contains_outside() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(!r.contains(5.0, 25.0));   // left of rect
        assert!(!r.contains(50.0, 15.0));  // above rect
        assert!(!r.contains(200.0, 40.0)); // right of rect
        assert!(!r.contains(50.0, 80.0));  // below rect
    }

    // ── Rect::intersect ──

    #[test]
    fn rect_intersect_overlapping() {
        let a = Rect::new(0.0, 0.0, 100.0, 100.0);
        let b = Rect::new(50.0, 50.0, 100.0, 100.0);
        let i = a.intersect(&b).unwrap();
        assert_eq!(i.x, 50.0);
        assert_eq!(i.y, 50.0);
        assert_eq!(i.w, 50.0);
        assert_eq!(i.h, 50.0);
    }

    #[test]
    fn rect_intersect_disjoint() {
        let a = Rect::new(0.0, 0.0, 50.0, 50.0);
        let b = Rect::new(100.0, 100.0, 50.0, 50.0);
        assert!(a.intersect(&b).is_none());
    }

    #[test]
    fn rect_intersect_touching_edge_is_none() {
        // Touching at edge: a ends at x=50, b starts at x=50 => no overlap (x1 == x0, not >)
        let a = Rect::new(0.0, 0.0, 50.0, 50.0);
        let b = Rect::new(50.0, 0.0, 50.0, 50.0);
        assert!(a.intersect(&b).is_none());
    }

    #[test]
    fn rect_intersect_fully_contained() {
        let outer = Rect::new(0.0, 0.0, 200.0, 200.0);
        let inner = Rect::new(50.0, 50.0, 30.0, 30.0);
        let i = outer.intersect(&inner).unwrap();
        assert_eq!(i.x, 50.0);
        assert_eq!(i.y, 50.0);
        assert_eq!(i.w, 30.0);
        assert_eq!(i.h, 30.0);
    }

    // ── Constraints::apply ──

    #[test]
    fn constraints_none_passes_through() {
        let c = Constraints::new();
        let (w, h) = c.apply(100.0, 50.0);
        assert_eq!(w, 100.0);
        assert_eq!(h, 50.0);
    }

    #[test]
    fn constraints_min_width_clamps_up() {
        let c = Constraints::new().min_width(200.0);
        let (w, _) = c.apply(100.0, 50.0);
        assert_eq!(w, 200.0);
    }

    #[test]
    fn constraints_max_width_clamps_down() {
        let c = Constraints::new().max_width(50.0);
        let (w, _) = c.apply(100.0, 50.0);
        assert_eq!(w, 50.0);
    }

    #[test]
    fn constraints_min_wins_over_max_on_conflict() {
        // min > max: min takes priority (applied after max)
        let c = Constraints::new().min_width(200.0).max_width(100.0);
        let (w, _) = c.apply(150.0, 50.0);
        assert_eq!(w, 200.0);
    }

    #[test]
    fn constraints_min_height_clamps_up() {
        let c = Constraints::new().min_height(100.0);
        let (_, h) = c.apply(50.0, 30.0);
        assert_eq!(h, 100.0);
    }

    #[test]
    fn constraints_aspect_ratio_adjusts_height() {
        // 2:1 ratio, w=200 => h should be 100
        let c = Constraints::new().aspect_ratio(2.0);
        let (w, h) = c.apply(200.0, 50.0);
        assert_eq!(w, 200.0);
        assert_eq!(h, 100.0);
    }
}
