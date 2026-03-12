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
