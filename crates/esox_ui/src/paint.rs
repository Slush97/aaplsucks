//! Drawing helpers — borders, focus rings, dashed outlines.

use esox_gfx::{BorderRadius, Color, Frame, ShapeBuilder};

use crate::layout::Rect;

/// Draw a 1px solid border around a rectangle.
pub fn draw_border(frame: &mut Frame, rect: Rect, color: Color) {
    let (x, y, w, h) = (rect.x, rect.y, rect.w, rect.h);
    frame.push(ShapeBuilder::rect(x, y, w, 1.0).color(color).build());
    frame.push(ShapeBuilder::rect(x, y + h - 1.0, w, 1.0).color(color).build());
    frame.push(ShapeBuilder::rect(x, y, 1.0, h).color(color).build());
    frame.push(ShapeBuilder::rect(x + w - 1.0, y, 1.0, h).color(color).build());
}

/// Draw a rounded rectangle.
pub fn draw_rounded_rect(frame: &mut Frame, rect: Rect, color: Color, radius: f32) {
    frame.push(
        ShapeBuilder::rect(rect.x, rect.y, rect.w, rect.h)
            .color(color)
            .border_radius(BorderRadius::uniform(radius))
            .build(),
    );
}

/// Draw a focus ring (expanded rounded rect behind the widget).
pub fn draw_focus_ring(frame: &mut Frame, rect: Rect, color: Color, radius: f32, expand: f32) {
    frame.push(
        ShapeBuilder::rect(
            rect.x - expand,
            rect.y - expand,
            rect.w + expand * 2.0,
            rect.h + expand * 2.0,
        )
        .color(color)
        .border_radius(BorderRadius::uniform(radius + expand))
        .build(),
    );
}

/// Linearly interpolate between two colors by `t` in [0, 1].
pub fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    Color::new(
        a.r + (b.r - a.r) * t,
        a.g + (b.g - a.g) * t,
        a.b + (b.b - a.b) * t,
        a.a + (b.a - a.a) * t,
    )
}

/// Draw a dashed border around a rectangle.
pub fn draw_dashed_border(
    frame: &mut Frame,
    rect: Rect,
    color: Color,
    dash: f32,
    gap: f32,
    thickness: f32,
) {
    let (x, y, w, h) = (rect.x, rect.y, rect.w, rect.h);

    // Top and bottom edges.
    let mut dx = x;
    while dx < x + w {
        let seg_w = dash.min(x + w - dx);
        frame.push(ShapeBuilder::rect(dx, y, seg_w, thickness).color(color).build());
        frame.push(
            ShapeBuilder::rect(dx, y + h - thickness, seg_w, thickness)
                .color(color)
                .build(),
        );
        dx += dash + gap;
    }

    // Left and right edges.
    let mut dy = y;
    while dy < y + h {
        let seg_h = dash.min(y + h - dy);
        frame.push(ShapeBuilder::rect(x, dy, thickness, seg_h).color(color).build());
        frame.push(
            ShapeBuilder::rect(x + w - thickness, dy, thickness, seg_h)
                .color(color)
                .build(),
        );
        dy += dash + gap;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lerp_color_at_zero_returns_a() {
        let a = Color::new(0.2, 0.4, 0.6, 1.0);
        let b = Color::new(0.8, 0.1, 0.3, 0.5);
        let result = lerp_color(a, b, 0.0);
        assert_eq!(result, a);
    }

    #[test]
    fn lerp_color_at_one_returns_b() {
        let a = Color::new(0.2, 0.4, 0.6, 1.0);
        let b = Color::new(0.8, 0.1, 0.3, 0.5);
        let result = lerp_color(a, b, 1.0);
        assert!((result.r - b.r).abs() < 1e-6);
        assert!((result.g - b.g).abs() < 1e-6);
        assert!((result.b - b.b).abs() < 1e-6);
        assert!((result.a - b.a).abs() < 1e-6);
    }

    #[test]
    fn lerp_color_at_half_returns_midpoint() {
        let a = Color::new(0.0, 0.0, 0.0, 0.0);
        let b = Color::new(1.0, 1.0, 1.0, 1.0);
        let result = lerp_color(a, b, 0.5);
        assert!((result.r - 0.5).abs() < 1e-6);
        assert!((result.g - 0.5).abs() < 1e-6);
        assert!((result.b - 0.5).abs() < 1e-6);
        assert!((result.a - 0.5).abs() < 1e-6);
    }

    #[test]
    fn lerp_color_midpoint_non_trivial() {
        let a = Color::new(0.2, 0.4, 0.6, 0.8);
        let b = Color::new(0.8, 0.2, 0.4, 0.4);
        let result = lerp_color(a, b, 0.5);
        assert!((result.r - 0.5).abs() < 1e-6);
        assert!((result.g - 0.3).abs() < 1e-6);
        assert!((result.b - 0.5).abs() < 1e-6);
        assert!((result.a - 0.6).abs() < 1e-6);
    }
}
