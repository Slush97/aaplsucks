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
