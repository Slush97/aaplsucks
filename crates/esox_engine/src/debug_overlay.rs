//! F3-toggled debug overlay showing engine statistics.

use esox_gfx::mesh3d::BatchStats3D;
use esox_gfx::{BorderRadius, Color, Frame, GpuContext, RenderResources, ShapeBuilder};
use esox_platform::perf::PerfMonitor;
use esox_ui::TextRenderer;

/// Collected engine statistics for the debug overlay.
pub struct EngineStats {
    /// Total physics step time for this frame (across all ticks), in microseconds.
    pub physics_step_us: u64,
    /// Render batch statistics from the last frame.
    pub batch_stats: BatchStats3D,
    /// Number of live entities in the ECS world.
    pub entity_count: usize,
}

/// Draw the debug overlay as labeled rows in a semi-transparent panel at the top-right.
pub fn draw_debug_overlay(
    frame: &mut Frame,
    gpu: &GpuContext,
    resources: &mut RenderResources,
    text: &mut TextRenderer,
    stats: &EngineStats,
    perf: &PerfMonitor,
    viewport: (u32, u32),
) {
    let font_size = 14.0_f32;
    let line_h = text.line_height(font_size);
    let line_count = 6.0;
    let pad = 8.0;
    let panel_w = 280.0;
    let panel_h = line_h * line_count + pad * 2.0;
    let margin = 8.0;

    let px = viewport.0 as f32 - panel_w - margin;
    let py = margin;

    // Background panel.
    frame.push(
        ShapeBuilder::rect(px, py, panel_w, panel_h)
            .color(Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.55,
            })
            .border_radius(BorderRadius::uniform(4.0))
            .build(),
    );

    // Text rows.
    let text_color = Color {
        r: 0.9,
        g: 0.95,
        b: 1.0,
        a: 1.0,
    };
    let tx = px + pad;
    let mut ty = py + pad;

    let lines = [
        format!("FPS: {:.0}", perf.fps),
        format!("Frame: {:.2} ms", perf.cpu_time_avg_ms),
        format!(
            "Draw calls: {}  Tris: {}",
            stats.batch_stats.draw_calls, stats.batch_stats.total_triangles,
        ),
        format!("Instances: {}", stats.batch_stats.total_instances),
        format!("Entities: {}", stats.entity_count),
        format!("Physics: {} us", stats.physics_step_us),
    ];

    for line in &lines {
        text.draw_ui_text(line, tx, ty, text_color, frame, gpu, resources);
        ty += line_h;
    }
}
