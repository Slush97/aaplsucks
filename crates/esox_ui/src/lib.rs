//! `esox_ui` — Immediate-mode GPU widget library for esox_platform apps.
//!
//! Widgets are method calls on `Ui` that return a `Response`. Layout is
//! cursor-based. State lives in the app. The borrow checker stays happy.

pub mod a11y;
pub mod id;
pub mod layout;
pub mod paint;
pub mod response;
pub mod rich_text;
pub mod state;
pub mod text;
pub mod theme;
mod widgets;

pub use id::{fnv1a_mix, fnv1a_runtime, HOVER_SALT};
pub use layout::{Constraints, Rect};
pub use paint::lerp_color;
pub use response::Response;
pub use rich_text::{RichText, Span};
pub use state::{
    A11yRole, A11yTree, DragPayload, DropZoneState, Easing, InputState, ModalAction, SelectState,
    SortDirection, TabState, TableState, ToastKind, ToastQueue, TooltipState, TreeState, UiState,
    VirtualScrollState, WidgetKind,
};
pub use text::TextRenderer;
pub use theme::{Theme, ThemeBuilder, ThemeTransition};
pub use widgets::table::{ColumnWidth, TableColumn};
pub use widgets::tree::TreeNodeResponse;

use esox_gfx::{Frame, GpuContext, RenderResources};
use layout::{Direction, LayoutContext, Vec2};

/// The main UI context. Created each frame, consumed by `finish()`.
pub struct Ui<'f> {
    pub(crate) frame: &'f mut Frame,
    pub(crate) gpu: &'f GpuContext,
    pub(crate) resources: &'f mut RenderResources,
    pub(crate) text: &'f mut TextRenderer,
    pub(crate) state: &'f mut UiState,
    pub(crate) theme: &'f Theme,

    // Layout cursor.
    cursor: Vec2,
    region: Rect,
    layout_stack: Vec<LayoutContext>,
    spacing: f32,
    /// Active hit-test clip rect. Widgets outside this rect won't receive clicks.
    hit_clip: Option<Rect>,
    /// Whether widgets are currently disabled (no interaction).
    disabled: bool,
}

impl<'f> Ui<'f> {
    /// Begin a new UI frame within the given viewport rectangle.
    pub fn begin(
        frame: &'f mut Frame,
        gpu: &'f GpuContext,
        resources: &'f mut RenderResources,
        text: &'f mut TextRenderer,
        state: &'f mut UiState,
        theme: &'f Theme,
        viewport: Rect,
    ) -> Self {
        text.advance_generation();
        state.begin_frame();

        Self {
            frame,
            gpu,
            resources,
            text,
            state,
            theme,
            cursor: Vec2 {
                x: viewport.x,
                y: viewport.y,
            },
            region: viewport,
            layout_stack: Vec::new(),
            spacing: theme.padding,
            hit_clip: None,
            disabled: false,
        }
    }

    /// Finish the frame — draw modals, overlays, toasts, tooltips, clean up per-frame state.
    /// Returns any overlay selection that occurred: (id, selected_index).
    pub fn finish(mut self) -> Option<(u64, usize)> {
        // Draw order: normal content (already drawn) → modals → dropdowns → toasts → tooltips
        self.draw_modals();
        let selection = self.draw_overlay();
        self.draw_toasts();
        self.draw_tooltip();
        self.state.end_frame();
        selection
    }

    // ── Layout ──

    /// Allocate a rectangle in the current layout direction.
    pub fn allocate_rect(&mut self, w: f32, h: f32) -> Rect {
        let rect = match self.layout_stack.last() {
            Some(ctx) if ctx.direction == Direction::Horizontal => {
                let r = Rect::new(self.cursor.x, self.cursor.y, w, h);
                self.cursor.x += w + self.spacing;
                r
            }
            _ => {
                // Vertical: respect the requested width, clamped to the region.
                let actual_w = w.min(self.region.w);
                let r = Rect::new(self.cursor.x, self.cursor.y, actual_w, h);
                self.cursor.y += h + self.spacing;
                r
            }
        };
        // Track max_cross for horizontal layouts.
        if let Some(ctx) = self.layout_stack.last_mut() {
            if ctx.direction == Direction::Horizontal && h > ctx.max_cross {
                ctx.max_cross = h;
            }
        }
        rect
    }

    /// Run a closure in a horizontal row layout.
    pub fn row(&mut self, f: impl FnOnce(&mut Self)) {
        let ctx = LayoutContext {
            direction: Direction::Horizontal,
            origin: self.cursor,
            region: self.region,
            saved_cursor: self.cursor,
            spacing: self.spacing,
            max_cross: 0.0,
            clip_rect: None,
        };
        self.layout_stack.push(ctx);
        f(self);
        let ctx = self.layout_stack.pop().unwrap();
        // Restore cursor to below the tallest child.
        self.cursor.x = ctx.saved_cursor.x;
        self.cursor.y = ctx.saved_cursor.y + ctx.max_cross + self.spacing;
    }

    /// Set spacing between subsequent widgets.
    pub fn spacing(&mut self, amount: f32) {
        self.spacing = amount;
    }

    /// Add extra vertical space.
    pub fn add_space(&mut self, amount: f32) {
        self.cursor.y += amount;
    }

    /// Run a closure within a max-width container, centered horizontally.
    pub fn max_width(&mut self, max_w: f32, f: impl FnOnce(&mut Self)) {
        let col_w = self.region.w.min(max_w);
        let col_x = self.cursor.x + (self.region.w - col_w) / 2.0;

        let saved_cursor = self.cursor;
        let saved_region = self.region;

        self.cursor.x = col_x;
        self.region = Rect::new(col_x, self.cursor.y, col_w, self.region.h);

        f(self);

        let new_y = self.cursor.y;
        self.cursor = saved_cursor;
        self.cursor.y = new_y;
        self.region = saved_region;
    }

    /// Run a closure with padding on all sides.
    pub fn padding(&mut self, amount: f32, f: impl FnOnce(&mut Self)) {
        let saved_cursor = self.cursor;
        let saved_region = self.region;

        self.cursor.x += amount;
        self.cursor.y += amount;
        self.region = Rect::new(
            self.cursor.x,
            self.cursor.y,
            self.region.w - amount * 2.0,
            self.region.h - amount * 2.0,
        );

        f(self);

        let new_y = self.cursor.y + amount;
        self.cursor = saved_cursor;
        self.cursor.y = new_y;
        self.region = saved_region;
    }

    /// Get the current cursor X position.
    pub fn cursor_x(&self) -> f32 {
        self.cursor.x
    }

    /// Get the current cursor Y position (useful for tracking content height).
    pub fn cursor_y(&self) -> f32 {
        self.cursor.y
    }

    /// Get the current region width.
    pub fn region_width(&self) -> f32 {
        self.region.w
    }

    /// Narrow the region: offset cursor.x and reduce region.w.
    /// Useful for centering content without a closure.
    pub fn indent(&mut self, offset: f32, width: f32) {
        self.cursor.x += offset;
        self.region = Rect::new(self.cursor.x, self.region.y, width, self.region.h);
    }

    // ── Flex/Weighted Columns ──

    /// Weighted column layout. Calls `f(ui, col_index)` for each column.
    /// Weights are relative: &[2.0, 1.0] -> 2/3 + 1/3 of available width.
    pub fn columns(&mut self, weights: &[f32], f: impl FnMut(&mut Self, usize)) {
        self.columns_spaced(0.0, weights, f);
    }

    /// Same as `columns` with explicit inter-column gap.
    pub fn columns_spaced(&mut self, gap: f32, weights: &[f32], mut f: impl FnMut(&mut Self, usize)) {
        if weights.is_empty() {
            return;
        }
        let total_weight: f32 = weights.iter().sum();
        if total_weight <= 0.0 {
            return;
        }

        let n = weights.len();
        let total_gap = gap * (n as f32 - 1.0).max(0.0);
        let available = self.region.w - total_gap;

        let saved_cursor = self.cursor;
        let saved_region = self.region;
        let saved_spacing = self.spacing;

        let mut col_x = self.cursor.x;
        let mut max_height: f32 = 0.0;

        for (i, &w) in weights.iter().enumerate() {
            let col_w = available * w / total_weight;

            self.cursor = Vec2 { x: col_x, y: saved_cursor.y };
            self.region = Rect::new(col_x, saved_region.y, col_w, saved_region.h);
            self.spacing = saved_spacing;

            let start_y = self.cursor.y;
            f(self, i);
            let col_height = self.cursor.y - start_y;
            if col_height > max_height {
                max_height = col_height;
            }

            col_x += col_w + gap;
        }

        self.cursor = saved_cursor;
        self.cursor.y += max_height;
        self.region = saved_region;
        self.spacing = saved_spacing;
    }

    // ── Constrained layout ──

    /// Run a closure within layout constraints.
    pub fn constrained(&mut self, c: layout::Constraints, f: impl FnOnce(&mut Self)) {
        let saved_cursor = self.cursor;
        let saved_region = self.region;

        let (cw, _) = c.apply(self.region.w, self.region.h);
        self.region = Rect::new(self.cursor.x, self.cursor.y, cw, self.region.h);

        f(self);

        let consumed_h = self.cursor.y - saved_cursor.y;
        let (_, ch) = c.apply(cw, consumed_h);

        self.cursor.x = saved_cursor.x;
        self.cursor.y = saved_cursor.y + ch;
        self.region = saved_region;
    }

    // ── Tree indent ──

    /// Indent children of an expanded tree node.
    pub fn tree_indent(&mut self, f: impl FnOnce(&mut Self)) {
        let indent = self.theme.tree_indent;
        let saved_cursor_x = self.cursor.x;
        let saved_region = self.region;

        self.cursor.x += indent;
        self.region = Rect::new(
            self.cursor.x,
            self.region.y,
            self.region.w - indent,
            self.region.h,
        );

        f(self);

        self.cursor.x = saved_cursor_x;
        self.region = saved_region;
    }

    // ── Drag and Drop ──

    /// Make a widget draggable. Call after the widget.
    /// Returns true when drag starts this frame.
    pub fn drag_source(&mut self, id: u64, payload: u64) -> bool {
        if self.disabled {
            return false;
        }

        // On mouse press, record drag start position.
        if let Some((cx, cy, _)) = self.state.mouse.pending_click {
            // Check if click is on this widget.
            if let Some((rect, _, _)) = self.state.hit_rects.iter().find(|(_, wid, _)| *wid == id) {
                if rect.contains(cx, cy) && self.state.drag.is_none() {
                    self.state.drag_start = Some((cx, cy));
                }
            }
        }

        // Check dead zone — start drag when mouse moves >4px from press.
        if self.state.drag.is_none() && self.state.mouse_pressed {
            if let Some((sx, sy)) = self.state.drag_start {
                if let Some((rect, _, _)) = self.state.hit_rects.iter().find(|(_, wid, _)| *wid == id) {
                    let dx = self.state.mouse.x - sx;
                    let dy = self.state.mouse.y - sy;
                    if dx * dx + dy * dy > 16.0 {
                        self.state.drag = Some(state::DragPayload {
                            source_id: id,
                            payload,
                            x: self.state.mouse.x,
                            y: self.state.mouse.y,
                            offset_x: sx - rect.x,
                            offset_y: sy - rect.y,
                        });
                        return true;
                    }
                }
            }
        }

        // Update drag position.
        if let Some(ref mut d) = self.state.drag {
            if d.source_id == id {
                d.x = self.state.mouse.x;
                d.y = self.state.mouse.y;
            }
        }

        false
    }

    /// Check if a drag is hovering over this rect. Returns payload if so.
    pub fn drop_target(&self, rect: Rect) -> Option<u64> {
        if let Some(ref d) = self.state.drag {
            if rect.contains(d.x, d.y) {
                return Some(d.payload);
            }
        }
        None
    }

    /// Check if a drop just completed on this rect. Returns payload.
    /// Only returns Some on the frame when mouse was released over target.
    pub fn accept_drop(&self, rect: Rect) -> Option<u64> {
        if let Some(ref d) = self.state.drag {
            if !self.state.mouse_pressed && rect.contains(d.x, d.y) {
                return Some(d.payload);
            }
        }
        None
    }

    // ── Interaction helpers (used by widgets) ──

    /// Register a widget for hit testing and focus chain.
    ///
    /// When `hit_clip` is active, the hit rect is intersected with it so
    /// widgets scrolled out of view don't receive clicks. The widget is
    /// still added to the focus chain (Tab still works).
    ///
    /// When disabled, skips both hit_rects and focus_chain — no cursor
    /// icon change, no Tab focus, no click consumption.
    pub(crate) fn register_widget(
        &mut self,
        id: u64,
        rect: Rect,
        kind: state::WidgetKind,
    ) {
        if self.disabled {
            return;
        }
        if let Some(clip) = &self.hit_clip {
            if let Some(clipped) = rect.intersect(clip) {
                self.state.hit_rects.push((clipped, id, kind));
            }
            // Skip hit_rects push if no intersection, but still add to focus chain.
        } else {
            self.state.hit_rects.push((rect, id, kind));
        }
        self.state.focus_chain.push(id);
    }

    /// Compute the Response for a widget given its ID and rect.
    /// When disabled, returns an inert Response with `disabled: true`.
    pub(crate) fn widget_response(&mut self, id: u64, rect: Rect) -> response::Response {
        if self.disabled {
            return response::Response {
                clicked: false,
                right_clicked: false,
                hovered: false,
                focused: false,
                changed: false,
                disabled: true,
            };
        }
        // Intersect with hit_clip so widgets outside the visible scroll area
        // don't respond to hover/click.
        let effective = match &self.hit_clip {
            Some(clip) => match rect.intersect(clip) {
                Some(r) => r,
                None => {
                    // Completely clipped — not hovered, not clickable.
                    return response::Response {
                        clicked: false,
                        right_clicked: false,
                        hovered: false,
                        focused: self.state.focused == Some(id),
                        changed: false,
                        disabled: false,
                    };
                }
            },
            None => rect,
        };
        let hovered = effective.contains(self.state.mouse.x, self.state.mouse.y);
        let focused = self.state.focused == Some(id);

        let mut clicked = false;
        if let Some((cx, cy, ref mut consumed)) = self.state.mouse.pending_click {
            if !*consumed && effective.contains(cx, cy) {
                clicked = true;
                *consumed = true;
                self.state.focused = Some(id);
                self.state.reset_blink();
            }
        }

        let mut right_clicked = false;
        if let Some((cx, cy, ref mut consumed)) = self.state.mouse.pending_right_click {
            if !*consumed && effective.contains(cx, cy) {
                right_clicked = true;
                *consumed = true;
            }
        }

        response::Response {
            clicked,
            right_clicked,
            hovered,
            focused,
            changed: false,
            disabled: false,
        }
    }

    /// Check if a point is hovered over a rect.
    pub fn is_hovered(&self, rect: Rect) -> bool {
        rect.contains(self.state.mouse.x, self.state.mouse.y)
    }

    /// Set the disabled flag directly.
    pub fn set_disabled(&mut self, disabled: bool) {
        self.disabled = disabled;
    }

    /// Run a closure with widgets disabled (or enabled). Restores previous state after.
    pub fn disabled(&mut self, disabled: bool, f: impl FnOnce(&mut Self)) {
        let prev = self.disabled;
        self.disabled = disabled;
        f(self);
        self.disabled = prev;
    }

    /// Whether the UI is currently in disabled mode.
    pub fn is_disabled(&self) -> bool {
        self.disabled
    }

    /// Access the theme.
    pub fn theme(&self) -> &Theme {
        self.theme
    }

    // ── Tooltip ──

    /// Attach a tooltip to the widget with the given ID. Call after the widget.
    pub fn tooltip(&mut self, id: u64, text: &str) {
        // Find widget rect from hit_rects.
        let anchor = match self.state.hit_rects.iter().find(|(_, wid, _)| *wid == id) {
            Some((rect, _, _)) => *rect,
            None => return, // disabled or not found
        };

        let hovered = anchor.contains(self.state.mouse.x, self.state.mouse.y);

        if hovered {
            match &mut self.state.tooltip {
                Some(tt) if tt.widget_id == id => {
                    // Same widget — check delay.
                    if !tt.visible {
                        let elapsed = tt.hover_start.elapsed().as_millis() as u64;
                        if elapsed >= self.theme.tooltip_delay_ms {
                            tt.visible = true;
                        }
                    }
                    tt.anchor = anchor;
                }
                _ => {
                    // New widget or no tooltip — reset timer.
                    self.state.tooltip = Some(state::TooltipState {
                        widget_id: id,
                        hover_start: std::time::Instant::now(),
                        anchor,
                        text: text.to_string(),
                        visible: false,
                    });
                }
            }
        } else if self.state.tooltip.as_ref().is_some_and(|tt| tt.widget_id == id) {
            self.state.tooltip = None;
        }
    }

    /// Draw the tooltip if visible. Called from `finish()`.
    fn draw_tooltip(&mut self) {
        let (text, anchor) = match &self.state.tooltip {
            Some(tt) if tt.visible => (tt.text.clone(), tt.anchor),
            _ => return,
        };

        let font_size = self.theme.tooltip_font_size;
        let pad = self.theme.tooltip_padding;
        let text_w = self.text.measure_text(&text, font_size);
        let tooltip_w = text_w + pad * 2.0;
        let tooltip_h = font_size + pad * 2.0;

        // Position below the anchor, centered, clamped to viewport.
        let gap = 4.0;
        let mut tx = anchor.x + (anchor.w - tooltip_w) / 2.0;
        let mut ty = anchor.y + anchor.h + gap;

        // Clamp to viewport.
        if tx < self.region.x {
            tx = self.region.x;
        }
        if tx + tooltip_w > self.region.x + self.region.w {
            tx = self.region.x + self.region.w - tooltip_w;
        }
        if ty + tooltip_h > self.region.y + self.region.h {
            // Show above instead.
            ty = anchor.y - tooltip_h - gap;
        }

        let tt_rect = Rect::new(tx, ty, tooltip_w, tooltip_h);

        // Shadow.
        paint::draw_rounded_rect(
            self.frame,
            Rect::new(tx + 1.0, ty + 1.0, tooltip_w, tooltip_h),
            self.theme.shadow,
            4.0,
        );

        // Background.
        paint::draw_rounded_rect(self.frame, tt_rect, self.theme.tooltip_bg, 4.0);

        // Text.
        self.text.draw_text(
            &text,
            tx + pad,
            ty + pad,
            font_size,
            self.theme.tooltip_fg,
            self.frame,
            self.gpu,
            self.resources,
        );
    }

    // ── Context Menu ──

    // ── Toast convenience ──

    /// Show an info toast notification.
    pub fn toast_info(&mut self, msg: &str) {
        let dur = self.theme.toast_duration_ms;
        self.state.toasts.push(state::ToastKind::Info, msg.to_string(), dur);
    }

    /// Show a success toast notification.
    pub fn toast_success(&mut self, msg: &str) {
        let dur = self.theme.toast_duration_ms;
        self.state.toasts.push(state::ToastKind::Success, msg.to_string(), dur);
    }

    /// Show an error toast notification.
    pub fn toast_error(&mut self, msg: &str) {
        let dur = self.theme.toast_duration_ms;
        self.state.toasts.push(state::ToastKind::Error, msg.to_string(), dur);
    }

    /// Show a warning toast notification.
    pub fn toast_warning(&mut self, msg: &str) {
        let dur = self.theme.toast_duration_ms;
        self.state.toasts.push(state::ToastKind::Warning, msg.to_string(), dur);
    }

    /// Show a toast with custom kind and duration.
    pub fn toast_custom(&mut self, kind: state::ToastKind, msg: &str, duration_ms: u64) {
        self.state.toasts.push(kind, msg.to_string(), duration_ms);
    }

    // ── Accessibility ──

    /// Set a pending accessibility label for the next widget.
    pub fn a11y_label(&mut self, label: &str) {
        if self.state.a11y_enabled {
            self.state.a11y_pending_label = Some(label.to_string());
        }
    }

    /// Set a pending accessibility role for the next widget.
    pub fn a11y_role(&mut self, role: state::A11yRole) {
        if self.state.a11y_enabled {
            self.state.a11y_pending_role = Some(role);
        }
    }

    /// Consume pending a11y label/role (called by widgets after register_widget).
    #[allow(dead_code)]
    pub(crate) fn consume_a11y(&mut self) -> (Option<String>, Option<state::A11yRole>) {
        (
            self.state.a11y_pending_label.take(),
            self.state.a11y_pending_role.take(),
        )
    }

    /// Push an accessibility node into the frame's a11y tree.
    ///
    /// Widgets call this after `register_widget` to emit their a11y representation.
    /// If a11y is disabled, this is a no-op.
    pub(crate) fn push_a11y_node(&mut self, node: state::A11yNode) {
        if self.state.a11y_enabled {
            self.state.a11y_tree.push(node);
        }
    }

    // ── Context Menu ──

    /// Open a context menu at the current mouse position. Call when `response.right_clicked`.
    pub fn context_menu(&mut self, id: u64, items: &[&str]) {
        let mx = self.state.mouse.x;
        let my = self.state.mouse.y;

        // Measure menu width.
        let pad = self.theme.input_padding;
        let font_size = self.theme.font_size;
        let mut max_w: f32 = 0.0;
        for item in items {
            let w = self.text.measure_text(item, font_size);
            if w > max_w {
                max_w = w;
            }
        }
        let menu_w = (max_w + pad * 2.0).max(self.theme.context_menu_min_w);
        let menu_h = items.len() as f32 * self.theme.item_height;

        // Clamp to viewport.
        let mut px = mx;
        let mut py = my;
        if px + menu_w > self.region.x + self.region.w {
            px = self.region.x + self.region.w - menu_w;
        }
        if py + menu_h > self.region.y + self.region.h {
            py = self.region.y + self.region.h - menu_h;
        }

        self.state.overlay = Some(state::Overlay::ContextMenu {
            id,
            position: Rect::new(px, py, menu_w, menu_h),
            items: items.iter().map(|s| s.to_string()).collect(),
            hovered: None,
        });
    }
}
