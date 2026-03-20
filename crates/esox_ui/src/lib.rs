//! `esox_ui` — Immediate-mode GPU widget library for esox_platform apps.
//!
//! ## Pattern
//!
//! Widgets are method calls on [`Ui`] that return a [`Response`]. Layout is
//! cursor-based (vertical by default, horizontal via [`Ui::row`]). All mutable
//! state lives in the app; the library stores nothing between frames.
//!
//! ```ignore
//! let mut ui = Ui::begin(&mut frame, &gpu, &mut resources, &mut text, &mut state, &theme, viewport);
//!
//! ui.label("Hello");
//! if ui.button(id!("click"), "Click me").clicked {
//!     count += 1;
//! }
//! ui.text_input(id!("name"), &mut name, "placeholder…");
//!
//! ui.finish();
//! ```
//!
//! ## Layout model
//!
//! - **Vertical** (default): widgets stack top-to-bottom
//! - **Horizontal**: `ui.row(|ui| { ... })` places widgets left-to-right
//! - **Columns**: `ui.columns(&[2.0, 1.0], |ui, i| { ... })` for weighted flex
//! - **Constraints**: `ui.constrained(c, |ui| { ... })` for min/max/aspect
//! - **Scroll**: `ui.scrollable(id, height, |ui| { ... })` for clipped content
//!
//! ## Animation API
//!
//! - [`Ui::animate`] — drive a value towards a target with easing
//! - [`Ui::animate_bool`] — convenience for 0→1 toggle animations
//! - [`Ui::is_animating`] — check if an animation is in-flight
//! - [`Easing`] — `Linear`, `EaseOutCubic`, `EaseInOutCubic`, `EaseOutExpo`
//! - [`lerp_color`] — interpolate between colors

pub mod a11y;
pub mod id;
pub mod layout;
pub mod layout_tree;
pub mod paint;
pub mod response;
pub mod rich_text;
pub mod state;
pub mod text;
pub mod theme;
mod widgets;

pub use id::{fnv1a_mix, fnv1a_runtime, HOVER_SALT};
pub use layout::{Align, Constraints, FlexItem, FlexWrap, Justify, Rect, Spacing};
pub use paint::lerp_color;
pub use response::Response;
pub use rich_text::{RichText, Span};
pub use state::{
    A11yNode, A11yRole, A11yTree, ClipboardProvider, DragPayload, DropZoneState, Easing, ImeState,
    InputState, ModalAction, SelectState, SortDirection, TabState, TableState, ToastKind,
    ToastQueue, TooltipState, TreeState, UiState, VirtualScrollState, WidgetKind,
};
pub use text::{TextRenderer, TruncationMode};
pub use theme::{StyleState, TextSize, Theme, ThemeBuilder, ThemeTransition, WidgetStyle};
pub use widgets::image::{ImageCache, ImageHandle};
pub use widgets::table::{ColumnWidth, TableColumn};
pub use widgets::form::FieldStatus;
pub use widgets::tree::TreeNodeResponse;
pub use widgets::menu_bar::{Menu, MenuEntry, MenuItem};

use esox_gfx::{Color, Frame, GpuContext, RenderResources};
use layout::{Direction, LayoutContext, Vec2};
use layout_tree::{LayoutStyle, LayoutTree, TreeBuildContext};

/// Responsive width classification for container queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WidthClass {
    /// Width < `theme.breakpoint_compact` (default 600px).
    Compact,
    /// Width between compact and expanded breakpoints.
    Medium,
    /// Width >= `theme.breakpoint_expanded` (default 1200px).
    Expanded,
}

/// Rendering access for custom widgets.
///
/// Obtained via [`Ui::painter()`], provides mutable access to the frame and
/// text renderer plus shared access to GPU context, resources, and theme.
pub struct Painter<'a> {
    pub frame: &'a mut Frame,
    pub text: &'a mut text::TextRenderer,
    pub gpu: &'a GpuContext,
    pub resources: &'a mut RenderResources,
    pub theme: &'a theme::Theme,
}

/// Builder for flex row/column layouts with gap, alignment, and justification.
pub struct FlexBuilder<'a, 'f> {
    ui: &'a mut Ui<'f>,
    direction: Direction,
    gap: f32,
    align: layout::Align,
    justify: layout::Justify,
    wrap: layout::FlexWrap,
}

impl<'a, 'f> FlexBuilder<'a, 'f> {
    /// Set the gap between children.
    pub fn gap(mut self, gap: f32) -> Self {
        self.gap = gap;
        self
    }

    /// Set cross-axis alignment.
    pub fn align(mut self, align: layout::Align) -> Self {
        self.align = align;
        self
    }

    /// Set main-axis justification.
    pub fn justify(mut self, justify: layout::Justify) -> Self {
        self.justify = justify;
        self
    }

    /// Set flex wrap mode.
    pub fn wrap(mut self, wrap: layout::FlexWrap) -> Self {
        self.wrap = wrap;
        self
    }

    /// Draw a flex layout with per-child grow/shrink via `FlexUi`.
    ///
    /// The tree solver handles grow/shrink distribution and justification.
    /// On frame 1 (no prev_layout), children render at natural cursor positions.
    /// On frame 2+, `allocate_rect` picks up solved positions from the tree.
    pub fn show_flex(self, id: u64, f: impl FnOnce(&mut FlexUi<'_, 'f>)) {
        self.ui.tree_build.open_container(Some(id), LayoutStyle {
            direction: self.direction,
            gap: self.gap,
            align_items: self.align,
            justify_content: self.justify,
            flex_wrap: self.wrap,
            ..Default::default()
        });

        let saved_cursor = self.ui.cursor;
        let saved_region = self.ui.region;
        let saved_spacing = self.ui.spacing;

        // Set up cursor direction for fallback positioning.
        if self.direction == Direction::Horizontal {
            self.ui.spacing = self.gap;
            let ctx = LayoutContext {
                direction: Direction::Horizontal,
                origin: self.ui.cursor,
                region: self.ui.region,
                saved_cursor: self.ui.cursor,
                spacing: self.gap,
                max_cross: 0.0,
                clip_rect: None,
            };
            self.ui.layout_stack.push(ctx);
        } else {
            self.ui.spacing = self.gap;
        }

        let mut flex_ui = FlexUi {
            ui: self.ui,
            direction: self.direction,
            child_index: 0,
        };

        f(&mut flex_ui);

        let child_count = flex_ui.child_index;

        if self.direction == Direction::Horizontal {
            let ctx = self.ui.layout_stack.pop().unwrap();
            let max_cross = ctx.max_cross;
            self.ui.cursor.x = saved_cursor.x;
            self.ui.cursor.y = saved_cursor.y + max_cross + saved_spacing;
            // If no children advanced the cursor, keep y stable.
            if child_count == 0 {
                self.ui.cursor.y = saved_cursor.y;
            }
        }

        self.ui.region = saved_region;
        self.ui.spacing = saved_spacing;
        self.ui.tree_build.close_container();
    }

    /// Draw the flex layout with the given content closure.
    ///
    /// Opens a tree container so the solver handles justification on frame 2+.
    /// Falls back to cursor-based positioning on frame 1.
    pub fn show(self, f: impl FnOnce(&mut Ui<'f>)) {
        let is_horizontal = self.direction == Direction::Horizontal;

        self.ui.tree_build.open_container(None, LayoutStyle {
            direction: self.direction,
            gap: self.gap,
            align_items: self.align,
            justify_content: self.justify,
            flex_wrap: self.wrap,
            ..Default::default()
        });

        if is_horizontal {
            let saved_spacing = self.ui.spacing;
            self.ui.spacing = self.gap;
            let ctx = LayoutContext {
                direction: Direction::Horizontal,
                origin: self.ui.cursor,
                region: self.ui.region,
                saved_cursor: self.ui.cursor,
                spacing: self.ui.spacing,
                max_cross: 0.0,
                clip_rect: None,
            };
            self.ui.layout_stack.push(ctx);
            f(self.ui);
            let ctx = self.ui.layout_stack.pop().unwrap();
            self.ui.cursor.x = ctx.saved_cursor.x;
            self.ui.cursor.y = ctx.saved_cursor.y + ctx.max_cross + saved_spacing;
            self.ui.spacing = saved_spacing;
        } else {
            let saved_spacing = self.ui.spacing;
            self.ui.spacing = self.gap;
            f(self.ui);
            self.ui.spacing = saved_spacing;
        }

        self.ui.tree_build.close_container();
    }
}

/// Context for placing flex items with grow/shrink within a `FlexBuilder::show_flex()`.
pub struct FlexUi<'a, 'f> {
    ui: &'a mut Ui<'f>,
    direction: Direction,
    child_index: usize,
}

impl<'a, 'f> FlexUi<'a, 'f> {
    /// Place a flex item with default properties (no grow/shrink).
    pub fn item_default(&mut self, f: impl FnOnce(&mut Ui<'f>)) {
        self.item(layout::FlexItem::default(), f);
    }

    /// Place a flex item with explicit grow/shrink/alignment properties.
    pub fn item(&mut self, props: layout::FlexItem, f: impl FnOnce(&mut Ui<'f>)) {
        let cross_dir = if self.direction == Direction::Horizontal {
            Direction::Vertical
        } else {
            Direction::Horizontal
        };

        self.ui.tree_build.open_container(None, LayoutStyle {
            direction: cross_dir,
            flex_grow: props.grow,
            flex_shrink: props.shrink,
            flex_basis: props.basis,
            align_self: props.align_self,
            margin: props.margin,
            gap: self.ui.spacing,
            ..Default::default()
        });

        // Apply margin for cursor fallback.
        if self.direction == Direction::Horizontal {
            self.ui.cursor.x += props.margin.left;
        } else {
            self.ui.cursor.y += props.margin.top;
        }

        f(self.ui);

        if self.direction == Direction::Horizontal {
            self.ui.cursor.x += props.margin.right;
        } else {
            self.ui.cursor.y += props.margin.bottom;
        }

        self.ui.tree_build.close_container();
        self.child_index += 1;
    }

    /// Access the inner `Ui` for direct widget calls outside flex items.
    pub fn ui(&mut self) -> &mut Ui<'f> {
        self.ui
    }
}

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
    /// Style override stack for per-widget styling.
    style_stack: Vec<WidgetStyle>,
    /// Layout tree being built this frame.
    tree_build: TreeBuildContext,
    /// Solved layout tree from the previous frame (for position lookups).
    prev_layout: Option<LayoutTree>,
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

        // Set up tile grid for partial redraw if available.
        if let Some(ref mut grid) = state.tile_grid {
            grid.resize(viewport.w as u32, viewport.h as u32);
            grid.begin_frame(&state.damage);
            frame.begin_partial(grid);
        }

        // Move the solved layout tree from last frame into prev_layout.
        let prev_layout = state.layout_cache.take();

        let mut tree_build = TreeBuildContext::new();
        tree_build.open_container(Some(u64::MAX), LayoutStyle {
            direction: Direction::Vertical,
            gap: theme.padding,
            ..Default::default()
        });

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
            style_stack: Vec::new(),
            tree_build,
            prev_layout,
        }
    }

    /// Enable tile-based partial redraw. Call once (e.g. after first resize)
    /// to opt in to tile caching. The grid is created lazily from the viewport.
    pub fn enable_partial_redraw(state: &mut UiState, viewport_w: u32, viewport_h: u32) {
        state.tile_grid = Some(esox_gfx::TileGrid::new(viewport_w, viewport_h));
    }

    /// Finish the frame — draw modals, overlays, toasts, tooltips, clean up per-frame state.
    /// Returns any overlay selection that occurred: (id, selected_index).
    pub fn finish(mut self) -> Option<(u64, usize)> {
        // Switch to overlay mode so modals/toasts/tooltips bypass tile routing.
        if self.state.tile_grid.is_some() {
            self.frame.set_overlay_mode(true);
        }

        // Draw order: normal content (already drawn) → modals → dropdowns → toasts → tooltips
        self.draw_modals();
        let selection = self.draw_overlay();
        self.draw_deferred_menu_bar();
        self.draw_toasts();
        self.draw_tooltip();
        self.draw_debug_overlay();

        // Finalize tile grid: merge dirty/clean tiles + overlay.
        if let Some(ref mut grid) = self.state.tile_grid {
            self.frame.finalize_partial(grid);
        }

        // Close root container and solve the layout tree.
        self.tree_build.close_container();
        debug_assert_eq!(self.tree_build.open_stack_len(), 0, "unclosed layout container");
        let viewport = self.region;
        let mut tree = self.tree_build.tree;
        tree.solve(viewport);
        self.state.layout_cache = Some(tree);

        self.state.end_frame();
        selection
    }

    // ── Layout ──

    /// Allocate a rectangle in the current layout direction.
    pub fn allocate_rect(&mut self, w: f32, h: f32) -> Rect {
        // Record this leaf in the layout tree.
        let node_id = self.tree_build.add_leaf(None, w, h);
        let node_key = self.tree_build.tree.node(node_id).key;

        let rect = if let Some(solved) = self.prev_layout.as_ref().and_then(|t| t.lookup(node_key)) {
            // Use solved position from previous frame. Advance cursor for compatibility.
            match self.layout_stack.last() {
                Some(ctx) if ctx.direction == Direction::Horizontal => {
                    self.cursor.x = solved.x + solved.w + self.spacing;
                }
                _ => {
                    self.cursor.y = solved.y + solved.h + self.spacing;
                }
            }
            solved
        } else {
            // Cursor fallback (first frame / new widgets).
            match self.layout_stack.last() {
                Some(ctx) if ctx.direction == Direction::Horizontal => {
                    let r = Rect::new(self.cursor.x, self.cursor.y, w, h);
                    self.cursor.x += w + self.spacing;
                    r
                }
                _ => {
                    let actual_w = w.min(self.region.w);
                    let r = Rect::new(self.cursor.x, self.cursor.y, actual_w, h);
                    self.cursor.y += h + self.spacing;
                    r
                }
            }
        };

        // Track max_cross for horizontal layouts.
        if let Some(ctx) = self.layout_stack.last_mut() {
            if ctx.direction == Direction::Horizontal && rect.h > ctx.max_cross {
                ctx.max_cross = rect.h;
            }
        }
        // Debug overlay: collect layout bounds.
        if self.state.debug_overlay {
            let depth = self.layout_stack.len();
            let kind = match depth {
                0 => "root",
                1 => "child",
                _ => "nested",
            };
            self.state.debug_widget_rects.push((rect, 0, kind));
        }
        rect
    }

    /// Set a key override for the next `allocate_rect` call.
    pub fn keyed(&mut self, key: u64) {
        self.tree_build.push_key_scope(key);
    }

    /// Allocate a rectangle with an explicit key for cross-frame stability.
    pub fn allocate_rect_keyed(&mut self, key: u64, w: f32, h: f32) -> Rect {
        self.tree_build.push_key_scope(key);
        self.allocate_rect(w, h)
    }

    /// Run a closure in a horizontal row layout.
    pub fn row(&mut self, f: impl FnOnce(&mut Self)) {
        self.tree_build.open_container(None, LayoutStyle {
            direction: Direction::Horizontal,
            gap: self.spacing,
            ..Default::default()
        });
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
        self.tree_build.close_container();
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
        self.tree_build.open_container(None, LayoutStyle {
            max_width: Some(max_w),
            direction: Direction::Vertical,
            gap: self.spacing,
            ..Default::default()
        });
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
        self.tree_build.close_container();
    }

    /// Run a closure with padding on all sides.
    pub fn padding(&mut self, amount: f32, f: impl FnOnce(&mut Self)) {
        self.tree_build.open_container(None, LayoutStyle {
            direction: Direction::Vertical,
            padding: Spacing::all(amount),
            gap: self.spacing,
            ..Default::default()
        });
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
        self.tree_build.close_container();
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

    /// Responsive width classification based on the current region width.
    pub fn width_class(&self) -> WidthClass {
        if self.region.w < self.theme.breakpoint_compact {
            WidthClass::Compact
        } else if self.region.w >= self.theme.breakpoint_expanded {
            WidthClass::Expanded
        } else {
            WidthClass::Medium
        }
    }

    /// Narrow the region: offset cursor.x and reduce region.w.
    /// Useful for centering content without a closure.
    pub fn indent(&mut self, offset: f32, width: f32) {
        self.cursor.x += offset;
        self.region = Rect::new(self.cursor.x, self.region.y, width, self.region.h);
    }

    /// Run a closure with a temporary spacing value. Restores original spacing after.
    pub fn with_spacing(&mut self, gap: f32, f: impl FnOnce(&mut Self)) {
        self.tree_build.open_container(None, LayoutStyle {
            direction: Direction::Vertical,
            gap,
            ..Default::default()
        });
        let saved = self.spacing;
        self.spacing = gap;
        f(self);
        self.spacing = saved;
        self.tree_build.close_container();
    }

    /// Run a closure in a horizontal row with a specific inter-widget gap.
    pub fn row_spaced(&mut self, gap: f32, f: impl FnOnce(&mut Self)) {
        let saved = self.spacing;
        self.spacing = gap;
        self.row(|ui| {
            f(ui);
        });
        self.spacing = saved;
    }

    // ── Flex Layout ──

    /// Create a horizontal flex layout builder.
    pub fn flex_row(&mut self) -> FlexBuilder<'_, 'f> {
        FlexBuilder {
            ui: self,
            direction: Direction::Horizontal,
            gap: 0.0,
            align: layout::Align::Start,
            justify: layout::Justify::Start,
            wrap: layout::FlexWrap::NoWrap,
        }
    }

    /// Create a vertical flex layout builder.
    pub fn flex_col(&mut self) -> FlexBuilder<'_, 'f> {
        FlexBuilder {
            ui: self,
            direction: Direction::Vertical,
            gap: 0.0,
            align: layout::Align::Start,
            justify: layout::Justify::Start,
            wrap: layout::FlexWrap::NoWrap,
        }
    }

    /// Center content horizontally within the current region.
    ///
    /// `content_width` is the expected width of the content inside.
    pub fn center_horizontal(&mut self, content_width: f32, f: impl FnOnce(&mut Self)) {
        self.tree_build.open_container(None, LayoutStyle {
            max_width: Some(content_width),
            direction: Direction::Vertical,
            gap: self.spacing,
            ..Default::default()
        });
        let cw = content_width.min(self.region.w);
        let offset = (self.region.w - cw) / 2.0;

        let saved_cursor = self.cursor;
        let saved_region = self.region;

        self.cursor.x += offset;
        self.region = Rect::new(self.cursor.x, self.cursor.y, cw, self.region.h);

        f(self);

        let new_y = self.cursor.y;
        self.cursor = saved_cursor;
        self.cursor.y = new_y;
        self.region = saved_region;
        self.tree_build.close_container();
    }

    /// In a row, advance cursor.x to right-align the remaining content.
    ///
    /// `reserve_right` is the width to reserve for trailing widgets.
    pub fn fill_space(&mut self, reserve_right: f32) {
        let target_x = self.region.x + self.region.w - reserve_right;
        if target_x > self.cursor.x {
            self.cursor.x = target_x;
        }
    }

    /// Measure the size a closure would occupy without actually drawing.
    ///
    /// Runs `f` to populate GPU instances, records cursor delta, then truncates
    /// the instance buffer to discard all generated instances. Use only for
    /// small subtrees — the closure is fully executed (double-render cost).
    pub fn measure(&mut self, f: impl FnOnce(&mut Self)) -> (f32, f32) {
        let saved_cursor = self.cursor;
        let start_len = self.frame.instance_len();

        f(self);

        let dx = self.cursor.x - saved_cursor.x;
        let dy = self.cursor.y - saved_cursor.y;

        // Discard all GPU instances generated by f.
        self.frame.truncate_instances(start_len);
        self.cursor = saved_cursor;

        (dx, dy)
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

        self.tree_build.open_container(None, LayoutStyle {
            direction: Direction::Horizontal,
            gap,
            ..Default::default()
        });

        let saved_cursor = self.cursor;
        let saved_region = self.region;
        let saved_spacing = self.spacing;

        let mut col_x = self.cursor.x;
        let mut max_height: f32 = 0.0;

        for (i, &w) in weights.iter().enumerate() {
            let col_w = available * w / total_weight;

            self.tree_build.open_container(None, LayoutStyle {
                direction: Direction::Vertical,
                gap: self.spacing,
                flex_grow: w / total_weight,
                ..Default::default()
            });

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
            self.tree_build.close_container();
        }

        self.cursor = saved_cursor;
        self.cursor.y += max_height;
        self.region = saved_region;
        self.spacing = saved_spacing;
        self.tree_build.close_container();
    }

    // ── Constrained layout ──

    /// Run a closure within layout constraints.
    pub fn constrained(&mut self, c: layout::Constraints, f: impl FnOnce(&mut Self)) {
        self.tree_build.open_container(None, LayoutStyle {
            min_width: c.min_width,
            max_width: c.max_width,
            min_height: c.min_height,
            max_height: c.max_height,
            direction: Direction::Vertical,
            gap: self.spacing,
            ..Default::default()
        });
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
        self.tree_build.close_container();
    }

    // ── Tree indent ──

    /// Indent children of an expanded tree node.
    pub fn tree_indent(&mut self, f: impl FnOnce(&mut Self)) {
        let indent = self.theme.tree_indent;
        self.tree_build.open_container(None, LayoutStyle {
            direction: Direction::Vertical,
            padding: Spacing { left: indent, ..Default::default() },
            gap: self.spacing,
            ..Default::default()
        });
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
        self.tree_build.close_container();
    }

    /// Animated tree indent — draws children with a clip-rect height animation.
    ///
    /// `anim_id` should be a unique ID for this tree node (used for the animation).
    /// `expanded` is the current expand state. Children are always drawn (for
    /// measurement), but clipped during the animation.
    pub fn animated_tree_indent(
        &mut self,
        anim_id: u64,
        expanded: bool,
        f: impl FnOnce(&mut Self),
    ) {
        let indent = self.theme.tree_indent;
        let duration = self.theme.tree_expand_duration_ms;
        let target = if expanded { 1.0 } else { 0.0 };
        let t = self.state.anim_t(anim_id, target, duration, state::Easing::EaseOutCubic);

        // If fully collapsed and animation settled, skip drawing entirely.
        if t < 0.001 && !expanded {
            return;
        }

        self.tree_build.open_container(None, LayoutStyle {
            direction: Direction::Vertical,
            padding: Spacing { left: indent, ..Default::default() },
            gap: self.spacing,
            ..Default::default()
        });
        let saved_cursor_x = self.cursor.x;
        let saved_cursor_y = self.cursor.y;
        let saved_region = self.region;
        let saved_clip = self.frame.active_clip();

        self.cursor.x += indent;
        self.region = Rect::new(
            self.cursor.x,
            self.region.y,
            self.region.w - indent,
            self.region.h,
        );

        let start_y = self.cursor.y;

        // Set clip rect if animating (0 < t < 1).
        let cached_h = self.state.tree_children_heights.get(&anim_id).copied().unwrap_or(0.0);
        if t < 0.999 {
            let clip_h = cached_h * t;
            self.frame.set_active_clip(Some([
                saved_cursor_x,
                start_y,
                saved_region.w,
                clip_h,
            ]));
        }

        f(self);

        let children_h = self.cursor.y - start_y;
        self.state.tree_children_heights.insert(anim_id, children_h);

        // Restore clip.
        self.frame.set_active_clip(saved_clip);

        // Advance cursor by animated height.
        let visible_h = if t >= 0.999 {
            children_h
        } else {
            cached_h * t
        };

        self.cursor.x = saved_cursor_x;
        self.cursor.y = saved_cursor_y + visible_h;
        self.region = saved_region;
        self.tree_build.close_container();
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
    pub fn register_widget(
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

        // Track the focused widget's kind for keyboard activation.
        if self.state.focused == Some(id) {
            self.state.focused_kind = Some(kind);
        }

        // Track rects for widgets with active animations (for targeted damage).
        if self.state.hover_anims.contains_key(&id) || self.state.anims.contains_key(&id) {
            self.state.anim_rects.insert(id, rect);
        }
    }

    /// Compute the Response for a widget given its ID and rect.
    /// When disabled, returns an inert Response with `disabled: true`.
    pub fn widget_response(&mut self, id: u64, rect: Rect) -> response::Response {
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

        // Keyboard activation: Enter/Space triggers click for activatable widgets.
        if !clicked && focused && !self.disabled {
            use esox_input::{Key, NamedKey};
            let activatable = matches!(
                self.state.focused_kind,
                Some(state::WidgetKind::Button)
                    | Some(state::WidgetKind::Checkbox)
                    | Some(state::WidgetKind::Toggle)
                    | Some(state::WidgetKind::Radio)
                    | Some(state::WidgetKind::Hyperlink)
                    | Some(state::WidgetKind::Tab)
            );
            if activatable {
                for (event, _) in &self.state.keys {
                    if event.pressed {
                        if matches!(
                            event.key,
                            Key::Named(NamedKey::Enter) | Key::Named(NamedKey::Space)
                        ) {
                            clicked = true;
                            self.state.focused = Some(id);
                            self.state.reset_blink();
                            break;
                        }
                    }
                }
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

    /// Borrow rendering resources for custom widget drawing.
    pub fn painter(&mut self) -> Painter<'_> {
        Painter {
            frame: self.frame,
            text: self.text,
            gpu: self.gpu,
            resources: self.resources,
            theme: self.theme,
        }
    }

    /// Get a hover animation value for a custom widget.
    ///
    /// Convenience wrapper around [`UiState::hover_t`]. Pass a unique `id`
    /// (typically `widget_id ^ HOVER_SALT`), the current `hovered` state,
    /// and the animation duration in milliseconds. Returns 0.0–1.0.
    pub fn hover_t(&mut self, id: u64, hovered: bool, duration_ms: f32) -> f32 {
        self.state.hover_t(id, hovered, duration_ms)
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

    // ── Widget Style Overrides ──

    /// Run a closure with a widget style override pushed onto the stack.
    pub fn with_style(&mut self, style: WidgetStyle, f: impl FnOnce(&mut Self)) {
        self.style_stack.push(style);
        f(self);
        self.style_stack.pop();
    }

    /// Resolve foreground color: style stack override or theme default.
    pub(crate) fn resolve_fg(&self) -> Color {
        for s in self.style_stack.iter().rev() {
            if let Some(c) = s.fg {
                return c;
            }
        }
        // Fallback: check inherited text_color for label-type widgets.
        for s in self.style_stack.iter().rev() {
            if let Some(c) = s.text_color {
                return c;
            }
        }
        self.theme.fg
    }

    /// Resolve background color for interactive widgets.
    pub(crate) fn resolve_bg(&self) -> Color {
        for s in self.style_stack.iter().rev() {
            if let Some(c) = s.bg {
                return c;
            }
        }
        self.theme.accent
    }

    /// Resolve font size: style stack override or theme default.
    pub(crate) fn resolve_font_size(&self) -> f32 {
        for s in self.style_stack.iter().rev() {
            if let Some(v) = s.font_size {
                return v;
            }
        }
        self.theme.font_size
    }

    /// Resolve corner radius: style stack override or theme default.
    pub(crate) fn resolve_corner_radius(&self) -> f32 {
        for s in self.style_stack.iter().rev() {
            if let Some(v) = s.corner_radius {
                return v;
            }
        }
        self.theme.corner_radius
    }

    /// Resolve button height: style stack override or theme default.
    pub(crate) fn resolve_height(&self) -> f32 {
        for s in self.style_stack.iter().rev() {
            if let Some(v) = s.height {
                return v;
            }
        }
        self.theme.button_height
    }

    /// Resolve border color: style stack override or theme default.
    #[allow(dead_code)]
    pub(crate) fn resolve_border_color(&self) -> Color {
        for s in self.style_stack.iter().rev() {
            if let Some(c) = s.border_color {
                return c;
            }
        }
        self.theme.border
    }

    /// Access the theme.
    /// Whether the debug overlay is currently enabled.
    pub fn is_debug_overlay(&self) -> bool {
        self.state.debug_overlay
    }

    pub fn theme(&self) -> &Theme {
        self.theme
    }

    // ── Convenience Combinators ──

    /// Scrollable + padding + max_width in one call (common page layout).
    pub fn page(&mut self, id: u64, scroll_h: f32, max_w: f32, f: impl FnOnce(&mut Self)) {
        self.scrollable(id, scroll_h, |ui| {
            ui.max_width(max_w, |ui| {
                ui.padding(ui.theme.padding, f);
            });
        });
    }

    /// Label-widget pair in a row: "Label    [widget]"
    pub fn labeled(&mut self, label: &str, f: impl FnOnce(&mut Self)) {
        self.row(|ui| {
            let label_w = ui.text.measure_text(label, ui.theme.font_size);
            let rect = ui.allocate_rect(label_w, ui.theme.button_height);
            ui.text.draw_ui_text(
                label,
                rect.x,
                rect.y + (rect.h - ui.theme.font_size) / 2.0,
                ui.theme.fg_label,
                ui.frame,
                ui.gpu,
                ui.resources,
            );
            f(ui);
        });
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

    /// Draw debug overlay outlines around all allocated rects. Called from `finish()`.
    fn draw_debug_overlay(&mut self) {
        if !self.state.debug_overlay {
            return;
        }

        // Depth-cycling colors for layout bounds.
        const DEPTH_COLORS: [[f32; 4]; 4] = [
            [0.2, 0.8, 0.2, 0.3],  // green
            [0.2, 0.5, 1.0, 0.3],  // blue
            [1.0, 0.6, 0.1, 0.3],  // orange
            [0.8, 0.2, 0.8, 0.3],  // purple
        ];

        let rects: Vec<_> = self.state.debug_widget_rects.drain(..).collect();
        for (i, (rect, _id, _kind)) in rects.iter().enumerate() {
            let color_idx = i % DEPTH_COLORS.len();
            let [r, g, b, a] = DEPTH_COLORS[color_idx];
            let outline_color = Color::new(r, g, b, a);
            // Draw 1px outline.
            let t = 1.0;
            // Top
            paint::draw_rounded_rect(self.frame, Rect::new(rect.x, rect.y, rect.w, t), outline_color, 0.0);
            // Bottom
            paint::draw_rounded_rect(self.frame, Rect::new(rect.x, rect.y + rect.h - t, rect.w, t), outline_color, 0.0);
            // Left
            paint::draw_rounded_rect(self.frame, Rect::new(rect.x, rect.y, t, rect.h), outline_color, 0.0);
            // Right
            paint::draw_rounded_rect(self.frame, Rect::new(rect.x + rect.w - t, rect.y, t, rect.h), outline_color, 0.0);
        }

        // Hovered widget gets a tooltip with rect info.
        let mouse_x = self.state.mouse.x;
        let mouse_y = self.state.mouse.y;
        if let Some((rect, id, kind)) = rects.iter().rev()
            .find(|(r, _, _)| r.contains(mouse_x, mouse_y))
        {
            // Alt+click: log widget info instead of sending click.
            if self.state.modifiers.alt() {
                if let Some((_, _, ref mut consumed)) = self.state.mouse.pending_click {
                    if !*consumed {
                        *consumed = true;
                        tracing::info!(
                            id = id,
                            kind = kind,
                            x = rect.x,
                            y = rect.y,
                            w = rect.w,
                            h = rect.h,
                            "debug: click-to-inspect widget"
                        );
                    }
                }
            }

            // Draw highlighted outline on hovered widget.
            let highlight = Color::new(1.0, 1.0, 0.0, 0.5);
            let t = 2.0;
            paint::draw_rounded_rect(self.frame, Rect::new(rect.x, rect.y, rect.w, t), highlight, 0.0);
            paint::draw_rounded_rect(self.frame, Rect::new(rect.x, rect.y + rect.h - t, rect.w, t), highlight, 0.0);
            paint::draw_rounded_rect(self.frame, Rect::new(rect.x, rect.y, t, rect.h), highlight, 0.0);
            paint::draw_rounded_rect(self.frame, Rect::new(rect.x + rect.w - t, rect.y, t, rect.h), highlight, 0.0);

            // Info tooltip.
            let info = format!("{kind} [{:.0}x{:.0}] @({:.0},{:.0})", rect.w, rect.h, rect.x, rect.y);
            let font_size = self.theme.tooltip_font_size;
            let pad = 4.0;
            let text_w = self.text.measure_text(&info, font_size);
            let tw = text_w + pad * 2.0;
            let th = font_size + pad * 2.0;
            let tx = (mouse_x + 12.0).min(self.region.x + self.region.w - tw);
            let ty = (mouse_y + 12.0).min(self.region.y + self.region.h - th);

            paint::draw_rounded_rect(
                self.frame,
                Rect::new(tx, ty, tw, th),
                Color::new(0.0, 0.0, 0.0, 0.85),
                3.0,
            );
            self.text.draw_text(
                &info,
                tx + pad,
                ty + pad,
                font_size,
                Color::new(1.0, 1.0, 0.0, 1.0),
                self.frame,
                self.gpu,
                self.resources,
            );
        }
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
    pub fn push_a11y_node(&mut self, node: state::A11yNode) {
        if self.state.a11y_enabled {
            self.state.a11y_tree.push(node);
        }
    }

    // ── Animation API ──

    /// Drive a custom animation. Returns the current interpolated value.
    ///
    /// `id` identifies the animation (use `id!()` or `fnv1a_mix`).
    /// `target` is the value to animate towards.
    /// On first call, starts settled at `target`. When `target` changes,
    /// restarts from the current value (smooth retargeting).
    pub fn animate(&mut self, id: u64, target: f32, duration_ms: f32, easing: Easing) -> f32 {
        self.state.anim_t(id, target, duration_ms, easing)
    }

    /// Boolean animation helper. Returns 0.0→1.0 interpolation.
    ///
    /// Equivalent to `animate(id, if active { 1.0 } else { 0.0 }, ...)`.
    pub fn animate_bool(&mut self, id: u64, active: bool, duration_ms: f32, easing: Easing) -> f32 {
        let target = if active { 1.0 } else { 0.0 };
        self.state.anim_t(id, target, duration_ms, easing)
    }

    /// Whether the given animation is currently in-flight (not settled).
    pub fn is_animating(&self, id: u64) -> bool {
        self.state.anim_active(id)
    }

    // ── Focus control ──

    /// Set focus to the given widget ID.
    pub fn request_focus(&mut self, id: u64) {
        self.state.focused = Some(id);
        self.state.reset_blink();
    }

    /// Remove focus from any widget.
    pub fn clear_focus(&mut self) {
        self.state.focused = None;
    }

    /// Check if the given widget ID currently has focus.
    pub fn has_focus(&self, id: u64) -> bool {
        self.state.focused == Some(id)
    }

    /// Move focus to the next widget in the focus chain.
    pub fn focus_next(&mut self) {
        self.state.focus_next();
    }

    /// Move focus to the previous widget in the focus chain.
    pub fn focus_prev(&mut self) {
        self.state.focus_prev();
    }

    // ── Damage tracking ──

    /// Whether the UI has damage that requires a redraw.
    ///
    /// Returns `true` if any widget state changed (hover, focus, animation,
    /// scroll, overlay) since the last frame. When this returns `false`,
    /// the platform layer can skip GPU submission to save power.
    pub fn needs_redraw(&self) -> bool {
        self.state.damage.is_full_invalidation()
            || self.state.damage.regions().map_or(false, |r| !r.is_empty())
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
