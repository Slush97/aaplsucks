//! Table widget — columnar data with headers, virtual scrolling, row selection,
//! resizable columns, sorting, and multi-select.
//!
//! # Examples
//!
//! ```ignore
//! let columns = &[
//!     TableColumn::new("Name", ColumnWidth::Weight(2.0)),
//!     TableColumn::new("Size", ColumnWidth::Fixed(80.0)),
//! ];
//! ui.table(id!("files"), &mut table_state, columns, &mut vs, 32.0, 400.0,
//!     |ui, row, col| {
//!         ui.label(&data[row][col]);
//!     },
//! );
//! ```

use esox_gfx::ShapeBuilder;
use esox_input::{Key, NamedKey};

use crate::id::fnv1a_mix;
use crate::layout::{Rect, Vec2};
use crate::paint;
use crate::response::Response;
use crate::state::{SortDirection, TableState, VirtualScrollState, WidgetKind};
use crate::Ui;

/// Column width specification for tables.
#[derive(Debug, Clone, Copy)]
pub enum ColumnWidth {
    Fixed(f32),
    Weight(f32),
    Auto,
}

/// A table column descriptor.
pub struct TableColumn<'a> {
    pub header: &'a str,
    pub width: ColumnWidth,
    pub sortable: bool,
}

impl<'a> TableColumn<'a> {
    pub fn new(header: &'a str, width: ColumnWidth) -> Self {
        Self {
            header,
            width,
            sortable: true,
        }
    }

    pub fn not_sortable(mut self) -> Self {
        self.sortable = false;
        self
    }
}

impl<'f> Ui<'f> {
    /// Draw a table with headers, virtual-scrolled body, and row selection.
    pub fn table(
        &mut self,
        id: u64,
        state: &mut TableState,
        columns: &[TableColumn<'_>],
        row_count: usize,
        visible_rows: usize,
        mut draw_cell: impl FnMut(&mut Self, usize, usize),
    ) -> Response {
        let font_size = self.theme.font_size;
        let pad = self.theme.input_padding;
        let header_h = self.theme.table_header_height;
        let item_h = self.theme.item_height;
        let scrollbar_w = self.theme.scrollbar_width;
        let resize_handle_w = self.theme.column_resize_handle_width;
        let resize_min_w = self.theme.column_resize_min_width;

        // Ensure column_widths vec is sized.
        if state.column_widths.len() != columns.len() {
            state.column_widths.resize(columns.len(), None);
        }

        // Resolve column widths.
        let total_w = self.region.w - scrollbar_w;
        let col_widths = resolve_column_widths(columns, total_w, &mut self.text, font_size, pad, &state.column_widths);

        // Full table rect for focus ring.
        let visible_height = visible_rows as f32 * item_h;
        let table_rect = Rect::new(self.cursor.x, self.cursor.y, self.region.w, header_h + visible_height);

        // Register table for focus/keyboard.
        self.register_widget(id, table_rect, WidgetKind::Button);
        let table_focused = self.state.focused == Some(id);

        // Focus ring.
        if table_focused {
            paint::draw_focus_ring(
                self.frame,
                table_rect,
                self.theme.accent_dim,
                self.theme.corner_radius,
                self.theme.focus_ring_expand,
            );
        }

        // Header row.
        let header_rect = self.allocate_rect_keyed(id, self.region.w, header_h);
        self.frame.push(
            ShapeBuilder::rect(header_rect.x, header_rect.y, header_rect.w, header_h)
                .color(self.theme.bg_surface)
                .build(),
        );

        // Handle column resize drag.
        if let Some((col, start_x, start_w)) = state.resize_drag {
            if self.state.mouse_pressed {
                let new_w = (start_w + (self.state.mouse.x - start_x)).max(resize_min_w);
                state.column_widths[col] = Some(new_w);
            } else {
                state.resize_drag = None;
            }
        }

        // Draw header labels with sort indicators and resize handles.
        let mut hx = header_rect.x;
        let mut sort_changed = false;

        for (i, col) in columns.iter().enumerate() {
            let col_w = col_widths[i];

            // Header text.
            self.text.draw_text(
                col.header,
                hx + pad,
                header_rect.y + (header_h - font_size) / 2.0,
                font_size,
                self.theme.fg_muted,
                self.frame,
                self.gpu,
                self.resources,
            );

            // Sort indicator.
            if let Some((sort_col, dir)) = &state.sort {
                if *sort_col == i {
                    let indicator = match dir {
                        SortDirection::Ascending => "\u{25B2}",  // ▲
                        SortDirection::Descending => "\u{25BC}", // ▼
                    };
                    let header_text_w = self.text.measure_text(col.header, font_size);
                    self.text.draw_text(
                        indicator,
                        hx + pad + header_text_w + 4.0,
                        header_rect.y + (header_h - font_size) / 2.0,
                        font_size * 0.7,
                        self.theme.fg_muted,
                        self.frame,
                        self.gpu,
                        self.resources,
                    );
                }
            }

            // Resize handle hit zone (invisible, at column border).
            if i < columns.len() - 1 {
                let handle_x = hx + col_w - resize_handle_w / 2.0;
                let handle_rect = Rect::new(handle_x, header_rect.y, resize_handle_w, header_h);
                let handle_id = fnv1a_mix(fnv1a_mix(id, 0xBE51_2E00), i as u64);

                self.state.hit_rects.push((handle_rect, handle_id, WidgetKind::ColumnResize));

                // Visual: 1px line at column border.
                let border_x = hx + col_w - 0.5;
                let handle_hovered = handle_rect.contains(self.state.mouse.x, self.state.mouse.y);
                let is_dragging = state.resize_drag.map_or(false, |(c, _, _)| c == i);
                if handle_hovered || is_dragging {
                    self.frame.push(
                        ShapeBuilder::rect(border_x, header_rect.y, 1.0, header_h)
                            .color(self.theme.accent)
                            .build(),
                    );
                } else {
                    self.frame.push(
                        ShapeBuilder::rect(border_x, header_rect.y, 1.0, header_h)
                            .color(self.theme.border)
                            .build(),
                    );
                }

                // Initiate resize drag.
                if state.resize_drag.is_none() {
                    if let Some((cx, cy, ref mut consumed)) = self.state.mouse.pending_click {
                        if !*consumed && handle_rect.contains(cx, cy) {
                            state.resize_drag = Some((i, cx, col_w));
                            *consumed = true;
                        }
                    }
                }
            }

            // Header click for sorting (only if not resize handle click).
            if col.sortable && state.resize_drag.is_none() {
                let header_click_rect = Rect::new(hx, header_rect.y, col_w - resize_handle_w, header_h);
                if let Some((cx, cy, ref mut consumed)) = self.state.mouse.pending_click {
                    if !*consumed && header_click_rect.contains(cx, cy) {
                        // Three-state cycle: None -> Ascending -> Descending -> None.
                        state.sort = match state.sort {
                            Some((sort_col, SortDirection::Ascending)) if sort_col == i => {
                                Some((i, SortDirection::Descending))
                            }
                            Some((sort_col, SortDirection::Descending)) if sort_col == i => None,
                            _ => Some((i, SortDirection::Ascending)),
                        };
                        sort_changed = true;
                        *consumed = true;
                    }
                }
            }

            hx += col_w;
        }

        // Header bottom border.
        self.frame.push(
            ShapeBuilder::rect(header_rect.x, header_rect.y + header_h - 1.0, header_rect.w, 1.0)
                .color(self.theme.border)
                .build(),
        );

        // Body via virtual scroll.
        let mut vs_state = VirtualScrollState::new(row_count);

        let mut response = Response::default();
        response.changed = sort_changed;

        // Keyboard navigation.
        let modifiers = self.state.modifiers;
        if table_focused {
            let keys: Vec<_> = self.state.keys.clone();
            for (event, _mods) in &keys {
                if !event.pressed {
                    continue;
                }
                let shift = modifiers.shift();
                match &event.key {
                    Key::Named(NamedKey::ArrowUp) => {
                        if let Some(sel) = state.selected_row {
                            if sel > 0 {
                                let new = sel - 1;
                                if shift {
                                    extend_selection(state, new);
                                } else {
                                    single_select(state, new);
                                }
                                vs_state.scroll_to = Some(new);
                                response.changed = true;
                            }
                        } else if row_count > 0 {
                            single_select(state, 0);
                            vs_state.scroll_to = Some(0);
                            response.changed = true;
                        }
                    }
                    Key::Named(NamedKey::ArrowDown) => {
                        if let Some(sel) = state.selected_row {
                            if sel + 1 < row_count {
                                let new = sel + 1;
                                if shift {
                                    extend_selection(state, new);
                                } else {
                                    single_select(state, new);
                                }
                                vs_state.scroll_to = Some(new);
                                response.changed = true;
                            }
                        } else if row_count > 0 {
                            single_select(state, 0);
                            vs_state.scroll_to = Some(0);
                            response.changed = true;
                        }
                    }
                    Key::Named(NamedKey::PageUp) => {
                        if let Some(sel) = state.selected_row {
                            let new = sel.saturating_sub(visible_rows);
                            single_select(state, new);
                            vs_state.scroll_to = Some(new);
                            response.changed = true;
                        }
                    }
                    Key::Named(NamedKey::PageDown) => {
                        if let Some(sel) = state.selected_row {
                            let new = (sel + visible_rows).min(row_count.saturating_sub(1));
                            single_select(state, new);
                            vs_state.scroll_to = Some(new);
                            response.changed = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        // Clone col_widths for use inside closure.
        let col_widths_clone = col_widths.clone();
        let accent_dim = self.theme.accent_dim;
        let bg_base = self.theme.bg_base;
        let zebra_bg = self.theme.table_zebra_bg;
        let ctrl = modifiers.ctrl();
        let shift = modifiers.shift();

        self.virtual_scroll(id, &mut vs_state, item_h, visible_height, |ui, row| {
            let row_rect = Rect::new(ui.cursor.x, ui.cursor.y, total_w, item_h);
            let row_id = fnv1a_mix(id, row as u64 + 1);

            ui.register_widget(row_id, row_rect, WidgetKind::TableRow);
            let row_resp = ui.widget_response(row_id, row_rect);

            // Row background.
            let is_selected = state.selected_rows.contains(&row)
                || (state.selected_rows.is_empty() && state.selected_row == Some(row));
            let bg = if is_selected {
                accent_dim
            } else if row % 2 == 1 {
                zebra_bg
            } else {
                bg_base
            };

            ui.frame.push(
                ShapeBuilder::rect(row_rect.x, row_rect.y, row_rect.w, row_rect.h)
                    .color(bg)
                    .build(),
            );

            // Hover highlight.
            if row_resp.hovered && !is_selected {
                ui.frame.push(
                    ShapeBuilder::rect(row_rect.x, row_rect.y, row_rect.w, row_rect.h)
                        .color(esox_gfx::Color::new(ui.theme.fg.r, ui.theme.fg.g, ui.theme.fg.b, 0.03))
                        .build(),
                );
            }

            // Click to select (with modifier support).
            if row_resp.clicked {
                if ctrl && shift {
                    // Ctrl+Shift+click: add range to existing selection.
                    if let Some(anchor) = state.anchor_row {
                        let (from, to) = if anchor <= row { (anchor, row) } else { (row, anchor) };
                        for r in from..=to {
                            state.selected_rows.insert(r);
                        }
                    }
                } else if ctrl {
                    // Ctrl+click: toggle in set.
                    if state.selected_rows.contains(&row) {
                        state.selected_rows.remove(&row);
                    } else {
                        state.selected_rows.insert(row);
                    }
                    state.anchor_row = Some(row);
                } else if shift {
                    // Shift+click: range from anchor.
                    state.selected_rows.clear();
                    if let Some(anchor) = state.anchor_row {
                        let (from, to) = if anchor <= row { (anchor, row) } else { (row, anchor) };
                        for r in from..=to {
                            state.selected_rows.insert(r);
                        }
                    } else {
                        state.selected_rows.insert(row);
                        state.anchor_row = Some(row);
                    }
                } else {
                    // Plain click: clear, select one.
                    single_select(state, row);
                }
                state.selected_row = Some(row);
            }

            // Draw cells.
            let saved_cursor = ui.cursor;
            let saved_region = ui.region;
            let mut cx = row_rect.x;
            for col in 0..col_widths_clone.len() {
                ui.cursor = Vec2 { x: cx + pad, y: row_rect.y + (item_h - ui.theme.font_size) / 2.0 };
                ui.region = Rect::new(cx, row_rect.y, col_widths_clone[col], item_h);
                draw_cell(ui, row, col);
                cx += col_widths_clone[col];
            }
            ui.cursor = saved_cursor;
            ui.region = saved_region;
        });

        response.focused = table_focused;
        response
    }
}

fn single_select(state: &mut TableState, row: usize) {
    state.selected_rows.clear();
    state.selected_rows.insert(row);
    state.selected_row = Some(row);
    state.anchor_row = Some(row);
}

fn extend_selection(state: &mut TableState, row: usize) {
    if let Some(anchor) = state.anchor_row {
        state.selected_rows.clear();
        let (from, to) = if anchor <= row { (anchor, row) } else { (row, anchor) };
        for r in from..=to {
            state.selected_rows.insert(r);
        }
    } else {
        single_select(state, row);
    }
    state.selected_row = Some(row);
}

fn resolve_column_widths(
    columns: &[TableColumn<'_>],
    total_w: f32,
    text: &mut crate::text::TextRenderer,
    font_size: f32,
    pad: f32,
    overrides: &[Option<f32>],
) -> Vec<f32> {
    let mut widths = vec![0.0f32; columns.len()];
    let mut fixed_total = 0.0f32;
    let mut weight_total = 0.0f32;

    for (i, col) in columns.iter().enumerate() {
        // Check for user override first.
        if let Some(Some(override_w)) = overrides.get(i) {
            widths[i] = *override_w;
            fixed_total += *override_w;
            continue;
        }

        match col.width {
            ColumnWidth::Fixed(w) => {
                widths[i] = w;
                fixed_total += w;
            }
            ColumnWidth::Auto => {
                let w = text.measure_text(col.header, font_size) + pad * 2.0;
                widths[i] = w;
                fixed_total += w;
            }
            ColumnWidth::Weight(w) => {
                weight_total += w;
            }
        }
    }

    let remaining = (total_w - fixed_total).max(0.0);
    if weight_total > 0.0 {
        for (i, col) in columns.iter().enumerate() {
            if overrides.get(i).and_then(|o| o.as_ref()).is_some() {
                continue;
            }
            if let ColumnWidth::Weight(w) = col.width {
                widths[i] = remaining * w / weight_total;
            }
        }
    }

    widths
}
