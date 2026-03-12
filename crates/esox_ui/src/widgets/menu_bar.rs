//! Menu bar widget — horizontal strip with dropdown menus and keyboard accelerator display.
//!
//! # Examples
//!
//! ```ignore
//! let menus = &[
//!     Menu::new("File", vec![
//!         MenuEntry::Item(MenuItem::new("New", 1).with_shortcut("Ctrl+N")),
//!         MenuEntry::Item(MenuItem::new("Open", 2).with_shortcut("Ctrl+O")),
//!         MenuEntry::Separator,
//!         MenuEntry::Item(MenuItem::new("Save", 3).with_shortcut("Ctrl+S")),
//!         MenuEntry::Separator,
//!         MenuEntry::Item(MenuItem::new("Quit", 4).with_shortcut("Ctrl+Q")),
//!     ]),
//!     Menu::new("Edit", vec![
//!         MenuEntry::Item(MenuItem::new("Undo", 10).with_shortcut("Ctrl+Z")),
//!         MenuEntry::Item(MenuItem::new("Redo", 11).with_shortcut("Ctrl+Shift+Z")),
//!         MenuEntry::Separator,
//!         MenuEntry::Item(MenuItem::new("Cut", 12).with_shortcut("Ctrl+X")),
//!         MenuEntry::Item(MenuItem::new("Copy", 13).with_shortcut("Ctrl+C")),
//!         MenuEntry::Item(MenuItem::new("Paste", 14).with_shortcut("Ctrl+V")),
//!     ]),
//! ];
//!
//! if let Some(action) = ui.menu_bar(menus) {
//!     match action {
//!         1 => new_document(),
//!         3 => save_document(),
//!         _ => {}
//!     }
//! }
//! ```

use esox_gfx::{BorderRadius, ShapeBuilder};
use winit::keyboard::{Key, NamedKey};

use crate::layout::Rect;
use crate::paint;
use crate::Ui;

/// A single actionable menu item.
pub struct MenuItem {
    pub label: String,
    pub shortcut: Option<String>,
    pub enabled: bool,
    pub id: u64,
}

impl MenuItem {
    pub fn new(label: impl Into<String>, id: u64) -> Self {
        Self {
            label: label.into(),
            shortcut: None,
            enabled: true,
            id,
        }
    }

    pub fn with_shortcut(mut self, shortcut: impl Into<String>) -> Self {
        self.shortcut = Some(shortcut.into());
        self
    }

    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

/// A top-level menu with a label and a list of entries.
pub struct Menu {
    pub label: String,
    pub items: Vec<MenuEntry>,
}

impl Menu {
    pub fn new(label: impl Into<String>, items: Vec<MenuEntry>) -> Self {
        Self {
            label: label.into(),
            items,
        }
    }
}

/// An entry in a menu — either an item or a separator line.
pub enum MenuEntry {
    Item(MenuItem),
    Separator,
}

impl<'f> Ui<'f> {
    /// Draw a menu bar and return the id of the clicked menu item, if any.
    ///
    /// The bar is a horizontal strip at the current cursor position. Each menu
    /// label is a clickable region; clicking opens a dropdown below it. Hovering
    /// between labels while a menu is open switches which menu is shown.
    pub fn menu_bar(&mut self, menus: &[Menu]) -> Option<u64> {
        if menus.is_empty() {
            return None;
        }

        let font_size = self.theme.font_size;
        let pad = self.theme.input_padding;
        let bar_h = self.theme.item_height;

        // Allocate the bar strip.
        let bar_rect = self.allocate_rect(self.region.w, bar_h);

        // Draw bar background.
        self.frame.push(
            ShapeBuilder::rect(bar_rect.x, bar_rect.y, bar_rect.w, bar_rect.h)
                .color(self.theme.bg_surface)
                .build(),
        );

        // Bottom border.
        self.frame.push(
            ShapeBuilder::rect(bar_rect.x, bar_rect.y + bar_h - 1.0, bar_rect.w, 1.0)
                .color(self.theme.border)
                .build(),
        );

        // Measure and draw each menu label, building label rects.
        let mut label_rects: Vec<Rect> = Vec::with_capacity(menus.len());
        let mut lx = bar_rect.x;
        for menu in menus {
            let text_w = self.text.measure_text(&menu.label, font_size);
            let label_w = text_w + pad * 2.0;
            label_rects.push(Rect::new(lx, bar_rect.y, label_w, bar_h));
            lx += label_w;
        }

        let menu_bar_open = self.state.menu_bar_open;
        let mut result: Option<u64> = None;
        let mut new_open = menu_bar_open;

        // Handle Escape to close.
        let mut escape_pressed = false;
        if menu_bar_open.is_some() {
            for (event, _) in &self.state.keys {
                if event.state.is_pressed() {
                    if let Key::Named(NamedKey::Escape) = &event.logical_key {
                        escape_pressed = true;
                    }
                }
            }
        }
        if escape_pressed {
            new_open = None;
        }

        // Handle clicks on bar labels and hover-switching.
        if !escape_pressed {
            if let Some((cx, cy, ref mut consumed)) = self.state.mouse.pending_click {
                let mut clicked_label = false;
                for (i, lr) in label_rects.iter().enumerate() {
                    if lr.contains(cx, cy) {
                        // Toggle: clicking the already-open menu closes it.
                        if menu_bar_open == Some(i) {
                            new_open = None;
                        } else {
                            new_open = Some(i);
                        }
                        *consumed = true;
                        clicked_label = true;
                        break;
                    }
                }

                // Click outside bar and outside dropdown -> close.
                if !clicked_label && menu_bar_open.is_some() {
                    // Check if click is inside the open dropdown (handled below).
                    let in_dropdown = if let Some(open_idx) = menu_bar_open {
                        let dd_rect = self.dropdown_rect(&menus[open_idx], &label_rects[open_idx]);
                        dd_rect.contains(cx, cy)
                    } else {
                        false
                    };
                    if !in_dropdown {
                        new_open = None;
                        // Don't consume — let the click fall through.
                    }
                }
            }

            // Hover-switch: when a menu is open and mouse hovers another label.
            if menu_bar_open.is_some() {
                for (i, lr) in label_rects.iter().enumerate() {
                    if lr.contains(self.state.mouse.x, self.state.mouse.y) && menu_bar_open != Some(i) {
                        new_open = Some(i);
                        break;
                    }
                }
            }
        }

        // Draw menu labels.
        for (i, (menu, lr)) in menus.iter().zip(label_rects.iter()).enumerate() {
            let is_open = new_open == Some(i);
            let is_hovered = lr.contains(self.state.mouse.x, self.state.mouse.y);

            // Highlight background.
            if is_open || is_hovered {
                self.frame.push(
                    ShapeBuilder::rect(lr.x, lr.y, lr.w, lr.h)
                        .color(self.theme.bg_raised)
                        .build(),
                );
            }

            let text_color = if is_open {
                self.theme.accent
            } else {
                self.theme.fg
            };

            self.text.draw_text(
                &menu.label,
                lr.x + pad,
                lr.y + (bar_h - font_size) / 2.0,
                font_size,
                text_color,
                self.frame,
                self.gpu,
                self.resources,
            );
        }

        // Draw the open dropdown.
        if let Some(open_idx) = new_open {
            if open_idx < menus.len() {
                result = self.draw_menu_dropdown(
                    &menus[open_idx],
                    &label_rects[open_idx],
                );
                // If an item was selected, close the menu.
                if result.is_some() {
                    new_open = None;
                }
            }
        }

        self.state.menu_bar_open = new_open;
        result
    }

    /// Compute the dropdown rect for a menu (used for hit-testing before drawing).
    fn dropdown_rect(&mut self, menu: &Menu, anchor: &Rect) -> Rect {
        let font_size = self.theme.font_size;
        let pad = self.theme.input_padding;
        let item_h = self.theme.item_height;
        let sep_h: f32 = 9.0; // 1px line + 4px padding above/below

        // Measure dropdown width.
        let mut max_label_w: f32 = 0.0;
        let mut max_shortcut_w: f32 = 0.0;
        let mut total_h: f32 = 0.0;
        for entry in &menu.items {
            match entry {
                MenuEntry::Item(item) => {
                    let lw = self.text.measure_text(&item.label, font_size);
                    if lw > max_label_w {
                        max_label_w = lw;
                    }
                    if let Some(ref sc) = item.shortcut {
                        let sw = self.text.measure_text(sc, font_size);
                        if sw > max_shortcut_w {
                            max_shortcut_w = sw;
                        }
                    }
                    total_h += item_h;
                }
                MenuEntry::Separator => {
                    total_h += sep_h;
                }
            }
        }

        let shortcut_gap = if max_shortcut_w > 0.0 { 24.0 } else { 0.0 };
        let dd_w = (pad + max_label_w + shortcut_gap + max_shortcut_w + pad)
            .max(anchor.w)
            .max(self.theme.context_menu_min_w);
        let dd_x = anchor.x;
        let dd_y = anchor.y + anchor.h;

        Rect::new(dd_x, dd_y, dd_w, total_h)
    }

    /// Draw a single menu's dropdown and return the selected item id, if any.
    fn draw_menu_dropdown(&mut self, menu: &Menu, anchor: &Rect) -> Option<u64> {
        let font_size = self.theme.font_size;
        let pad = self.theme.input_padding;
        let item_h = self.theme.item_height;
        let sep_h: f32 = 9.0;
        let corner_r = self.theme.corner_radius;

        // Measure dropdown.
        let mut max_label_w: f32 = 0.0;
        let mut max_shortcut_w: f32 = 0.0;
        let mut total_h: f32 = 0.0;
        for entry in &menu.items {
            match entry {
                MenuEntry::Item(item) => {
                    let lw = self.text.measure_text(&item.label, font_size);
                    if lw > max_label_w {
                        max_label_w = lw;
                    }
                    if let Some(ref sc) = item.shortcut {
                        let sw = self.text.measure_text(sc, font_size);
                        if sw > max_shortcut_w {
                            max_shortcut_w = sw;
                        }
                    }
                    total_h += item_h;
                }
                MenuEntry::Separator => {
                    total_h += sep_h;
                }
            }
        }

        let shortcut_gap = if max_shortcut_w > 0.0 { 24.0 } else { 0.0 };
        let dd_w = (pad + max_label_w + shortcut_gap + max_shortcut_w + pad)
            .max(anchor.w)
            .max(self.theme.context_menu_min_w);
        let dd_x = anchor.x;
        let dd_y = anchor.y + anchor.h;

        // Handle click inside dropdown.
        let mut result: Option<u64> = None;
        if let Some((cx, cy, ref mut consumed)) = self.state.mouse.pending_click {
            if cx >= dd_x && cx < dd_x + dd_w && cy >= dd_y && cy < dd_y + total_h {
                // Find which item was clicked.
                let mut iy = dd_y;
                for entry in &menu.items {
                    match entry {
                        MenuEntry::Item(item) => {
                            if cy >= iy && cy < iy + item_h && item.enabled {
                                result = Some(item.id);
                                *consumed = true;
                                break;
                            }
                            iy += item_h;
                        }
                        MenuEntry::Separator => {
                            iy += sep_h;
                        }
                    }
                }
                if result.is_none() {
                    // Clicked on disabled item or separator — consume but don't act.
                    *consumed = true;
                }
            }
        }

        // Shadow.
        self.frame.push(
            ShapeBuilder::rect(dd_x - 1.0, dd_y - 1.0, dd_w + 2.0, total_h + 2.0)
                .color(self.theme.shadow)
                .border_radius(BorderRadius::uniform(corner_r))
                .build(),
        );

        // Background.
        self.frame.push(
            ShapeBuilder::rect(dd_x, dd_y, dd_w, total_h)
                .color(self.theme.bg_raised)
                .border_radius(BorderRadius::uniform(corner_r))
                .build(),
        );

        // Border.
        paint::draw_border(
            self.frame,
            Rect::new(dd_x, dd_y, dd_w, total_h),
            self.theme.border,
        );

        // Items.
        let mut iy = dd_y;
        for entry in &menu.items {
            match entry {
                MenuEntry::Item(item) => {
                    let row_rect = Rect::new(dd_x, iy, dd_w, item_h);
                    let hovered = row_rect.contains(self.state.mouse.x, self.state.mouse.y)
                        && item.enabled;

                    // Hover highlight.
                    if hovered {
                        self.frame.push(
                            ShapeBuilder::rect(dd_x + 1.0, iy, dd_w - 2.0, item_h)
                                .color(self.theme.bg_input)
                                .build(),
                        );
                    }

                    // Label text.
                    let text_color = if !item.enabled {
                        self.theme.fg_dim
                    } else if hovered {
                        self.theme.fg
                    } else {
                        self.theme.fg
                    };

                    self.text.draw_text(
                        &item.label,
                        dd_x + pad,
                        iy + (item_h - font_size) / 2.0,
                        font_size,
                        text_color,
                        self.frame,
                        self.gpu,
                        self.resources,
                    );

                    // Shortcut text (right-aligned, muted).
                    if let Some(ref sc) = item.shortcut {
                        let sc_w = self.text.measure_text(sc, font_size);
                        let sc_color = if !item.enabled {
                            self.theme.fg_dim
                        } else {
                            self.theme.fg_muted
                        };

                        self.text.draw_text(
                            sc,
                            dd_x + dd_w - pad - sc_w,
                            iy + (item_h - font_size) / 2.0,
                            font_size,
                            sc_color,
                            self.frame,
                            self.gpu,
                            self.resources,
                        );
                    }

                    iy += item_h;
                }
                MenuEntry::Separator => {
                    let sep_y = iy + sep_h / 2.0;
                    self.frame.push(
                        ShapeBuilder::rect(dd_x + pad, sep_y, dd_w - pad * 2.0, 1.0)
                            .color(self.theme.border)
                            .build(),
                    );
                    iy += sep_h;
                }
            }
        }

        result
    }
}
