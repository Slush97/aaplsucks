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
use esox_input::{Key, NamedKey};

use crate::layout::Rect;
use crate::paint;
use crate::Ui;

/// Data needed to paint a menu dropdown in `finish()` (deferred for z-order).
pub(crate) struct MenuBarDeferred {
    pub items: Vec<MenuBarDeferredItem>,
    pub dd_rect: Rect,
}

pub(crate) struct MenuBarDeferredItem {
    pub label: String,
    pub shortcut: Option<String>,
    pub enabled: bool,
    pub is_separator: bool,
}

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
                if event.pressed {
                    if let Key::Named(NamedKey::Escape) = &event.key {
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

        // Hit-test the open dropdown and defer painting to finish().
        if let Some(open_idx) = new_open {
            if open_idx < menus.len() {
                let menu = &menus[open_idx];
                let anchor = &label_rects[open_idx];
                let dd_rect = self.dropdown_rect(menu, anchor);

                // Hit-test for clicks inside dropdown.
                result = self.hit_test_menu_dropdown(menu, &dd_rect);
                if result.is_some() {
                    new_open = None;
                }

                // Store deferred paint data.
                let items = menu.items.iter().map(|entry| match entry {
                    MenuEntry::Item(item) => MenuBarDeferredItem {
                        label: item.label.clone(),
                        shortcut: item.shortcut.clone(),
                        enabled: item.enabled,
                        is_separator: false,
                    },
                    MenuEntry::Separator => MenuBarDeferredItem {
                        label: String::new(),
                        shortcut: None,
                        enabled: false,
                        is_separator: true,
                    },
                }).collect();

                self.state.menu_bar_deferred = Some(MenuBarDeferred {
                    items,
                    dd_rect,
                });
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

    /// Hit-test a menu dropdown for clicks. Returns the selected item id, if any.
    fn hit_test_menu_dropdown(&mut self, menu: &Menu, dd_rect: &Rect) -> Option<u64> {
        let item_h = self.theme.item_height;
        let sep_h: f32 = 9.0;

        let mut result: Option<u64> = None;
        if let Some((cx, cy, ref mut consumed)) = self.state.mouse.pending_click {
            if dd_rect.contains(cx, cy) {
                let mut iy = dd_rect.y;
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
                    *consumed = true;
                }
            }
        }
        result
    }

    /// Paint the deferred menu bar dropdown. Called from `finish()`.
    pub(crate) fn draw_deferred_menu_bar(&mut self) {
        let deferred = match self.state.menu_bar_deferred.take() {
            Some(d) => d,
            None => return,
        };

        let font_size = self.theme.font_size;
        let pad = self.theme.input_padding;
        let item_h = self.theme.item_height;
        let sep_h: f32 = 9.0;
        let corner_r = self.theme.corner_radius;

        let dd = &deferred.dd_rect;

        // Shadow.
        self.frame.push(
            ShapeBuilder::rect(dd.x - 1.0, dd.y - 1.0, dd.w + 2.0, dd.h + 2.0)
                .color(self.theme.shadow)
                .border_radius(BorderRadius::uniform(corner_r))
                .build(),
        );

        // Background.
        self.frame.push(
            ShapeBuilder::rect(dd.x, dd.y, dd.w, dd.h)
                .color(self.theme.bg_raised)
                .border_radius(BorderRadius::uniform(corner_r))
                .build(),
        );

        // Border.
        paint::draw_border(self.frame, *dd, self.theme.border);

        // Items.
        let mut iy = dd.y;
        for item in &deferred.items {
            if item.is_separator {
                let sep_y = iy + sep_h / 2.0;
                self.frame.push(
                    ShapeBuilder::rect(dd.x + pad, sep_y, dd.w - pad * 2.0, 1.0)
                        .color(self.theme.border)
                        .build(),
                );
                iy += sep_h;
            } else {
                let row_rect = Rect::new(dd.x, iy, dd.w, item_h);
                let hovered = row_rect.contains(self.state.mouse.x, self.state.mouse.y)
                    && item.enabled;

                if hovered {
                    self.frame.push(
                        ShapeBuilder::rect(dd.x + 1.0, iy, dd.w - 2.0, item_h)
                            .color(self.theme.bg_input)
                            .build(),
                    );
                }

                let text_color = if !item.enabled {
                    self.theme.fg_dim
                } else {
                    self.theme.fg
                };

                self.text.draw_text(
                    &item.label,
                    dd.x + pad,
                    iy + (item_h - font_size) / 2.0,
                    font_size,
                    text_color,
                    self.frame,
                    self.gpu,
                    self.resources,
                );

                if let Some(ref sc) = item.shortcut {
                    let sc_w = self.text.measure_text(sc, font_size);
                    let sc_color = if !item.enabled {
                        self.theme.fg_dim
                    } else {
                        self.theme.fg_muted
                    };

                    self.text.draw_text(
                        sc,
                        dd.x + dd.w - pad - sc_w,
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
        }
    }
}
