//! Collapsing header / accordion widget — expandable section with header row.

use esox_gfx::ShapeBuilder;
use esox_input::{Key, NamedKey};

use crate::paint;
use crate::response::Response;
use crate::state::{A11yNode, A11yRole, WidgetKind};
use crate::Ui;

impl<'f> Ui<'f> {
    /// Draw a collapsing header. When open, calls `f` to draw content indented below.
    ///
    /// State is stored in `UiState.collapsing_open`. On first encounter with
    /// `default_open == true`, the id is inserted.
    pub fn collapsing_header(
        &mut self,
        id: u64,
        label: &str,
        default_open: bool,
        f: impl FnOnce(&mut Self),
    ) -> Response {
        // Initialize default state on first encounter.
        // We use a side set to track "seen" ids — if not yet seen, apply default.
        let is_open = if self.state.collapsing_open.contains(&id) {
            true
        } else {
            // Check if we've ever toggled this — we can't distinguish "closed" from
            // "never seen" with just a HashSet, so we use a simple heuristic:
            // if default_open and not in the set, insert it (first frame).
            // Subsequent closes will remove it.
            if default_open {
                self.state.collapsing_open.insert(id);
                true
            } else {
                false
            }
        };

        let item_h = self.theme.item_height;
        let font_size = self.theme.font_size;
        let pad = self.theme.input_padding;

        // Draw header row.
        let rect = self.allocate_rect_keyed(id, self.region.w, item_h);
        self.register_widget(id, rect, WidgetKind::Button);
        let mut response = self.widget_response(id, rect);

        self.push_a11y_node(A11yNode {
            id,
            role: A11yRole::Group,
            label: label.to_string(),
            value: None,
            rect,
            focused: response.focused,
            disabled: response.disabled,
            expanded: Some(is_open),
            selected: None,
            checked: None,
            value_range: None,
            children: Vec::new(),
        });

        // Click toggles.
        if response.clicked {
            if is_open {
                self.state.collapsing_open.remove(&id);
            } else {
                self.state.collapsing_open.insert(id);
            }
            response.changed = true;
        }

        // Keyboard: ArrowLeft collapses, ArrowRight expands.
        // (Enter/Space activation is handled by widget_response since this is WidgetKind::Button.)
        if response.focused {
            let keys: Vec<_> = self.state.keys.clone();
            for (event, _mods) in &keys {
                if !event.pressed {
                    continue;
                }
                match &event.key {
                    Key::Named(NamedKey::ArrowLeft) => {
                        if is_open {
                            self.state.collapsing_open.remove(&id);
                            response.changed = true;
                        }
                    }
                    Key::Named(NamedKey::ArrowRight) => {
                        if !is_open {
                            self.state.collapsing_open.insert(id);
                            response.changed = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        // Focus ring.
        if response.focused {
            paint::draw_focus_ring(
                self.frame,
                rect,
                self.theme.accent_dim,
                self.theme.corner_radius,
                self.theme.focus_ring_expand,
            );
        }

        // Background on hover.
        if response.hovered {
            self.frame.push(
                ShapeBuilder::rect(rect.x, rect.y, rect.w, rect.h)
                    .color(self.theme.bg_raised)
                    .build(),
            );
        }

        // Expand icon: ▶ / ▼
        let icon_x = rect.x + pad;
        let icon_y = rect.y + (item_h - font_size) / 2.0;
        let icon = if is_open { "\u{25BC}" } else { "\u{25B6}" };
        let icon_w = self.text.draw_text(
            icon,
            icon_x,
            icon_y,
            font_size * 0.7,
            self.theme.fg_muted,
            self.frame,
            self.gpu,
            self.resources,
        );

        // Label.
        let label_x = icon_x + icon_w + pad;
        self.text.draw_text(
            label,
            label_x,
            icon_y,
            font_size,
            self.theme.fg,
            self.frame,
            self.gpu,
            self.resources,
        );

        // Re-read open state after potential keyboard toggle.
        let now_open = self.state.collapsing_open.contains(&id);

        // Draw content when open, indented by padding.
        if now_open {
            let indent = self.theme.padding;
            let saved_cursor_x = self.cursor.x;
            let saved_region = self.region;

            self.cursor.x += indent;
            self.region = crate::layout::Rect::new(
                self.cursor.x,
                self.region.y,
                self.region.w - indent,
                self.region.h,
            );

            f(self);

            self.cursor.x = saved_cursor_x;
            self.region = saved_region;
        }

        response
    }
}
