//! Tab bar widget — switchable tabs with optional content area and fade animation.
//!
//! # Examples
//!
//! ```ignore
//! let mut tab = TabState::new();
//! ui.tabs(id!("settings"), &mut tab, &["General", "Advanced"], |ui, idx| {
//!     match idx {
//!         0 => ui.label("General settings"),
//!         1 => ui.label("Advanced settings"),
//!         _ => {}
//!     }
//! });
//! ```

use esox_gfx::ShapeBuilder;
use esox_input::{Key, NamedKey};

use crate::id::fnv1a_mix;
use crate::layout::Rect;
use crate::paint;
use crate::response::Response;
use crate::state::{Easing, TabState, WidgetKind};
use crate::Ui;

const TAB_FADE_SALT: u64 = 0xFADE_7AB5_0000_0001;

impl<'f> Ui<'f> {
    /// Tab bar + content area. Closure draws content for the selected tab.
    pub fn tabs(
        &mut self,
        id: u64,
        state: &mut TabState,
        labels: &[&str],
        content: impl FnOnce(&mut Self, usize),
    ) -> Response {
        let prev_selected = state.selected;
        let response = self.tab_bar(id, state, labels);

        // Tab content fade animation.
        if state.selected != prev_selected || response.changed {
            // Selection changed — restart fade.
        }
        let fade_id = fnv1a_mix(id, TAB_FADE_SALT);
        let _fade_t = self.state.anim_t(fade_id, 1.0, self.theme.tab_fade_duration_ms, Easing::EaseOutCubic);

        content(self, state.selected);
        response
    }

    /// Tab bar only (no content area).
    pub fn tab_bar(&mut self, id: u64, state: &mut TabState, labels: &[&str]) -> Response {
        let font_size = self.theme.font_size;
        let pad = self.theme.input_padding;
        let indicator_h = self.theme.tab_indicator_height;
        let bar_height = font_size + pad * 2.0 + indicator_h;

        let bar_rect = self.allocate_rect_keyed(id, self.region.w, bar_height);

        let mut response = Response::default();

        // Keyboard: Left/Right cycle tabs when focused.
        let bar_focused = self.state.focused == Some(id);
        if bar_focused {
            let keys: Vec<_> = self.state.keys.clone();
            for (event, _mods) in &keys {
                if !event.pressed {
                    continue;
                }
                match &event.key {
                    Key::Named(NamedKey::ArrowLeft) if state.selected > 0 => {
                        state.selected -= 1;
                        response.changed = true;
                    }
                    Key::Named(NamedKey::ArrowRight) if state.selected + 1 < labels.len() => {
                        state.selected += 1;
                        response.changed = true;
                    }
                    _ => {}
                }
            }
        }

        // Focus ring.
        if bar_focused {
            paint::draw_focus_ring(
                self.frame,
                bar_rect,
                self.theme.accent_dim,
                self.theme.corner_radius,
                self.theme.focus_ring_expand,
            );
        }

        // Draw separator line under tab bar.
        self.frame.push(
            ShapeBuilder::rect(bar_rect.x, bar_rect.y + bar_height - 1.0, bar_rect.w, 1.0)
                .color(self.theme.border)
                .build(),
        );

        // Draw each tab.
        let mut tab_x = bar_rect.x;
        for (i, label) in labels.iter().enumerate() {
            let text_w = self.text.measure_text(label, font_size);
            let tab_w = text_w + pad * 2.0;
            let tab_rect = Rect::new(tab_x, bar_rect.y, tab_w, bar_height);
            let tab_id = fnv1a_mix(id, i as u64);

            self.register_widget(tab_id, tab_rect, WidgetKind::Tab);
            let tab_response = self.widget_response(tab_id, tab_rect);

            if tab_response.clicked {
                state.selected = i;
                response.changed = true;
                self.state.focused = Some(id);
                // Reset fade animation.
                let fade_id = fnv1a_mix(id, TAB_FADE_SALT);
                if let Some(anim) = self.state.anims.get_mut(&fade_id) {
                    anim.from = 0.0;
                    anim.to = 1.0;
                    anim.start = std::time::Instant::now();
                }
            }

            let selected = state.selected == i;

            self.push_a11y_node(crate::state::A11yNode {
                id: tab_id, role: crate::state::A11yRole::Tab, label: label.to_string(),
                value: None, rect: tab_rect, focused: tab_response.focused, disabled: false,
                expanded: None, selected: Some(selected), checked: None,
                value_range: None, children: Vec::new(),
            });

            // Hover animation.
            let hover_t = self.state.hover_t(tab_id, tab_response.hovered && !selected, self.theme.hover_duration_ms);

            // Text color.
            let text_color = if selected {
                self.theme.accent
            } else {
                paint::lerp_color(self.theme.fg_muted, self.theme.fg, hover_t)
            };

            // Draw text.
            self.text.draw_text(
                label,
                tab_x + pad,
                bar_rect.y + pad,
                font_size,
                text_color,
                self.frame,
                self.gpu,
                self.resources,
            );

            // Selected indicator.
            if selected {
                self.frame.push(
                    ShapeBuilder::rect(
                        tab_x,
                        bar_rect.y + bar_height - indicator_h,
                        tab_w,
                        indicator_h,
                    )
                    .color(self.theme.accent)
                    .build(),
                );
            }

            tab_x += tab_w;
        }

        response.focused = bar_focused;
        response
    }
}
