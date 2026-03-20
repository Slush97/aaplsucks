//! Empty state widget — centered placeholder for empty lists/tables.
//!
//! # Examples
//!
//! ```ignore
//! if items.is_empty() {
//!     ui.empty_state("No items yet");
//! }
//!
//! // With an action button
//! if items.is_empty() {
//!     if ui.empty_state_with_action(id!("add"), "No items yet", "Add item").clicked {
//!         show_add_dialog = true;
//!     }
//! }
//! ```

use crate::response::Response;
use crate::Ui;

impl<'f> Ui<'f> {
    /// Draw a centered empty-state message in muted, slightly larger text.
    pub fn empty_state(&mut self, message: &str) {
        let pad_y = self.theme.padding * 2.0;
        let font_size = self.theme.font_size * 1.15;
        let text_w = self.text.measure_text(message, font_size);

        self.add_space(pad_y);

        let rect = self.allocate_rect(self.region.w, font_size + self.theme.label_pad_y);
        let tx = rect.x + (rect.w - text_w) / 2.0;
        self.text.draw_text(
            message,
            tx,
            rect.y,
            font_size,
            self.theme.fg_muted,
            self.frame,
            self.gpu,
            self.resources,
        );

        self.add_space(pad_y);
    }

    /// Draw an empty-state message with an action button below. Returns a Response
    /// where `clicked` means the action button was pressed.
    pub fn empty_state_with_action(
        &mut self,
        id: u64,
        message: &str,
        action_label: &str,
    ) -> Response {
        let pad_y = self.theme.padding * 2.0;
        let font_size = self.theme.font_size * 1.15;
        let text_w = self.text.measure_text(message, font_size);

        self.add_space(pad_y);

        // Centered message.
        let msg_rect = self.allocate_rect_keyed(id, self.region.w, font_size + self.theme.label_pad_y);
        let tx = msg_rect.x + (msg_rect.w - text_w) / 2.0;
        self.text.draw_text(
            message,
            tx,
            msg_rect.y,
            font_size,
            self.theme.fg_muted,
            self.frame,
            self.gpu,
            self.resources,
        );

        self.add_space(self.theme.padding);

        // Centered ghost button.
        let btn_label_w = self.text.measure_text(action_label, self.theme.font_size);
        let btn_w =
            (btn_label_w + self.theme.input_padding * 4.0).max(self.theme.small_button_min_w);

        let mut response = Response::default();
        self.center_horizontal(btn_w, |ui| {
            response = ui.ghost_button(id, action_label);
        });

        self.add_space(pad_y);

        response
    }
}
