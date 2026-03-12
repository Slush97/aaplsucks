//! Form field helpers — label + widget + validation helper text.
//!
//! # Examples
//!
//! ```ignore
//! ui.form_field("Email", FieldStatus::Error, "Invalid email address", |ui| {
//!     ui.text_input(id!("email"), &mut email, "user@example.com")
//! });
//! ```

use crate::response::Response;
use crate::Ui;

/// Validation status for a form field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldStatus {
    /// No validation state — default appearance.
    None,
    /// Validation error — red indicator.
    Error,
    /// Validation success — green indicator.
    Success,
    /// Warning — amber indicator.
    Warning,
}

impl<'f> Ui<'f> {
    /// Draw a form field: label above, widget via closure, optional helper text below.
    ///
    /// Returns the `Response` from the widget closure.
    pub fn form_field(
        &mut self,
        label: &str,
        status: FieldStatus,
        helper: &str,
        f: impl FnOnce(&mut Self) -> Response,
    ) -> Response {
        // Label.
        self.label_colored(label, self.theme.fg_label);
        self.add_space(self.theme.form_label_gap);

        // Widget.
        let response = f(self);

        // Helper text.
        if !helper.is_empty() {
            let color = match status {
                FieldStatus::None => self.theme.fg_dim,
                FieldStatus::Error => self.theme.red,
                FieldStatus::Success => self.theme.green,
                FieldStatus::Warning => self.theme.amber,
            };
            self.add_space(self.theme.form_helper_gap);
            let rect = self.allocate_rect(
                self.region.w,
                self.theme.form_helper_font_size,
            );
            self.text.draw_text(
                helper,
                rect.x,
                rect.y,
                self.theme.form_helper_font_size,
                color,
                self.frame,
                self.gpu,
                self.resources,
            );
        }

        response
    }
}
