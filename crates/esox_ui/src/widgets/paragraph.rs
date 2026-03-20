//! Paragraph widget — word-wrapped, styled, hover-aware text block.

use crate::response::Response;
use crate::state::{A11yNode, A11yRole};
use crate::Ui;

impl<'f> Ui<'f> {
    /// Draw a paragraph of wrapped text. Returns a `Response` with hover state.
    ///
    /// Measures text via `measure_text_wrapped`, allocates a rect, draws
    /// wrapped lines with `theme.line_spacing`, and registers for hit testing.
    pub fn paragraph(&mut self, id: u64, text: &str) -> Response {
        let size = self.resolve_font_size();
        let fg = self.resolve_fg();
        let max_width = self.region.w;
        let line_spacing = self.theme.line_spacing;
        let line_height = self.text.line_height(size);

        let (_, measured_h) = self.text.measure_text_wrapped(text, size, max_width, line_spacing);
        let total_height = measured_h + self.theme.label_pad_y;
        let rect = self.allocate_rect_keyed(id, max_width, total_height);

        let lines = self.text.wrap_lines(text, size, max_width);
        let step = line_height + line_spacing;
        for (i, &(start, end)) in lines.iter().enumerate() {
            let line = &text[start..end].trim_start();
            self.text.draw_text(
                line,
                rect.x,
                rect.y + i as f32 * step,
                size,
                fg,
                self.frame,
                self.gpu,
                self.resources,
            );
        }

        let hovered = self.is_hovered(rect);

        self.push_a11y_node(A11yNode {
            id,
            role: A11yRole::Label,
            label: text.to_string(),
            value: None,
            rect,
            focused: false,
            disabled: false,
            expanded: None,
            selected: None,
            checked: None,
            value_range: None,
            children: Vec::new(),
        });

        Response {
            clicked: false,
            right_clicked: false,
            hovered,
            focused: false,
            changed: false,
            disabled: false,
        }
    }
}
