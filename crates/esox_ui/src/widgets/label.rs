//! Label widgets — text display without interaction.

use esox_gfx::Color;

use crate::rich_text::RichText;
use crate::state::{A11yNode, A11yRole};
use crate::text::TruncationMode;
use crate::theme::TextSize;
use crate::Ui;

impl<'f> Ui<'f> {
    /// Draw a label with the standard text color.
    pub fn label(&mut self, text: &str) {
        let font_size = self.resolve_font_size();
        let fg = self.resolve_fg();
        let rect = self.allocate_rect(self.region.w, font_size + self.theme.label_pad_y);
        self.push_a11y_node(A11yNode {
            id: crate::id::fnv1a_runtime(text), role: A11yRole::Label,
            label: text.to_string(), value: None, rect, focused: false,
            disabled: false, expanded: None, selected: None, checked: None,
            value_range: None, children: Vec::new(),
        });
        if (font_size - self.theme.font_size).abs() < 0.01 {
            self.text.draw_ui_text(
                text,
                rect.x,
                rect.y,
                fg,
                self.frame,
                self.gpu,
                self.resources,
            );
        } else {
            self.text.draw_text(
                text,
                rect.x,
                rect.y,
                font_size,
                fg,
                self.frame,
                self.gpu,
                self.resources,
            );
        }
    }

    /// Draw a label at a semantic text size.
    pub fn label_sized(&mut self, text: &str, size: TextSize) {
        let font_size = self.theme.resolve_text_size(size);
        let rect = self.allocate_rect(self.region.w, font_size + self.theme.label_pad_y);
        self.text.draw_text(
            text,
            rect.x,
            rect.y,
            font_size,
            self.theme.fg,
            self.frame,
            self.gpu,
            self.resources,
        );
    }

    /// Draw a label with a custom color.
    pub fn label_colored(&mut self, text: &str, color: Color) {
        let rect = self.allocate_rect(self.region.w, self.theme.font_size + self.theme.label_pad_y);
        self.text.draw_ui_text(
            text,
            rect.x,
            rect.y,
            color,
            self.frame,
            self.gpu,
            self.resources,
        );
    }

    /// Draw a heading (larger text).
    pub fn heading(&mut self, text: &str) {
        let rect = self.allocate_rect(self.region.w, self.theme.heading_height);
        self.text.draw_text(
            text,
            rect.x,
            rect.y,
            self.theme.heading_font_size,
            self.theme.fg,
            self.frame,
            self.gpu,
            self.resources,
        );
    }

    /// Draw a muted label (dimmer text).
    pub fn muted_label(&mut self, text: &str) {
        let rect = self.allocate_rect(self.region.w, self.theme.font_size + self.theme.label_pad_y);
        self.text.draw_ui_text(
            text,
            rect.x,
            rect.y,
            self.theme.fg_muted,
            self.frame,
            self.gpu,
            self.resources,
        );
    }

    /// Draw a small header label (for categories).
    pub fn header_label(&mut self, text: &str) {
        let header_font_size = self.theme.header_font_size;
        let rect = self.allocate_rect(self.region.w, header_font_size + self.theme.label_pad_y);
        self.text.draw_header_text(
            text,
            rect.x,
            rect.y,
            self.theme.fg_dim,
            self.frame,
            self.gpu,
            self.resources,
        );
    }

    /// Draw a word-wrapped label. Height varies based on content.
    pub fn label_wrapped(&mut self, text: &str) {
        let size = self.theme.font_size;
        let max_width = self.region.w;
        let line_height = self.text.line_height(size);
        let line_spacing = self.theme.line_spacing;
        let (_, measured_h) = self.text.measure_text_wrapped(text, size, max_width, line_spacing);
        let total_height = measured_h + self.theme.label_pad_y;
        let rect = self.allocate_rect(max_width, total_height);

        let lines = self.text.wrap_lines(text, size, max_width);
        let step = line_height + line_spacing;
        for (i, &(start, end)) in lines.iter().enumerate() {
            let line = &text[start..end].trim_start();
            self.text.draw_ui_text(
                line,
                rect.x,
                rect.y + i as f32 * step,
                self.theme.fg,
                self.frame,
                self.gpu,
                self.resources,
            );
        }
    }

    /// Draw a single-line label truncated with "…" if it overflows.
    pub fn label_truncated(&mut self, text: &str) {
        let rect = self.allocate_rect(self.region.w, self.theme.font_size + self.theme.label_pad_y);
        self.text.draw_text_truncated(
            text,
            rect.x,
            rect.y,
            self.theme.font_size,
            rect.w,
            self.theme.fg,
            self.frame,
            self.gpu,
            self.resources,
        );
    }

    /// Draw a single-line label truncated with a specific truncation mode.
    pub fn label_truncated_mode(&mut self, text: &str, mode: TruncationMode) {
        let rect = self.allocate_rect(self.region.w, self.theme.font_size + self.theme.label_pad_y);
        self.text.draw_text_truncated_mode(
            text,
            rect.x,
            rect.y,
            self.theme.font_size,
            rect.w,
            self.theme.fg,
            mode,
            self.frame,
            self.gpu,
            self.resources,
        );
    }

    /// Draw a single-line rich text label with styled spans.
    pub fn rich_label(&mut self, rich: &RichText<'_>) {
        let font_size = self.theme.font_size;
        let fg = self.theme.fg;

        // Measure total width.
        let mut total_w = 0.0f32;
        for span in &rich.spans {
            let size = span.size.unwrap_or(font_size);
            total_w += self.text.measure_text(span.text, size);
        }

        let rect = self.allocate_rect(
            total_w.min(self.region.w),
            font_size + self.theme.label_pad_y,
        );

        let mut pen_x = rect.x;
        for span in &rich.spans {
            let size = span.size.unwrap_or(font_size);
            let color = span.color.unwrap_or(fg);
            let style: u8 = if span.bold { 1 } else { 0 };
            let advance = self.text.draw_text_styled(
                span.text, pen_x, rect.y, size, color, style,
                self.frame, self.gpu, self.resources,
            );
            pen_x += advance;
        }
    }

    /// Draw a word-wrapped rich text label. Height varies based on content.
    pub fn rich_label_wrapped(&mut self, rich: &RichText<'_>) {
        let font_size = self.theme.font_size;
        let fg = self.theme.fg;
        let max_width = self.region.w;
        let line_height = self.text.line_height(font_size);

        // Flatten spans into words with style info.
        struct StyledWord<'a> {
            text: &'a str,
            color: Option<Color>,
            bold: bool,
            size: Option<f32>,
            width: f32,
        }

        let space_width = self.text.measure_text(" ", font_size);
        let mut words: Vec<StyledWord<'_>> = Vec::new();

        for span in &rich.spans {
            let size = span.size.unwrap_or(font_size);
            // Split span into words.
            for word in span.text.split_whitespace() {
                let w = self.text.measure_text(word, size);
                words.push(StyledWord {
                    text: word,
                    color: span.color,
                    bold: span.bold,
                    size: span.size,
                    width: w,
                });
            }
        }

        // Break into lines.
        struct LinePlan {
            start: usize,
            end: usize, // exclusive
        }

        let mut lines: Vec<LinePlan> = Vec::new();
        let mut line_start = 0;
        let mut line_w = 0.0f32;

        for (i, word) in words.iter().enumerate() {
            if line_w == 0.0 {
                line_w = word.width;
            } else if line_w + space_width + word.width <= max_width {
                line_w += space_width + word.width;
            } else {
                lines.push(LinePlan {
                    start: line_start,
                    end: i,
                });
                line_start = i;
                line_w = word.width;
            }
        }
        if line_start < words.len() || words.is_empty() {
            lines.push(LinePlan {
                start: line_start,
                end: words.len(),
            });
        }

        let total_height = lines.len() as f32 * line_height + self.theme.label_pad_y;
        let rect = self.allocate_rect(max_width, total_height);

        for (line_idx, line) in lines.iter().enumerate() {
            let mut pen_x = rect.x;
            let pen_y = rect.y + line_idx as f32 * line_height;

            for word_idx in line.start..line.end {
                let word = &words[word_idx];
                if word_idx > line.start {
                    pen_x += space_width;
                }
                let size = word.size.unwrap_or(font_size);
                let color = word.color.unwrap_or(fg);
                let style: u8 = if word.bold { 1 } else { 0 };
                let advance = self.text.draw_text_styled(
                    word.text, pen_x, pen_y, size, color, style,
                    self.frame, self.gpu, self.resources,
                );
                pen_x += advance;
            }
        }
    }
}
