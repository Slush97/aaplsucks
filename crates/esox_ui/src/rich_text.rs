//! Rich text — multi-span text with color, bold, and size variations.

use esox_gfx::Color;

/// A single span of styled text.
#[derive(Debug, Clone, Copy)]
pub struct Span<'a> {
    pub text: &'a str,
    pub color: Option<Color>,
    pub bold: bool,
    pub size: Option<f32>,
}

/// Builder for multi-span rich text.
pub struct RichText<'a> {
    pub(crate) spans: Vec<Span<'a>>,
}

impl<'a> RichText<'a> {
    /// Create a new empty rich text builder.
    pub fn new() -> Self {
        Self { spans: Vec::new() }
    }

    /// Add a plain text span.
    pub fn span(mut self, text: &'a str) -> Self {
        self.spans.push(Span {
            text,
            color: None,
            bold: false,
            size: None,
        });
        self
    }

    /// Add a bold text span.
    pub fn bold(mut self, text: &'a str) -> Self {
        self.spans.push(Span {
            text,
            color: None,
            bold: true,
            size: None,
        });
        self
    }

    /// Add a colored text span.
    pub fn colored(mut self, text: &'a str, color: Color) -> Self {
        self.spans.push(Span {
            text,
            color: Some(color),
            bold: false,
            size: None,
        });
        self
    }

    /// Add a colored bold text span.
    pub fn colored_bold(mut self, text: &'a str, color: Color) -> Self {
        self.spans.push(Span {
            text,
            color: Some(color),
            bold: true,
            size: None,
        });
        self
    }

    /// Add a span with custom size.
    pub fn sized(mut self, text: &'a str, size: f32) -> Self {
        self.spans.push(Span {
            text,
            color: None,
            bold: false,
            size: Some(size),
        });
        self
    }

    /// Add a fully customized span.
    pub fn push(mut self, span: Span<'a>) -> Self {
        self.spans.push(span);
        self
    }
}

impl<'a> Default for RichText<'a> {
    fn default() -> Self {
        Self::new()
    }
}
