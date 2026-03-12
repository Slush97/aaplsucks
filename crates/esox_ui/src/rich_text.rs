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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_starts_empty() {
        let rt = RichText::new();
        assert!(rt.spans.is_empty());
    }

    #[test]
    fn builder_chain_produces_correct_spans() {
        let color = Color::new(1.0, 0.0, 0.0, 1.0);
        let rt = RichText::new()
            .span("a")
            .bold("b")
            .colored("c", color)
            .sized("d", 20.0);

        assert_eq!(rt.spans.len(), 4);

        // span("a") — plain
        assert_eq!(rt.spans[0].text, "a");
        assert!(!rt.spans[0].bold);
        assert!(rt.spans[0].color.is_none());
        assert!(rt.spans[0].size.is_none());

        // bold("b")
        assert_eq!(rt.spans[1].text, "b");
        assert!(rt.spans[1].bold);
        assert!(rt.spans[1].color.is_none());
        assert!(rt.spans[1].size.is_none());

        // colored("c", red)
        assert_eq!(rt.spans[2].text, "c");
        assert!(!rt.spans[2].bold);
        assert_eq!(rt.spans[2].color, Some(color));
        assert!(rt.spans[2].size.is_none());

        // sized("d", 20.0)
        assert_eq!(rt.spans[3].text, "d");
        assert!(!rt.spans[3].bold);
        assert!(rt.spans[3].color.is_none());
        assert_eq!(rt.spans[3].size, Some(20.0));
    }

    #[test]
    fn push_custom_span() {
        let custom = Span {
            text: "custom",
            color: Some(Color::new(0.0, 1.0, 0.0, 1.0)),
            bold: true,
            size: Some(32.0),
        };
        let rt = RichText::new().push(custom);
        assert_eq!(rt.spans.len(), 1);
        assert_eq!(rt.spans[0].text, "custom");
        assert!(rt.spans[0].bold);
        assert_eq!(rt.spans[0].color, Some(Color::new(0.0, 1.0, 0.0, 1.0)));
        assert_eq!(rt.spans[0].size, Some(32.0));
    }
}
