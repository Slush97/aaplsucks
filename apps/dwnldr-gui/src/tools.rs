//! Tool registry — data-driven definitions for all dwnldr operations.
//!
//! Each tool is a static definition. The UI reads these to generate layout
//! nodes, and the dispatcher matches on `id` to call dwnldr library functions.

/// How the tool accepts input.
#[derive(Debug, Clone)]
pub enum InputKind {
    /// One or more files with extension filter.
    File {
        /// Accepted extensions (e.g., `[".mp4", ".mkv"]`).
        accept: &'static [&'static str],
        /// Whether multiple files can be selected.
        multiple: bool,
    },
    /// A folder path.
    Folder,
    /// Free-form text input.
    Text {
        /// Placeholder hint.
        placeholder: &'static str,
    },
    /// A URL input.
    Url,
}

/// A single option field rendered in the tool view.
#[derive(Debug, Clone)]
pub enum OptionKind {
    /// Numeric input.
    Number {
        /// Placeholder hint.
        placeholder: Option<&'static str>,
    },
    /// Dropdown select.
    Select {
        /// Available choices.
        choices: &'static [&'static str],
    },
    /// Range slider.
    Slider {
        /// Minimum value.
        min: f32,
        /// Maximum value.
        max: f32,
        /// Default value.
        default: f32,
    },
    /// Time input (MM:SS).
    Time,
    /// Text input.
    Text {
        /// Placeholder hint.
        placeholder: Option<&'static str>,
    },
    /// Boolean toggle.
    Toggle,
}

/// Definition of a single option in a tool.
#[derive(Debug, Clone)]
pub struct OptionDef {
    /// Key used when passing to the backend.
    pub key: &'static str,
    /// Human-readable label.
    pub label: &'static str,
    /// The type of input control.
    pub kind: OptionKind,
}

/// What the tool produces.
#[derive(Debug, Clone, Copy)]
pub enum OutputKind {
    /// A single file.
    File,
    /// Multiple files.
    Files,
    /// Text output.
    Text,
}

/// Tool category for sidebar grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    /// Download from URL.
    Download,
    /// Video operations.
    Video,
    /// Image operations.
    Image,
    /// PDF operations.
    Pdf,
    /// QR code and OCR.
    QrOcr,
    /// Archive operations.
    Archive,
}

impl Category {
    /// Display label for the sidebar.
    pub fn label(self) -> &'static str {
        match self {
            Self::Download => "DOWNLOAD",
            Self::Video => "VIDEO",
            Self::Image => "IMAGE",
            Self::Pdf => "PDF",
            Self::QrOcr => "QR & TEXT",
            Self::Archive => "ARCHIVE",
        }
    }

    /// All categories in sidebar order.
    pub const ALL: &[Category] = &[
        Self::Download,
        Self::Video,
        Self::Image,
        Self::Pdf,
        Self::QrOcr,
        Self::Archive,
    ];
}

/// A complete tool definition. The UI is generated from this.
#[derive(Debug, Clone)]
pub struct ToolDef {
    /// Unique identifier.
    pub id: &'static str,
    /// Display label.
    pub label: &'static str,
    /// Unicode glyph icon for the sidebar.
    pub icon: &'static str,
    /// Sidebar category.
    pub category: Category,
    /// How the tool accepts input.
    pub input: InputKind,
    /// Configurable options.
    pub options: &'static [OptionDef],
    /// What the tool produces.
    pub output: OutputKind,
    /// Action button label.
    pub action_label: &'static str,
}

/// The complete tool registry.
pub static TOOLS: &[ToolDef] = &[
    // ── Download ──
    ToolDef {
        id: "download",
        label: "Download",
        icon: "\u{2B07}",
        category: Category::Download,
        input: InputKind::Url,
        options: &[
            OptionDef {
                key: "quality",
                label: "Quality",
                kind: OptionKind::Select {
                    choices: &["best", "1080p", "720p", "480p", "audio-only"],
                },
            },
            OptionDef {
                key: "format",
                label: "Format",
                kind: OptionKind::Select {
                    choices: &["mp4", "mkv", "webm", "mp3", "flac", "wav"],
                },
            },
        ],
        output: OutputKind::File,
        action_label: "Download",
    },
    // ── Video ──
    ToolDef {
        id: "trim",
        label: "Trim",
        icon: "\u{2702}",
        category: Category::Video,
        input: InputKind::File {
            accept: &[".mp4", ".mkv", ".webm", ".avi", ".mov", ".mp3", ".flac", ".wav"],
            multiple: false,
        },
        options: &[
            OptionDef {
                key: "start",
                label: "Start",
                kind: OptionKind::Time,
            },
            OptionDef {
                key: "end",
                label: "End",
                kind: OptionKind::Time,
            },
        ],
        output: OutputKind::File,
        action_label: "Trim",
    },
    ToolDef {
        id: "convert",
        label: "Convert",
        icon: "\u{21C4}",
        category: Category::Video,
        input: InputKind::File {
            accept: &[
                ".mp4", ".mkv", ".webm", ".avi", ".mov", ".mp3", ".flac", ".wav", ".ogg", ".png",
                ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff",
            ],
            multiple: false,
        },
        options: &[OptionDef {
            key: "format",
            label: "Format",
            kind: OptionKind::Select {
                choices: &[
                    "mp4", "mkv", "webm", "mp3", "flac", "wav", "ogg", "aac", "png", "jpg",
                    "webp", "gif",
                ],
            },
        }],
        output: OutputKind::File,
        action_label: "Convert",
    },
    // ── Image ──
    ToolDef {
        id: "resize",
        label: "Resize",
        icon: "\u{2922}",
        category: Category::Image,
        input: InputKind::File {
            accept: &[".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"],
            multiple: false,
        },
        options: &[
            OptionDef {
                key: "width",
                label: "Width (px)",
                kind: OptionKind::Number { placeholder: None },
            },
            OptionDef {
                key: "height",
                label: "Height (px)",
                kind: OptionKind::Number { placeholder: None },
            },
            OptionDef {
                key: "scale",
                label: "Scale (%)",
                kind: OptionKind::Number { placeholder: None },
            },
        ],
        output: OutputKind::File,
        action_label: "Resize",
    },
    ToolDef {
        id: "compress-image",
        label: "Compress",
        icon: "\u{25BC}",
        category: Category::Image,
        input: InputKind::File {
            accept: &[".png", ".jpg", ".jpeg", ".webp"],
            multiple: false,
        },
        options: &[OptionDef {
            key: "quality",
            label: "Quality",
            kind: OptionKind::Slider {
                min: 1.0,
                max: 100.0,
                default: 80.0,
            },
        }],
        output: OutputKind::File,
        action_label: "Compress",
    },
    // ── PDF ──
    ToolDef {
        id: "pdf-split",
        label: "Split",
        icon: "\u{2702}",
        category: Category::Pdf,
        input: InputKind::File {
            accept: &[".pdf"],
            multiple: false,
        },
        options: &[OptionDef {
            key: "ranges",
            label: "Page ranges",
            kind: OptionKind::Text {
                placeholder: Some("1-3, 5, 7-10"),
            },
        }],
        output: OutputKind::Files,
        action_label: "Split",
    },
    ToolDef {
        id: "pdf-merge",
        label: "Merge",
        icon: "\u{2295}",
        category: Category::Pdf,
        input: InputKind::File {
            accept: &[".pdf"],
            multiple: true,
        },
        options: &[],
        output: OutputKind::File,
        action_label: "Merge",
    },
    ToolDef {
        id: "pdf-compress",
        label: "Compress PDF",
        icon: "\u{25BC}",
        category: Category::Pdf,
        input: InputKind::File {
            accept: &[".pdf"],
            multiple: false,
        },
        options: &[OptionDef {
            key: "quality",
            label: "Quality",
            kind: OptionKind::Select {
                choices: &["screen", "ebook", "printer"],
            },
        }],
        output: OutputKind::File,
        action_label: "Compress",
    },
    // ── QR & Text ──
    ToolDef {
        id: "qr-gen",
        label: "Generate QR",
        icon: "\u{25A3}",
        category: Category::QrOcr,
        input: InputKind::Text {
            placeholder: "URL or text to encode...",
        },
        options: &[],
        output: OutputKind::File,
        action_label: "Generate",
    },
    ToolDef {
        id: "qr-decode",
        label: "Decode QR",
        icon: "\u{25A3}",
        category: Category::QrOcr,
        input: InputKind::File {
            accept: &[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
            multiple: false,
        },
        options: &[],
        output: OutputKind::Text,
        action_label: "Decode",
    },
    // ── Archive ──
    ToolDef {
        id: "zip",
        label: "Zip",
        icon: "\u{1F4E6}",
        category: Category::Archive,
        input: InputKind::File {
            accept: &["*"],
            multiple: true,
        },
        options: &[],
        output: OutputKind::File,
        action_label: "Zip",
    },
    ToolDef {
        id: "unzip",
        label: "Unzip",
        icon: "\u{1F4E4}",
        category: Category::Archive,
        input: InputKind::File {
            accept: &[".zip"],
            multiple: false,
        },
        options: &[],
        output: OutputKind::Files,
        action_label: "Extract",
    },
];

/// Look up a tool by ID.
pub fn find_tool(id: &str) -> Option<&'static ToolDef> {
    TOOLS.iter().find(|t| t.id == id)
}

/// Get all tools in a given category.
pub fn tools_in_category(category: Category) -> impl Iterator<Item = &'static ToolDef> {
    TOOLS.iter().filter(move |t| t.category == category)
}
