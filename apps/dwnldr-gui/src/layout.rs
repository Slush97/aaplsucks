//! Layout computation for the three-panel layout.

use crate::tools::{self, Category};

/// Sidebar width in logical pixels.
pub const SIDEBAR_WIDTH: f32 = 180.0;
/// Activity bar height in logical pixels.
const ACTIVITY_BAR_HEIGHT: f32 = 36.0;
/// Tool item height in logical pixels.
const ITEM_HEIGHT: f32 = 32.0;
/// Padding in logical pixels.
const PADDING: f32 = 12.0;
/// Height of sidebar category header rows.
const SIDEBAR_HEADER_HEIGHT: f32 = 24.0;
/// Gap between sidebar category groups.
const SIDEBAR_GROUP_GAP: f32 = PADDING;
/// Height of the sidebar footer (theme toggle button).
pub const SIDEBAR_FOOTER_H: f32 = 40.0;

/// A positioned rectangle.
#[derive(Debug, Clone, Copy)]
pub struct LayoutRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// A sidebar item with its position and associated tool.
#[derive(Debug, Clone, Copy)]
pub struct SidebarItem {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    /// Tool ID for tool items; `""` for category headers.
    pub tool_id: &'static str,
    pub is_header: bool,
    /// Category label for header items (e.g. `"VIDEO"`); `""` for tool items.
    pub header_label: &'static str,
}

/// Complete layout tree — all positioned elements for one frame.
pub struct LayoutTree {
    pub sidebar: LayoutRect,
    pub content: LayoutRect,
    pub activity_bar: LayoutRect,
    pub sidebar_items: Vec<SidebarItem>,
}

impl LayoutTree {
    pub fn new(vw: f32, vh: f32) -> Self {
        Self::new_scaled(vw, vh, 1.0)
    }

    pub fn new_scaled(vw: f32, vh: f32, scale: f32) -> Self {
        let sidebar_w = SIDEBAR_WIDTH * scale;
        let activity_h = ACTIVITY_BAR_HEIGHT * scale;
        let item_h = ITEM_HEIGHT * scale;
        let pad = PADDING * scale;
        let header_h = SIDEBAR_HEADER_HEIGHT * scale;
        let group_gap = SIDEBAR_GROUP_GAP * scale;

        let sidebar = LayoutRect {
            x: 0.0,
            y: 0.0,
            w: sidebar_w,
            h: vh - activity_h,
        };

        let content = LayoutRect {
            x: sidebar_w,
            y: 0.0,
            w: vw - sidebar_w,
            h: vh - activity_h,
        };

        let activity_bar = LayoutRect {
            x: 0.0,
            y: vh - activity_h,
            w: vw,
            h: activity_h,
        };

        let mut sidebar_items = Vec::new();
        let mut y_cursor = pad;

        for &category in Category::ALL {
            sidebar_items.push(SidebarItem {
                x: 0.0,
                y: y_cursor,
                w: sidebar_w,
                h: header_h,
                tool_id: "",
                is_header: true,
                header_label: category.label(),
            });
            y_cursor += header_h;

            for tool in tools::tools_in_category(category) {
                sidebar_items.push(SidebarItem {
                    x: 0.0,
                    y: y_cursor,
                    w: sidebar_w,
                    h: item_h,
                    tool_id: tool.id,
                    is_header: false,
                    header_label: "",
                });
                y_cursor += item_h;
            }

            y_cursor += group_gap;
        }

        Self {
            sidebar,
            content,
            activity_bar,
            sidebar_items,
        }
    }
}
