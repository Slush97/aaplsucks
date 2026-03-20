//! Two-pass layout tree: measure (bottom-up) then arrange (top-down).
//!
//! Built each frame by widget calls, solved after `finish()`, cached for the
//! next frame so widgets can look up their solved positions.

use std::collections::HashMap;

use crate::layout::{Align, Direction, FlexWrap, Justify, Rect, Spacing};

// ── Overflow mode ──

/// Controls whether a container clips/scrolls its children.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Overflow {
    /// Children may exceed the container bounds.
    #[default]
    Visible,
    /// Children are measured at intrinsic height (for scroll containers).
    Scroll,
}

// ── Node identity ──

/// Index into `LayoutTree::nodes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u32);

impl NodeId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

// ── Per-node style ──

/// Layout style for a single node. Container properties (direction, gap, …) and
/// per-child flex properties (grow, shrink, …) are stored together — a leaf
/// simply ignores the container fields.
#[derive(Debug, Clone)]
pub struct LayoutStyle {
    pub direction: Direction,
    pub padding: Spacing,
    pub gap: f32,
    pub align_items: Align,
    pub justify_content: Justify,
    pub flex_wrap: FlexWrap,
    pub min_width: Option<f32>,
    pub max_width: Option<f32>,
    pub min_height: Option<f32>,
    pub max_height: Option<f32>,
    pub overflow: Overflow,
    // Per-child (used by parent's solver):
    pub flex_grow: f32,
    pub flex_shrink: f32,
    pub flex_basis: Option<f32>,
    pub align_self: Option<Align>,
    pub margin: Spacing,
}

impl Default for LayoutStyle {
    fn default() -> Self {
        Self {
            direction: Direction::Vertical,
            padding: Spacing::default(),
            gap: 0.0,
            align_items: Align::Start,
            justify_content: Justify::Start,
            flex_wrap: FlexWrap::NoWrap,
            min_width: None,
            max_width: None,
            min_height: None,
            max_height: None,
            overflow: Overflow::default(),
            flex_grow: 0.0,
            flex_shrink: 1.0,
            flex_basis: None,
            align_self: None,
            margin: Spacing::default(),
        }
    }
}

// ── Tree node ──

#[derive(Debug, Clone)]
pub struct LayoutNode {
    pub key: u64,
    pub style: LayoutStyle,
    /// Measured (intrinsic) size — set by widgets for leaves, computed for containers.
    pub intrinsic: (f32, f32),
    /// Solved position + size (output of arrange pass).
    pub rect: Rect,
    /// Whether this is a leaf (widget) or container.
    pub is_leaf: bool,

    // Tree links (child-sibling representation).
    pub parent: Option<NodeId>,
    pub first_child: Option<NodeId>,
    pub last_child: Option<NodeId>,
    pub next_sibling: Option<NodeId>,
}

impl LayoutNode {
    fn new(key: u64) -> Self {
        Self {
            key,
            style: LayoutStyle::default(),
            intrinsic: (0.0, 0.0),
            rect: Rect::default(),
            is_leaf: true,
            parent: None,
            first_child: None,
            last_child: None,
            next_sibling: None,
        }
    }
}

// ── Layout tree ──

#[derive(Debug, Clone, Default)]
pub struct LayoutTree {
    pub nodes: Vec<LayoutNode>,
    pub key_index: HashMap<u64, NodeId>,
    pub root: Option<NodeId>,
}

impl LayoutTree {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, key: u64) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(LayoutNode::new(key));
        self.key_index.insert(key, id);
        if self.root.is_none() {
            self.root = Some(id);
        }
        id
    }

    pub fn node(&self, id: NodeId) -> &LayoutNode {
        &self.nodes[id.index()]
    }

    pub fn node_mut(&mut self, id: NodeId) -> &mut LayoutNode {
        &mut self.nodes[id.index()]
    }

    pub fn append_child(&mut self, parent: NodeId, child: NodeId) {
        self.nodes[child.index()].parent = Some(parent);
        let p = &mut self.nodes[parent.index()];
        p.is_leaf = false;
        if let Some(last) = p.last_child {
            self.nodes[last.index()].next_sibling = Some(child);
        } else {
            p.first_child = Some(child);
        }
        self.nodes[parent.index()].last_child = Some(child);
    }

    /// Lookup a solved rect by widget key. Returns `None` if the key wasn't in
    /// the previous frame's tree.
    pub fn lookup(&self, key: u64) -> Option<Rect> {
        self.key_index.get(&key).map(|id| self.nodes[id.index()].rect)
    }

    /// Lookup the intrinsic (measured) size by widget key.
    pub fn intrinsic_size(&self, key: u64) -> Option<(f32, f32)> {
        self.key_index.get(&key).map(|id| self.nodes[id.index()].intrinsic)
    }

    /// Iterate children of `parent`.
    pub fn children(&self, parent: NodeId) -> ChildIter<'_> {
        ChildIter { tree: self, next: self.nodes[parent.index()].first_child }
    }

    /// Count children of `parent`.
    #[cfg(test)]
    fn child_count(&self, parent: NodeId) -> usize {
        self.children(parent).count()
    }

    // ── Two-pass solver ──

    /// Run measure (bottom-up) then arrange (top-down) on the whole tree.
    pub fn solve(&mut self, viewport: Rect) {
        if let Some(root) = self.root {
            self.measure(root);
            self.arrange(root, viewport);
        }
    }

    /// Bottom-up: compute `intrinsic` sizes from leaves to root.
    fn measure(&mut self, id: NodeId) {
        // Recurse children first.
        let mut child = self.nodes[id.index()].first_child;
        while let Some(c) = child {
            self.measure(c);
            child = self.nodes[c.index()].next_sibling;
        }

        if self.nodes[id.index()].is_leaf {
            // Leaf intrinsic is already set by the widget.
            return;
        }

        // Container: sum children on main axis, max on cross axis.
        let node = &self.nodes[id.index()];
        let dir = node.style.direction;
        let pad = node.style.padding;
        let gap = node.style.gap;

        let mut main_sum = 0.0f32;
        let mut cross_max = 0.0f32;
        let mut count = 0u32;

        let mut child = node.first_child;
        while let Some(c) = child {
            let cn = &self.nodes[c.index()];
            let margin = &cn.style.margin;
            let (cw, ch) = cn.intrinsic;
            let (c_main, c_cross) = match dir {
                Direction::Horizontal => (
                    cw + margin.horizontal(),
                    ch + margin.vertical(),
                ),
                Direction::Vertical => (
                    ch + margin.vertical(),
                    cw + margin.horizontal(),
                ),
            };
            main_sum += c_main;
            cross_max = cross_max.max(c_cross);
            count += 1;
            child = cn.next_sibling;
        }

        if count > 1 {
            main_sum += gap * (count - 1) as f32;
        }

        let (mut w, mut h) = match dir {
            Direction::Horizontal => (main_sum, cross_max),
            Direction::Vertical => (cross_max, main_sum),
        };

        w += pad.horizontal();
        h += pad.vertical();

        // Apply min/max constraints.
        let style = &self.nodes[id.index()].style;
        if let Some(max) = style.max_width {
            w = w.min(max);
        }
        if let Some(min) = style.min_width {
            w = w.max(min);
        }
        if let Some(max) = style.max_height {
            h = h.min(max);
        }
        if let Some(min) = style.min_height {
            h = h.max(min);
        }

        self.nodes[id.index()].intrinsic = (w, h);
    }

    /// Top-down: distribute space from root to leaves.
    ///
    /// `available` already has the node's own margin applied by the parent.
    fn arrange(&mut self, id: NodeId, available: Rect) {
        self.nodes[id.index()].rect = available;

        // For scroll containers, use intrinsic height for content layout.
        let rect = if self.nodes[id.index()].style.overflow == Overflow::Scroll {
            let (iw, ih) = self.nodes[id.index()].intrinsic;
            Rect::new(available.x, available.y, available.w.max(iw), ih.max(available.h))
        } else {
            available
        };

        if self.nodes[id.index()].is_leaf {
            return;
        }

        let node = &self.nodes[id.index()];
        let dir = node.style.direction;
        let pad = node.style.padding;
        let gap = node.style.gap;
        let align = node.style.align_items;
        let justify = node.style.justify_content;
        let wrap = node.style.flex_wrap;

        let content_x = rect.x + pad.left;
        let content_y = rect.y + pad.top;
        let content_w = (rect.w - pad.horizontal()).max(0.0);
        let content_h = (rect.h - pad.vertical()).max(0.0);

        let available_main = match dir {
            Direction::Horizontal => content_w,
            Direction::Vertical => content_h,
        };
        let available_cross = match dir {
            Direction::Horizontal => content_h,
            Direction::Vertical => content_w,
        };

        let area = ContentArea {
            x: content_x,
            y: content_y,
            w: content_w,
            h: content_h,
        };

        if wrap != FlexWrap::NoWrap {
            self.arrange_wrapped(id, dir, &area, gap, align, justify, wrap);
            return;
        }

        // Collect child info for flex distribution.
        let mut child_info: Vec<ChildFlexInfo> = Vec::new();
        {
            let mut child = self.nodes[id.index()].first_child;
            while let Some(c) = child {
                let cn = &self.nodes[c.index()];
                let (iw, ih) = cn.intrinsic;
                let basis = cn.style.flex_basis.unwrap_or(match dir {
                    Direction::Horizontal => iw,
                    Direction::Vertical => ih,
                });
                let cross = match dir {
                    Direction::Horizontal => ih,
                    Direction::Vertical => iw,
                };
                child_info.push(ChildFlexInfo {
                    id: c,
                    basis,
                    cross,
                    grow: cn.style.flex_grow,
                    shrink: cn.style.flex_shrink,
                    margin: cn.style.margin,
                    align_self: cn.style.align_self,
                });
                child = cn.next_sibling;
            }
        }

        let n = child_info.len();
        if n == 0 {
            return;
        }

        // Compute total basis + margins on main axis.
        let total_gap = if n > 1 { gap * (n - 1) as f32 } else { 0.0 };
        let total_basis: f32 = child_info.iter().map(|ci| {
            ci.basis + match dir {
                Direction::Horizontal => ci.margin.horizontal(),
                Direction::Vertical => ci.margin.vertical(),
            }
        }).sum();

        let remaining = available_main - total_gap - total_basis;

        // Distribute remaining space.
        let mut sizes: Vec<f32> = child_info.iter().map(|ci| ci.basis).collect();
        if remaining > 0.0 {
            let total_grow: f32 = child_info.iter().map(|ci| ci.grow).sum();
            if total_grow > 0.0 {
                for (i, ci) in child_info.iter().enumerate() {
                    sizes[i] += remaining * ci.grow / total_grow;
                }
            }
        } else if remaining < 0.0 {
            let total_shrink: f32 = child_info.iter().map(|ci| ci.basis * ci.shrink).sum();
            if total_shrink > 0.0 {
                let overflow = -remaining;
                for (i, ci) in child_info.iter().enumerate() {
                    let scaled = ci.basis * ci.shrink;
                    sizes[i] = (sizes[i] - overflow * scaled / total_shrink).max(0.0);
                }
            }
        }

        // Justify: compute start offset and inter-item spacing adjustment.
        let used: f32 = sizes.iter().sum::<f32>()
            + child_info.iter().map(|ci| match dir {
                Direction::Horizontal => ci.margin.horizontal(),
                Direction::Vertical => ci.margin.vertical(),
            }).sum::<f32>()
            + total_gap;
        let free = (available_main - used).max(0.0);

        let (start_offset, extra_gap) = match justify {
            Justify::Start => (0.0, 0.0),
            Justify::End => (free, 0.0),
            Justify::Center => (free / 2.0, 0.0),
            Justify::SpaceBetween => {
                if n > 1 {
                    (0.0, free / (n - 1) as f32)
                } else {
                    (0.0, 0.0)
                }
            }
        };

        // Position children.
        let mut main_pos = match dir {
            Direction::Horizontal => area.x,
            Direction::Vertical => area.y,
        } + start_offset;

        for (i, ci) in child_info.iter().enumerate() {
            let main_size = sizes[i];
            let eff_align = ci.align_self.unwrap_or(align);

            // Cross-axis sizing and alignment.
            let (cross_size, cross_offset) = {
                let avail_cross = available_cross - match dir {
                    Direction::Horizontal => ci.margin.vertical(),
                    Direction::Vertical => ci.margin.horizontal(),
                };
                match eff_align {
                    Align::Stretch => (avail_cross, 0.0),
                    Align::Start => (ci.cross, 0.0),
                    Align::Center => (ci.cross, (avail_cross - ci.cross) / 2.0),
                    Align::End => (ci.cross, avail_cross - ci.cross),
                }
            };

            let child_rect = match dir {
                Direction::Horizontal => Rect::new(
                    main_pos + ci.margin.left,
                    content_y + ci.margin.top + cross_offset,
                    main_size,
                    cross_size,
                ),
                Direction::Vertical => Rect::new(
                    content_x + ci.margin.left + cross_offset,
                    main_pos + ci.margin.top,
                    cross_size,
                    main_size,
                ),
            };

            self.arrange(ci.id, child_rect);

            main_pos += main_size + gap + extra_gap + match dir {
                Direction::Horizontal => ci.margin.horizontal(),
                Direction::Vertical => ci.margin.vertical(),
            };
        }
    }

    /// Arrange children with wrapping.
    #[allow(clippy::too_many_arguments)]
    fn arrange_wrapped(
        &mut self,
        id: NodeId,
        dir: Direction,
        area: &ContentArea,
        gap: f32,
        align: Align,
        justify: Justify,
        wrap: FlexWrap,
    ) {
        let available_main = match dir {
            Direction::Horizontal => area.w,
            Direction::Vertical => area.h,
        };

        // Collect children.
        let mut children: Vec<NodeId> = Vec::new();
        let mut child = self.nodes[id.index()].first_child;
        while let Some(c) = child {
            children.push(c);
            child = self.nodes[c.index()].next_sibling;
        }

        if children.is_empty() {
            return;
        }

        // Break into wrap lines.
        struct WrapLine {
            start: usize,
            end: usize,
            main_used: f32,
            cross_max: f32,
        }

        let mut lines: Vec<WrapLine> = Vec::new();
        let mut line_start = 0;
        let mut line_main = 0.0f32;
        let mut line_cross = 0.0f32;
        let mut line_count = 0u32;

        for (i, &cid) in children.iter().enumerate() {
            let cn = &self.nodes[cid.index()];
            let (iw, ih) = cn.intrinsic;
            let margin = &cn.style.margin;
            let c_main = match dir {
                Direction::Horizontal => iw + margin.horizontal(),
                Direction::Vertical => ih + margin.vertical(),
            };
            let c_cross = match dir {
                Direction::Horizontal => ih + margin.vertical(),
                Direction::Vertical => iw + margin.horizontal(),
            };

            let with_gap = if line_count > 0 { gap } else { 0.0 };
            if line_count > 0 && line_main + with_gap + c_main > available_main {
                lines.push(WrapLine {
                    start: line_start,
                    end: i,
                    main_used: line_main,
                    cross_max: line_cross,
                });
                line_start = i;
                line_main = c_main;
                line_cross = c_cross;
                line_count = 1;
            } else {
                line_main += with_gap + c_main;
                line_cross = line_cross.max(c_cross);
                line_count += 1;
            }
        }
        lines.push(WrapLine {
            start: line_start,
            end: children.len(),
            main_used: line_main,
            cross_max: line_cross,
        });

        if wrap == FlexWrap::WrapReverse {
            lines.reverse();
        }

        // Position each line.
        let mut cross_pos = match dir {
            Direction::Horizontal => area.y,
            Direction::Vertical => area.x,
        };

        for line in &lines {
            let items = &children[line.start..line.end];
            let n = items.len();
            let free = (available_main - line.main_used).max(0.0);

            let (start_offset, extra_gap) = match justify {
                Justify::Start => (0.0, 0.0),
                Justify::End => (free, 0.0),
                Justify::Center => (free / 2.0, 0.0),
                Justify::SpaceBetween => {
                    if n > 1 { (0.0, free / (n - 1) as f32) } else { (0.0, 0.0) }
                }
            };

            let mut main_pos = match dir {
                Direction::Horizontal => area.x,
                Direction::Vertical => area.y,
            } + start_offset;

            for &cid in items {
                let cn = &self.nodes[cid.index()];
                let (iw, ih) = cn.intrinsic;
                let margin = cn.style.margin;
                let align_self = cn.style.align_self.unwrap_or(align);
                let basis = cn.style.flex_basis.unwrap_or(match dir {
                    Direction::Horizontal => iw,
                    Direction::Vertical => ih,
                });
                let cross_natural = match dir {
                    Direction::Horizontal => ih,
                    Direction::Vertical => iw,
                };

                let avail_cross = line.cross_max - match dir {
                    Direction::Horizontal => margin.vertical(),
                    Direction::Vertical => margin.horizontal(),
                };
                let (cross_size, cross_offset) = match align_self {
                    Align::Stretch => (avail_cross.max(0.0), 0.0),
                    Align::Start => (cross_natural, 0.0),
                    Align::Center => (cross_natural, ((avail_cross - cross_natural) / 2.0).max(0.0)),
                    Align::End => (cross_natural, (avail_cross - cross_natural).max(0.0)),
                };

                let child_rect = match dir {
                    Direction::Horizontal => Rect::new(
                        main_pos + margin.left,
                        cross_pos + margin.top + cross_offset,
                        basis,
                        cross_size,
                    ),
                    Direction::Vertical => Rect::new(
                        cross_pos + margin.left + cross_offset,
                        main_pos + margin.top,
                        cross_size,
                        basis,
                    ),
                };

                self.arrange(cid, child_rect);

                main_pos += basis + gap + extra_gap + match dir {
                    Direction::Horizontal => margin.horizontal(),
                    Direction::Vertical => margin.vertical(),
                };
            }

            cross_pos += line.cross_max + gap;
        }
    }
}

/// Info about a child needed for flex distribution.
struct ChildFlexInfo {
    id: NodeId,
    basis: f32,
    cross: f32,
    grow: f32,
    shrink: f32,
    margin: Spacing,
    align_self: Option<Align>,
}

/// Content area after padding.
struct ContentArea {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

// ── Child iterator ──

pub struct ChildIter<'a> {
    tree: &'a LayoutTree,
    next: Option<NodeId>,
}

impl<'a> Iterator for ChildIter<'a> {
    type Item = NodeId;
    fn next(&mut self) -> Option<NodeId> {
        let id = self.next?;
        self.next = self.tree.nodes[id.index()].next_sibling;
        Some(id)
    }
}

// ── Build context (used during frame) ──

/// Tracks the tree being built during the current frame.
pub struct TreeBuildContext {
    pub tree: LayoutTree,
    /// Stack of open container nodes.
    open_stack: Vec<NodeId>,
    /// Child counter per open container (for key derivation).
    child_counters: Vec<u32>,
    /// Override key for the next `add_leaf` or `open_container`.
    key_override: Option<u64>,
}

impl Default for TreeBuildContext {
    fn default() -> Self {
        Self {
            tree: LayoutTree::new(),
            open_stack: Vec::new(),
            child_counters: Vec::new(),
            key_override: None,
        }
    }
}

impl TreeBuildContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of currently open containers on the stack.
    pub fn open_stack_len(&self) -> usize {
        self.open_stack.len()
    }

    /// Set a key override consumed by the next `add_leaf` or `open_container`.
    pub fn push_key_scope(&mut self, key: u64) {
        self.key_override = Some(key);
    }

    /// Derive a key for an anonymous child of the current container.
    fn derive_key(&self, child_index: u32) -> u64 {
        match self.open_stack.last() {
            Some(&parent_id) => {
                let parent_key = self.tree.nodes[parent_id.index()].key;
                parent_key ^ (child_index as u64).wrapping_mul(0x517cc1b727220a95)
            }
            None => child_index as u64,
        }
    }

    /// Current child index in the innermost open container.
    fn current_child_index(&self) -> u32 {
        self.child_counters.last().copied().unwrap_or(0)
    }

    /// Increment the child counter.
    fn advance_child(&mut self) {
        if let Some(c) = self.child_counters.last_mut() {
            *c += 1;
        }
    }

    /// Open a container node. Returns the NodeId.
    pub fn open_container(&mut self, key: Option<u64>, style: LayoutStyle) -> NodeId {
        let child_idx = self.current_child_index();
        let key = self.key_override.take().or(key).unwrap_or_else(|| self.derive_key(child_idx));
        let id = self.tree.add_node(key);
        self.tree.node_mut(id).style = style;
        self.tree.node_mut(id).is_leaf = false;

        if let Some(&parent) = self.open_stack.last() {
            self.tree.append_child(parent, id);
        }
        self.advance_child();

        self.open_stack.push(id);
        self.child_counters.push(0);
        id
    }

    /// Close the current container.
    pub fn close_container(&mut self) {
        self.open_stack.pop();
        self.child_counters.pop();
    }

    /// Add a leaf node with intrinsic size.
    pub fn add_leaf(&mut self, key: Option<u64>, w: f32, h: f32) -> NodeId {
        let child_idx = self.current_child_index();
        let key = self.key_override.take().or(key).unwrap_or_else(|| self.derive_key(child_idx));
        let id = self.tree.add_node(key);
        self.tree.node_mut(id).intrinsic = (w, h);
        self.tree.node_mut(id).is_leaf = true;

        if let Some(&parent) = self.open_stack.last() {
            self.tree.append_child(parent, id);
        }
        self.advance_child();
        id
    }

    /// Set flex properties on an already-added node.
    pub fn set_flex(&mut self, id: NodeId, grow: f32, shrink: f32, basis: Option<f32>) {
        let node = self.tree.node_mut(id);
        node.style.flex_grow = grow;
        node.style.flex_shrink = shrink;
        node.style.flex_basis = basis;
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx(a: f32, b: f32) {
        assert!((a - b).abs() < 0.01, "expected {b}, got {a}");
    }

    fn assert_rect(r: Rect, x: f32, y: f32, w: f32, h: f32) {
        assert_approx(r.x, x);
        assert_approx(r.y, y);
        assert_approx(r.w, w);
        assert_approx(r.h, h);
    }

    // ── Basic vertical stacking ──

    #[test]
    fn vertical_stack() {
        let mut tree = LayoutTree::new();
        let root = tree.add_node(0);
        tree.node_mut(root).is_leaf = false;
        tree.node_mut(root).style.direction = Direction::Vertical;

        let a = tree.add_node(1);
        tree.node_mut(a).intrinsic = (100.0, 30.0);
        tree.append_child(root, a);

        let b = tree.add_node(2);
        tree.node_mut(b).intrinsic = (80.0, 40.0);
        tree.append_child(root, b);

        tree.solve(Rect::new(0.0, 0.0, 200.0, 200.0));

        // Root should be the viewport.
        assert_rect(tree.node(root).rect, 0.0, 0.0, 200.0, 200.0);
        // Children stacked vertically, each getting full width at Start align.
        assert_rect(tree.node(a).rect, 0.0, 0.0, 100.0, 30.0);
        assert_rect(tree.node(b).rect, 0.0, 30.0, 80.0, 40.0);
    }

    // ── Basic horizontal stacking ──

    #[test]
    fn horizontal_stack() {
        let mut tree = LayoutTree::new();
        let root = tree.add_node(0);
        tree.node_mut(root).is_leaf = false;
        tree.node_mut(root).style.direction = Direction::Horizontal;

        let a = tree.add_node(1);
        tree.node_mut(a).intrinsic = (60.0, 30.0);
        tree.append_child(root, a);

        let b = tree.add_node(2);
        tree.node_mut(b).intrinsic = (40.0, 50.0);
        tree.append_child(root, b);

        tree.solve(Rect::new(0.0, 0.0, 200.0, 100.0));

        assert_rect(tree.node(a).rect, 0.0, 0.0, 60.0, 30.0);
        assert_rect(tree.node(b).rect, 60.0, 0.0, 40.0, 50.0);
    }

    // ── Flex grow ──

    #[test]
    fn flex_grow() {
        let mut tree = LayoutTree::new();
        let root = tree.add_node(0);
        tree.node_mut(root).is_leaf = false;
        tree.node_mut(root).style.direction = Direction::Horizontal;

        let a = tree.add_node(1);
        tree.node_mut(a).intrinsic = (50.0, 30.0);
        tree.node_mut(a).style.flex_grow = 1.0;
        tree.append_child(root, a);

        let b = tree.add_node(2);
        tree.node_mut(b).intrinsic = (50.0, 30.0);
        tree.node_mut(b).style.flex_grow = 3.0;
        tree.append_child(root, b);

        tree.solve(Rect::new(0.0, 0.0, 200.0, 100.0));

        // 100px remaining, split 1:3 => a gets 25 extra, b gets 75 extra.
        assert_rect(tree.node(a).rect, 0.0, 0.0, 75.0, 30.0);
        assert_rect(tree.node(b).rect, 75.0, 0.0, 125.0, 30.0);
    }

    // ── Flex shrink ──

    #[test]
    fn flex_shrink() {
        let mut tree = LayoutTree::new();
        let root = tree.add_node(0);
        tree.node_mut(root).is_leaf = false;
        tree.node_mut(root).style.direction = Direction::Horizontal;

        // Two children that total 300 in 200 of space.
        let a = tree.add_node(1);
        tree.node_mut(a).intrinsic = (200.0, 30.0);
        tree.node_mut(a).style.flex_shrink = 1.0;
        tree.append_child(root, a);

        let b = tree.add_node(2);
        tree.node_mut(b).intrinsic = (100.0, 30.0);
        tree.node_mut(b).style.flex_shrink = 1.0;
        tree.append_child(root, b);

        tree.solve(Rect::new(0.0, 0.0, 200.0, 100.0));

        // Overflow = 100. Weighted shrink: a=200*1=200, b=100*1=100, total=300.
        // a shrinks by 100*200/300 = 66.67 => 133.33
        // b shrinks by 100*100/300 = 33.33 => 66.67
        assert_approx(tree.node(a).rect.w, 133.33);
        assert_approx(tree.node(b).rect.w, 66.67);
    }

    // ── Padding ──

    #[test]
    fn padding() {
        let mut tree = LayoutTree::new();
        let root = tree.add_node(0);
        tree.node_mut(root).is_leaf = false;
        tree.node_mut(root).style.direction = Direction::Vertical;
        tree.node_mut(root).style.padding = Spacing::all(10.0);

        let a = tree.add_node(1);
        tree.node_mut(a).intrinsic = (50.0, 20.0);
        tree.append_child(root, a);

        tree.solve(Rect::new(0.0, 0.0, 200.0, 200.0));

        // Child should be inset by padding.
        assert_rect(tree.node(a).rect, 10.0, 10.0, 50.0, 20.0);
    }

    // ── Gap ──

    #[test]
    fn gap() {
        let mut tree = LayoutTree::new();
        let root = tree.add_node(0);
        tree.node_mut(root).is_leaf = false;
        tree.node_mut(root).style.direction = Direction::Vertical;
        tree.node_mut(root).style.gap = 10.0;

        let a = tree.add_node(1);
        tree.node_mut(a).intrinsic = (50.0, 20.0);
        tree.append_child(root, a);

        let b = tree.add_node(2);
        tree.node_mut(b).intrinsic = (50.0, 20.0);
        tree.append_child(root, b);

        tree.solve(Rect::new(0.0, 0.0, 200.0, 200.0));

        assert_rect(tree.node(a).rect, 0.0, 0.0, 50.0, 20.0);
        assert_rect(tree.node(b).rect, 0.0, 30.0, 50.0, 20.0);
    }

    // ── Min/max constraints ──

    #[test]
    fn min_max_constraints() {
        let mut tree = LayoutTree::new();
        let root = tree.add_node(0);
        tree.node_mut(root).is_leaf = false;
        tree.node_mut(root).style.direction = Direction::Vertical;
        tree.node_mut(root).style.max_width = Some(150.0);
        tree.node_mut(root).style.min_height = Some(100.0);

        let a = tree.add_node(1);
        tree.node_mut(a).intrinsic = (50.0, 20.0);
        tree.append_child(root, a);

        tree.solve(Rect::new(0.0, 0.0, 200.0, 200.0));

        // Measure: intrinsic w=50 < max 150 ok, intrinsic h=20 < min 100 => 100.
        let root_intrinsic = tree.node(root).intrinsic;
        assert_approx(root_intrinsic.0, 50.0);
        assert_approx(root_intrinsic.1, 100.0);
    }

    // ── Stretch alignment ──

    #[test]
    fn align_stretch() {
        let mut tree = LayoutTree::new();
        let root = tree.add_node(0);
        tree.node_mut(root).is_leaf = false;
        tree.node_mut(root).style.direction = Direction::Horizontal;
        tree.node_mut(root).style.align_items = Align::Stretch;

        let a = tree.add_node(1);
        tree.node_mut(a).intrinsic = (60.0, 20.0);
        tree.append_child(root, a);

        tree.solve(Rect::new(0.0, 0.0, 200.0, 100.0));

        // Stretch: child height should fill cross axis (100).
        assert_rect(tree.node(a).rect, 0.0, 0.0, 60.0, 100.0);
    }

    // ── Center alignment ──

    #[test]
    fn align_center() {
        let mut tree = LayoutTree::new();
        let root = tree.add_node(0);
        tree.node_mut(root).is_leaf = false;
        tree.node_mut(root).style.direction = Direction::Horizontal;
        tree.node_mut(root).style.align_items = Align::Center;

        let a = tree.add_node(1);
        tree.node_mut(a).intrinsic = (60.0, 20.0);
        tree.append_child(root, a);

        tree.solve(Rect::new(0.0, 0.0, 200.0, 100.0));

        // Centered: y = (100 - 20) / 2 = 40.
        assert_rect(tree.node(a).rect, 0.0, 40.0, 60.0, 20.0);
    }

    // ── Justify space-between ──

    #[test]
    fn justify_space_between() {
        let mut tree = LayoutTree::new();
        let root = tree.add_node(0);
        tree.node_mut(root).is_leaf = false;
        tree.node_mut(root).style.direction = Direction::Horizontal;
        tree.node_mut(root).style.justify_content = Justify::SpaceBetween;

        let a = tree.add_node(1);
        tree.node_mut(a).intrinsic = (40.0, 20.0);
        tree.append_child(root, a);

        let b = tree.add_node(2);
        tree.node_mut(b).intrinsic = (40.0, 20.0);
        tree.append_child(root, b);

        tree.solve(Rect::new(0.0, 0.0, 200.0, 100.0));

        // Free space = 200 - 80 = 120. SpaceBetween with 2 items: gap = 120.
        assert_approx(tree.node(a).rect.x, 0.0);
        assert_approx(tree.node(b).rect.x, 160.0);
    }

    // ── Justify center ──

    #[test]
    fn justify_center() {
        let mut tree = LayoutTree::new();
        let root = tree.add_node(0);
        tree.node_mut(root).is_leaf = false;
        tree.node_mut(root).style.direction = Direction::Horizontal;
        tree.node_mut(root).style.justify_content = Justify::Center;

        let a = tree.add_node(1);
        tree.node_mut(a).intrinsic = (40.0, 20.0);
        tree.append_child(root, a);

        let b = tree.add_node(2);
        tree.node_mut(b).intrinsic = (40.0, 20.0);
        tree.append_child(root, b);

        tree.solve(Rect::new(0.0, 0.0, 200.0, 100.0));

        // Free = 120, center offset = 60.
        assert_approx(tree.node(a).rect.x, 60.0);
        assert_approx(tree.node(b).rect.x, 100.0);
    }

    // ── Nested containers ──

    #[test]
    fn nested_containers() {
        let mut tree = LayoutTree::new();

        // Root: vertical
        let root = tree.add_node(0);
        tree.node_mut(root).is_leaf = false;
        tree.node_mut(root).style.direction = Direction::Vertical;

        // Row inside root: horizontal
        let row = tree.add_node(10);
        tree.node_mut(row).is_leaf = false;
        tree.node_mut(row).style.direction = Direction::Horizontal;
        tree.append_child(root, row);

        let a = tree.add_node(11);
        tree.node_mut(a).intrinsic = (60.0, 25.0);
        tree.append_child(row, a);

        let b = tree.add_node(12);
        tree.node_mut(b).intrinsic = (40.0, 25.0);
        tree.append_child(row, b);

        // Another leaf directly in root.
        let c = tree.add_node(20);
        tree.node_mut(c).intrinsic = (100.0, 30.0);
        tree.append_child(root, c);

        tree.solve(Rect::new(0.0, 0.0, 200.0, 200.0));

        // Row intrinsic: w=100, h=25. Placed at (0,0).
        assert_rect(tree.node(row).rect, 0.0, 0.0, 100.0, 25.0);
        assert_rect(tree.node(a).rect, 0.0, 0.0, 60.0, 25.0);
        assert_rect(tree.node(b).rect, 60.0, 0.0, 40.0, 25.0);
        // c after row: y = 25.
        assert_rect(tree.node(c).rect, 0.0, 25.0, 100.0, 30.0);
    }

    // ── Wrap ──

    #[test]
    fn flex_wrap_basic() {
        let mut tree = LayoutTree::new();
        let root = tree.add_node(0);
        tree.node_mut(root).is_leaf = false;
        tree.node_mut(root).style.direction = Direction::Horizontal;
        tree.node_mut(root).style.flex_wrap = FlexWrap::Wrap;

        // 3 items of 80px in 200px => first 2 fit (160), third wraps.
        for i in 1..=3 {
            let c = tree.add_node(i);
            tree.node_mut(c).intrinsic = (80.0, 30.0);
            tree.append_child(root, c);
        }

        tree.solve(Rect::new(0.0, 0.0, 200.0, 200.0));

        let n1 = tree.node(NodeId(1));
        let n2 = tree.node(NodeId(2));
        let n3 = tree.node(NodeId(3));

        // Line 1: items 1, 2 at y=0.
        assert_approx(n1.rect.y, 0.0);
        assert_approx(n2.rect.y, 0.0);
        assert_approx(n2.rect.x, 80.0);
        // Line 2: item 3 at y=30.
        assert_approx(n3.rect.y, 30.0);
        assert_approx(n3.rect.x, 0.0);
    }

    // ── TreeBuildContext ──

    #[test]
    fn build_context_basic() {
        let mut ctx = TreeBuildContext::new();

        let root_style = LayoutStyle {
            direction: Direction::Vertical,
            ..Default::default()
        };
        ctx.open_container(Some(100), root_style);
        ctx.add_leaf(Some(101), 50.0, 20.0);
        ctx.add_leaf(Some(102), 60.0, 30.0);
        ctx.close_container();

        let tree = &ctx.tree;
        assert_eq!(tree.nodes.len(), 3);
        assert_eq!(tree.child_count(NodeId(0)), 2);
        assert_eq!(tree.node(NodeId(1)).intrinsic, (50.0, 20.0));
        assert_eq!(tree.node(NodeId(2)).intrinsic, (60.0, 30.0));
    }

    #[test]
    fn build_context_auto_keys() {
        let mut ctx = TreeBuildContext::new();
        ctx.open_container(Some(0), LayoutStyle::default());
        let a = ctx.add_leaf(None, 10.0, 10.0);
        let b = ctx.add_leaf(None, 10.0, 10.0);
        ctx.close_container();

        // Auto-derived keys should be different.
        assert_ne!(ctx.tree.node(a).key, ctx.tree.node(b).key);
    }

    // ── Flex basis override ──

    #[test]
    fn flex_basis() {
        let mut tree = LayoutTree::new();
        let root = tree.add_node(0);
        tree.node_mut(root).is_leaf = false;
        tree.node_mut(root).style.direction = Direction::Horizontal;

        let a = tree.add_node(1);
        tree.node_mut(a).intrinsic = (50.0, 30.0);
        tree.node_mut(a).style.flex_basis = Some(100.0);
        tree.append_child(root, a);

        let b = tree.add_node(2);
        tree.node_mut(b).intrinsic = (50.0, 30.0);
        tree.append_child(root, b);

        tree.solve(Rect::new(0.0, 0.0, 200.0, 100.0));

        // a uses basis 100 instead of intrinsic 50.
        assert_approx(tree.node(a).rect.w, 100.0);
        assert_approx(tree.node(b).rect.x, 100.0);
        assert_approx(tree.node(b).rect.w, 50.0);
    }

    // ── Margin ──

    #[test]
    fn margin() {
        let mut tree = LayoutTree::new();
        let root = tree.add_node(0);
        tree.node_mut(root).is_leaf = false;
        tree.node_mut(root).style.direction = Direction::Vertical;

        let a = tree.add_node(1);
        tree.node_mut(a).intrinsic = (50.0, 20.0);
        tree.node_mut(a).style.margin = Spacing { top: 5.0, right: 0.0, bottom: 5.0, left: 10.0 };
        tree.append_child(root, a);

        tree.solve(Rect::new(0.0, 0.0, 200.0, 200.0));

        assert_rect(tree.node(a).rect, 10.0, 5.0, 50.0, 20.0);
    }

    // ── Lookup ──

    #[test]
    fn lookup_by_key() {
        let mut tree = LayoutTree::new();
        let root = tree.add_node(0);
        tree.node_mut(root).is_leaf = false;
        tree.node_mut(root).style.direction = Direction::Vertical;

        let a = tree.add_node(42);
        tree.node_mut(a).intrinsic = (80.0, 25.0);
        tree.append_child(root, a);

        tree.solve(Rect::new(0.0, 0.0, 200.0, 200.0));

        let r = tree.lookup(42).unwrap();
        assert_approx(r.w, 80.0);
        assert_approx(r.h, 25.0);

        assert!(tree.lookup(999).is_none());
    }
}
