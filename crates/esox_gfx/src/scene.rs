use crate::error::Error;
use crate::primitive::{Primitive, Rect};

/// Maximum number of nodes allowed in a single scene graph.
pub const MAX_NODES: usize = 100_000;

/// Maximum number of primitives in a single [`NodeContent::Batch`].
pub const MAX_BATCH_PRIMITIVES: usize = 100_000;

/// Handle into the scene's node arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// What a node contains.
#[derive(Debug, Clone)]
pub enum NodeContent {
    /// A grouping node with no visual of its own.
    Container,
    /// A single drawable primitive.
    Leaf(Primitive),
    /// A batch of primitives (e.g., all glyphs on a terminal line).
    Batch(Vec<Primitive>),
}

impl NodeContent {
    /// Create a batch node, truncating to [`MAX_BATCH_PRIMITIVES`] if over limit.
    pub fn batch(prims: Vec<Primitive>) -> Self {
        if prims.len() > MAX_BATCH_PRIMITIVES {
            tracing::warn!(
                count = prims.len(),
                max = MAX_BATCH_PRIMITIVES,
                "batch exceeds maximum primitives, truncating"
            );
            let mut prims = prims;
            prims.truncate(MAX_BATCH_PRIMITIVES);
            Self::Batch(prims)
        } else {
            Self::Batch(prims)
        }
    }
}

/// A node in the arena-based scene graph.
#[derive(Debug, Clone)]
pub struct Node {
    /// Parent node (None for roots).
    pub parent: Option<NodeId>,
    /// Ordered child node indices.
    pub children: Vec<NodeId>,
    /// Pixel offset relative to parent.
    pub offset: (f32, f32),
    /// Optional clip rectangle (in parent-relative coordinates).
    pub clip: Option<Rect>,
    /// Drawing order (higher = on top).
    pub z_order: i32,
    /// What this node draws.
    pub content: NodeContent,
    /// Whether this node needs re-rendering.
    pub dirty: bool,
    /// Opacity multiplier (0.0–1.0).
    pub opacity: f32,
}

/// A primitive resolved to absolute coordinates for rendering.
#[derive(Debug, Clone)]
pub struct ResolvedPrimitive {
    /// The primitive with positions in absolute (window) coordinates.
    pub primitive: Primitive,
    /// The clip rectangle in absolute coordinates, if any.
    pub clip: Option<Rect>,
    /// Combined opacity from all ancestors.
    pub opacity: f32,
    /// Z-order for sorting (higher = on top).
    pub z_order: i32,
}

/// Arena-based scene graph. Nodes are stored in a flat `Vec` for cache locality.
pub struct Scene {
    nodes: Vec<Option<Node>>,
    free_list: Vec<usize>,
}

impl Scene {
    /// Create an empty scene.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            free_list: Vec::new(),
        }
    }

    /// Insert a new node into the scene, returning its ID.
    ///
    /// Returns [`Error::SceneGraphFull`] if the scene already contains
    /// [`MAX_NODES`] live nodes.
    pub fn insert(&mut self, node: Node) -> Result<NodeId, Error> {
        if self.node_count() >= MAX_NODES {
            return Err(Error::SceneGraphFull(MAX_NODES));
        }
        if let Some(idx) = self.free_list.pop() {
            self.nodes[idx] = Some(node);
            Ok(NodeId(idx))
        } else {
            let idx = self.nodes.len();
            self.nodes.push(Some(node));
            Ok(NodeId(idx))
        }
    }

    /// Number of live (non-removed) nodes in the scene.
    pub fn node_count(&self) -> usize {
        self.nodes.len() - self.free_list.len()
    }

    /// Remove a node and all its children from the scene.
    pub fn remove(&mut self, id: NodeId) {
        // Iterative depth-first removal to avoid stack overflow on deep trees.
        let mut stack = vec![id];
        while let Some(nid) = stack.pop() {
            if let Some(node) = self.nodes[nid.0].take() {
                stack.extend(&node.children);
                // Remove from parent's child list (only for the root of the removal).
                if let Some(parent_id) = node.parent
                    && let Some(Some(parent)) = self.nodes.get_mut(parent_id.0)
                {
                    parent.children.retain(|c| *c != nid);
                }
                self.free_list.push(nid.0);
            }
        }
    }

    /// Get an immutable reference to a node.
    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.0).and_then(|n| n.as_ref())
    }

    /// Get a mutable reference to a node.
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(id.0).and_then(|n| n.as_mut())
    }

    /// Mark a node (and its ancestors) as dirty.
    pub fn mark_dirty(&mut self, id: NodeId) {
        let mut current = Some(id);
        while let Some(nid) = current {
            if let Some(Some(node)) = self.nodes.get_mut(nid.0) {
                if node.dirty {
                    break; // Already dirty up the chain.
                }
                node.dirty = true;
                current = node.parent;
            } else {
                break;
            }
        }
    }

    /// Collect all primitives in the scene, resolved to absolute coordinates.
    ///
    /// Reuses the provided buffer to avoid per-frame allocation. The buffer is
    /// cleared and filled with the resolved primitives, then sorted by z-order.
    pub fn collect_primitives_into(&self, out: &mut Vec<ResolvedPrimitive>) {
        out.clear();

        // Iterative depth-first traversal to avoid stack overflow on deep trees.
        // Each work item carries the inherited absolute offset, clip, opacity, and z_order.
        let mut stack: Vec<(NodeId, f32, f32, Option<Rect>, f32, i32)> = Vec::new();

        // Seed with root nodes (no parent).
        for (idx, slot) in self.nodes.iter().enumerate() {
            if let Some(node) = slot
                && node.parent.is_none()
            {
                stack.push((NodeId(idx), 0.0, 0.0, None, 1.0, 0));
            }
        }

        while let Some((id, parent_abs_x, parent_abs_y, parent_clip, parent_opacity, parent_z_order)) = stack.pop() {
            let Some(node) = self.get(id) else { continue };

            let abs_x = parent_abs_x + node.offset.0;
            let abs_y = parent_abs_y + node.offset.1;
            let opacity = parent_opacity * node.opacity;
            let z_order = parent_z_order + node.z_order;

            // Compute effective clip.
            let clip = match (parent_clip, node.clip) {
                (None, None) => None,
                (Some(c), None) => Some(c),
                (None, Some(local)) => {
                    Some(Rect {
                        x: abs_x + local.x,
                        y: abs_y + local.y,
                        width: local.width,
                        height: local.height,
                    })
                }
                (Some(parent), Some(local)) => {
                    let abs_local = Rect {
                        x: abs_x + local.x,
                        y: abs_y + local.y,
                        width: local.width,
                        height: local.height,
                    };
                    rect_intersect(parent, abs_local)
                }
            };

            // Emit primitives from this node.
            match &node.content {
                NodeContent::Container => {}
                NodeContent::Leaf(p) => {
                    out.push(ResolvedPrimitive {
                        primitive: offset_primitive(p, abs_x, abs_y),
                        clip,
                        opacity,
                        z_order,
                    });
                }
                NodeContent::Batch(prims) => {
                    for p in prims {
                        out.push(ResolvedPrimitive {
                            primitive: offset_primitive(p, abs_x, abs_y),
                            clip,
                            opacity,
                            z_order,
                        });
                    }
                }
            }

            // Push children onto the stack (reverse order to preserve left-to-right traversal).
            let num_children = node.children.len();
            for ci in (0..num_children).rev() {
                let child_id = self.nodes[id.0].as_ref().map(|n| n.children[ci]);
                if let Some(child) = child_id {
                    stack.push((child, abs_x, abs_y, clip, opacity, z_order));
                }
            }
        }

        out.sort_by_key(|rp| rp.z_order);
    }

    /// Collect all primitives in the scene, resolved to absolute coordinates.
    ///
    /// Allocates a fresh buffer. Prefer [`collect_primitives_into`] in hot
    /// paths to reuse an existing buffer across frames.
    pub fn collect_primitives(&self) -> Vec<ResolvedPrimitive> {
        let mut out = Vec::new();
        self.collect_primitives_into(&mut out);
        out
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the intersection of two rectangles, returning `None` if disjoint.
fn rect_intersect(a: Rect, b: Rect) -> Option<Rect> {
    let x = a.x.max(b.x);
    let y = a.y.max(b.y);
    let right = (a.x + a.width).min(b.x + b.width);
    let bottom = (a.y + a.height).min(b.y + b.height);
    if right > x && bottom > y {
        Some(Rect {
            x,
            y,
            width: right - x,
            height: bottom - y,
        })
    } else {
        None
    }
}

/// Offset a primitive's position by an absolute amount.
#[cfg(test)]
fn make_leaf(offset: (f32, f32), parent: Option<NodeId>) -> Node {
    Node {
        parent,
        children: Vec::new(),
        offset,
        clip: None,
        z_order: 0,
        content: NodeContent::Leaf(Primitive::SolidRect {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                width: 10.0,
                height: 10.0,
            },
            color: crate::Color::WHITE,
            border_radius: crate::primitive::BorderRadius::ZERO,
        }),
        dirty: false,
        opacity: 1.0,
    }
}

#[cfg(test)]
fn make_container(offset: (f32, f32), parent: Option<NodeId>) -> Node {
    Node {
        parent,
        children: Vec::new(),
        offset,
        clip: None,
        z_order: 0,
        content: NodeContent::Container,
        dirty: false,
        opacity: 1.0,
    }
}

fn offset_primitive(p: &Primitive, dx: f32, dy: f32) -> Primitive {
    match *p {
        Primitive::SolidRect {
            rect,
            color,
            border_radius,
        } => Primitive::SolidRect {
            rect: Rect {
                x: rect.x + dx,
                y: rect.y + dy,
                ..rect
            },
            color,
            border_radius,
        },
        Primitive::TexturedRect {
            rect,
            uv,
            color,
            layer,
        } => Primitive::TexturedRect {
            rect: Rect {
                x: rect.x + dx,
                y: rect.y + dy,
                ..rect
            },
            uv,
            color,
            layer,
        },
        Primitive::ShaderRect {
            rect,
            shader,
            params,
        } => Primitive::ShaderRect {
            rect: Rect {
                x: rect.x + dx,
                y: rect.y + dy,
                ..rect
            },
            shader,
            params,
        },
        Primitive::Circle {
            center_x,
            center_y,
            radius,
            color,
        } => Primitive::Circle {
            center_x: center_x + dx,
            center_y: center_y + dy,
            radius,
            color,
        },
        Primitive::Ellipse {
            center_x,
            center_y,
            rx,
            ry,
            color,
        } => Primitive::Ellipse {
            center_x: center_x + dx,
            center_y: center_y + dy,
            rx,
            ry,
            color,
        },
        Primitive::Ring {
            center_x,
            center_y,
            outer_r,
            inner_r,
            color,
        } => Primitive::Ring {
            center_x: center_x + dx,
            center_y: center_y + dy,
            outer_r,
            inner_r,
            color,
        },
        Primitive::Line {
            x1,
            y1,
            x2,
            y2,
            thickness,
            color,
        } => Primitive::Line {
            x1: x1 + dx,
            y1: y1 + dy,
            x2: x2 + dx,
            y2: y2 + dy,
            thickness,
            color,
        },
        Primitive::Arc {
            center_x,
            center_y,
            radius,
            thickness,
            angle_start,
            angle_sweep,
            color,
        } => Primitive::Arc {
            center_x: center_x + dx,
            center_y: center_y + dy,
            radius,
            thickness,
            angle_start,
            angle_sweep,
            color,
        },
        Primitive::Triangle { rect, color } => Primitive::Triangle {
            rect: Rect {
                x: rect.x + dx,
                y: rect.y + dy,
                ..rect
            },
            color,
        },
        Primitive::Polygon {
            center_x,
            center_y,
            radius,
            sides,
            color,
        } => Primitive::Polygon {
            center_x: center_x + dx,
            center_y: center_y + dy,
            radius,
            sides,
            color,
        },
        Primitive::Star {
            center_x,
            center_y,
            points,
            inner_r,
            outer_r,
            color,
        } => Primitive::Star {
            center_x: center_x + dx,
            center_y: center_y + dy,
            points,
            inner_r,
            outer_r,
            color,
        },
        Primitive::Sector {
            center_x,
            center_y,
            radius,
            angle_start,
            angle_sweep,
            color,
        } => Primitive::Sector {
            center_x: center_x + dx,
            center_y: center_y + dy,
            radius,
            angle_start,
            angle_sweep,
            color,
        },
        Primitive::Capsule { rect, color } => Primitive::Capsule {
            rect: Rect {
                x: rect.x + dx,
                y: rect.y + dy,
                ..rect
            },
            color,
        },
        Primitive::CrossShape {
            center_x,
            center_y,
            arm_width,
            arm_length,
            color,
        } => Primitive::CrossShape {
            center_x: center_x + dx,
            center_y: center_y + dy,
            arm_width,
            arm_length,
            color,
        },
        Primitive::Bezier {
            x0,
            y0,
            cx,
            cy,
            x1,
            y1,
            thickness,
            color,
        } => Primitive::Bezier {
            x0: x0 + dx,
            y0: y0 + dy,
            cx: cx + dx,
            cy: cy + dy,
            x1: x1 + dx,
            y1: y1 + dy,
            thickness,
            color,
        },
        Primitive::ArbitraryTriangle {
            x1,
            y1,
            x2,
            y2,
            x3,
            y3,
            color,
        } => Primitive::ArbitraryTriangle {
            x1: x1 + dx,
            y1: y1 + dy,
            x2: x2 + dx,
            y2: y2 + dy,
            x3: x3 + dx,
            y3: y3 + dy,
            color,
        },
        Primitive::Trapezoid {
            center_x,
            center_y,
            top_half_w,
            bottom_half_w,
            half_h,
            color,
        } => Primitive::Trapezoid {
            center_x: center_x + dx,
            center_y: center_y + dy,
            top_half_w,
            bottom_half_w,
            half_h,
            color,
        },
        Primitive::Heart {
            center_x,
            center_y,
            scale,
            color,
        } => Primitive::Heart {
            center_x: center_x + dx,
            center_y: center_y + dy,
            scale,
            color,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prim_rect(p: &Primitive) -> &Rect {
        match p {
            Primitive::SolidRect { rect, .. }
            | Primitive::TexturedRect { rect, .. }
            | Primitive::ShaderRect { rect, .. }
            | Primitive::Triangle { rect, .. } => rect,
            _ => panic!("primitive has no rect field"),
        }
    }

    // ── Insert / Get / Remove ──

    #[test]
    fn insert_and_get() {
        let mut scene = Scene::new();
        let id = scene.insert(make_leaf((0.0, 0.0), None)).unwrap();
        assert!(scene.get(id).is_some());
    }

    #[test]
    fn remove_makes_node_inaccessible() {
        let mut scene = Scene::new();
        let id = scene.insert(make_leaf((0.0, 0.0), None)).unwrap();
        scene.remove(id);
        assert!(scene.get(id).is_none());
    }

    #[test]
    fn remove_recycles_slot() {
        let mut scene = Scene::new();
        let id1 = scene.insert(make_leaf((0.0, 0.0), None)).unwrap();
        scene.remove(id1);
        let id2 = scene.insert(make_leaf((0.0, 0.0), None)).unwrap();
        // Should reuse the freed slot.
        assert_eq!(id1, id2);
    }

    #[test]
    fn remove_cascades_to_children() {
        let mut scene = Scene::new();
        let parent_id = scene.insert(make_container((0.0, 0.0), None)).unwrap();
        let mut child = make_leaf((5.0, 5.0), Some(parent_id));
        child.parent = Some(parent_id);
        let child_id = scene.insert(child).unwrap();
        scene.get_mut(parent_id).unwrap().children.push(child_id);

        scene.remove(parent_id);
        assert!(scene.get(parent_id).is_none());
        assert!(scene.get(child_id).is_none());
    }

    #[test]
    fn remove_child_updates_parent() {
        let mut scene = Scene::new();
        let parent_id = scene.insert(make_container((0.0, 0.0), None)).unwrap();
        let child = make_leaf((5.0, 5.0), Some(parent_id));
        let child_id = scene.insert(child).unwrap();
        scene.get_mut(parent_id).unwrap().children.push(child_id);

        scene.remove(child_id);
        assert!(scene.get(parent_id).unwrap().children.is_empty());
    }

    // ── Dirty propagation ──

    #[test]
    fn mark_dirty_propagates_to_ancestors() {
        let mut scene = Scene::new();
        let root = scene.insert(make_container((0.0, 0.0), None)).unwrap();
        let child = scene.insert(make_leaf((0.0, 0.0), Some(root))).unwrap();
        scene.get_mut(root).unwrap().children.push(child);

        scene.mark_dirty(child);
        assert!(scene.get(child).unwrap().dirty);
        assert!(scene.get(root).unwrap().dirty);
    }

    #[test]
    fn mark_dirty_stops_at_already_dirty() {
        let mut scene = Scene::new();
        let root = scene.insert(make_container((0.0, 0.0), None)).unwrap();
        let mid = scene
            .insert(make_container((0.0, 0.0), Some(root)))
            .unwrap();
        scene.get_mut(root).unwrap().children.push(mid);
        let leaf = scene.insert(make_leaf((0.0, 0.0), Some(mid))).unwrap();
        scene.get_mut(mid).unwrap().children.push(leaf);

        // Mark mid dirty first.
        scene.mark_dirty(mid);
        // Reset root's dirty flag to test short-circuiting.
        scene.get_mut(root).unwrap().dirty = false;

        // Now mark the leaf dirty — should stop at mid (already dirty).
        scene.mark_dirty(leaf);
        assert!(scene.get(leaf).unwrap().dirty);
        assert!(scene.get(mid).unwrap().dirty);
        assert!(!scene.get(root).unwrap().dirty);
    }

    // ── Primitive collection ──

    #[test]
    fn collect_empty_scene() {
        let scene = Scene::new();
        assert!(scene.collect_primitives().is_empty());
    }

    #[test]
    fn collect_single_leaf() {
        let mut scene = Scene::new();
        scene.insert(make_leaf((10.0, 20.0), None)).unwrap();
        let prims = scene.collect_primitives();
        assert_eq!(prims.len(), 1);
        let rect = prim_rect(&prims[0].primitive);
        assert_eq!(rect.x, 10.0);
        assert_eq!(rect.y, 20.0);
    }

    #[test]
    fn collect_nested_offsets_accumulate() {
        let mut scene = Scene::new();
        let root = scene.insert(make_container((100.0, 200.0), None)).unwrap();
        let leaf = scene.insert(make_leaf((10.0, 20.0), Some(root))).unwrap();
        scene.get_mut(root).unwrap().children.push(leaf);

        let prims = scene.collect_primitives();
        assert_eq!(prims.len(), 1);
        let rect = prim_rect(&prims[0].primitive);
        assert_eq!(rect.x, 110.0);
        assert_eq!(rect.y, 220.0);
    }

    #[test]
    fn collect_opacity_multiplies() {
        let mut scene = Scene::new();
        let mut root = make_container((0.0, 0.0), None);
        root.opacity = 0.5;
        let root_id = scene.insert(root).unwrap();
        let mut leaf = make_leaf((0.0, 0.0), Some(root_id));
        leaf.opacity = 0.5;
        let leaf_id = scene.insert(leaf).unwrap();
        scene.get_mut(root_id).unwrap().children.push(leaf_id);

        let prims = scene.collect_primitives();
        assert_eq!(prims.len(), 1);
        assert!((prims[0].opacity - 0.25).abs() < 1e-5);
    }

    #[test]
    fn collect_batch_emits_multiple() {
        let mut scene = Scene::new();
        let p = Primitive::SolidRect {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                width: 5.0,
                height: 5.0,
            },
            color: crate::Color::WHITE,
            border_radius: crate::primitive::BorderRadius::ZERO,
        };
        let node = Node {
            parent: None,
            children: Vec::new(),
            offset: (0.0, 0.0),
            clip: None,
            z_order: 0,
            content: NodeContent::batch(vec![p, p, p]),
            dirty: false,
            opacity: 1.0,
        };
        scene.insert(node).unwrap();
        assert_eq!(scene.collect_primitives().len(), 3);
    }

    #[test]
    fn collect_container_emits_nothing() {
        let mut scene = Scene::new();
        scene.insert(make_container((0.0, 0.0), None)).unwrap();
        assert!(scene.collect_primitives().is_empty());
    }

    // ── z_order sort ──

    #[test]
    fn collect_sorts_by_z_order() {
        let mut scene = Scene::new();
        let mut high = make_leaf((0.0, 0.0), None);
        high.z_order = 10;
        let mut low = make_leaf((1.0, 0.0), None);
        low.z_order = 1;
        let mut mid = make_leaf((2.0, 0.0), None);
        mid.z_order = 5;

        scene.insert(high).unwrap();
        scene.insert(low).unwrap();
        scene.insert(mid).unwrap();

        let prims = scene.collect_primitives();
        assert_eq!(prims.len(), 3);
        assert_eq!(prims[0].z_order, 1);
        assert_eq!(prims[1].z_order, 5);
        assert_eq!(prims[2].z_order, 10);
    }

    // ── clip intersection ──

    #[test]
    fn clip_intersection_narrows_rect() {
        let mut scene = Scene::new();
        let mut parent = make_container((0.0, 0.0), None);
        parent.clip = Some(Rect {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
        });
        let parent_id = scene.insert(parent).unwrap();

        let mut child = make_leaf((0.0, 0.0), Some(parent_id));
        child.clip = Some(Rect {
            x: 50.0,
            y: 50.0,
            width: 100.0,
            height: 100.0,
        });
        let child_id = scene.insert(child).unwrap();
        scene.get_mut(parent_id).unwrap().children.push(child_id);

        let prims = scene.collect_primitives();
        assert_eq!(prims.len(), 1);
        let clip = prims[0].clip.unwrap();
        assert_eq!(clip.x, 50.0);
        assert_eq!(clip.y, 50.0);
        assert_eq!(clip.width, 50.0);
        assert_eq!(clip.height, 50.0);
    }

    #[test]
    fn clip_intersection_disjoint_removes_clip() {
        let mut scene = Scene::new();
        let mut parent = make_container((0.0, 0.0), None);
        parent.clip = Some(Rect {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
        });
        let parent_id = scene.insert(parent).unwrap();

        let mut child = make_leaf((0.0, 0.0), Some(parent_id));
        child.clip = Some(Rect {
            x: 100.0,
            y: 100.0,
            width: 10.0,
            height: 10.0,
        });
        let child_id = scene.insert(child).unwrap();
        scene.get_mut(parent_id).unwrap().children.push(child_id);

        let prims = scene.collect_primitives();
        assert_eq!(prims.len(), 1);
        assert!(prims[0].clip.is_none());
    }

    // ── offset_primitive ──

    #[test]
    fn offset_solid_rect() {
        let p = Primitive::SolidRect {
            rect: Rect {
                x: 5.0,
                y: 10.0,
                width: 20.0,
                height: 30.0,
            },
            color: crate::Color::WHITE,
            border_radius: crate::primitive::BorderRadius::ZERO,
        };
        let shifted = offset_primitive(&p, 100.0, 200.0);
        let rect = prim_rect(&shifted);
        assert_eq!(rect.x, 105.0);
        assert_eq!(rect.y, 210.0);
        assert_eq!(rect.width, 20.0);
        assert_eq!(rect.height, 30.0);
    }

    #[test]
    fn offset_circle() {
        let p = Primitive::Circle {
            center_x: 10.0,
            center_y: 20.0,
            radius: 5.0,
            color: crate::Color::WHITE,
        };
        let shifted = offset_primitive(&p, 100.0, 200.0);
        match shifted {
            Primitive::Circle {
                center_x,
                center_y,
                radius,
                ..
            } => {
                assert_eq!(center_x, 110.0);
                assert_eq!(center_y, 220.0);
                assert_eq!(radius, 5.0);
            }
            _ => panic!("expected Circle"),
        }
    }

    #[test]
    fn offset_line() {
        let p = Primitive::Line {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
            thickness: 2.0,
            color: crate::Color::WHITE,
        };
        let shifted = offset_primitive(&p, 5.0, 5.0);
        match shifted {
            Primitive::Line { x1, y1, x2, y2, .. } => {
                assert_eq!(x1, 5.0);
                assert_eq!(y1, 5.0);
                assert_eq!(x2, 15.0);
                assert_eq!(y2, 15.0);
            }
            _ => panic!("expected Line"),
        }
    }

    #[test]
    fn remove_already_removed_is_noop() {
        let mut scene = Scene::new();
        let id = scene.insert(make_leaf((0.0, 0.0), None)).unwrap();
        scene.remove(id);
        // Second remove should not panic.
        scene.remove(id);
        assert!(scene.get(id).is_none());
    }

    #[test]
    fn deep_nesting_accumulates_correctly() {
        let mut scene = Scene::new();
        let mut parent_id = scene.insert(make_container((10.0, 10.0), None)).unwrap();
        for _ in 1..10 {
            let mut child = make_container((10.0, 10.0), Some(parent_id));
            child.opacity = 0.9;
            let child_id = scene.insert(child).unwrap();
            scene.get_mut(parent_id).unwrap().children.push(child_id);
            parent_id = child_id;
        }
        // Add a leaf at the deepest level.
        let leaf = make_leaf((10.0, 10.0), Some(parent_id));
        let leaf_id = scene.insert(leaf).unwrap();
        scene.get_mut(parent_id).unwrap().children.push(leaf_id);

        let prims = scene.collect_primitives();
        assert_eq!(prims.len(), 1);
        let rect = prim_rect(&prims[0].primitive);
        // 10 containers + 1 leaf, each with offset (10, 10) = 110, 110
        assert_eq!(rect.x, 110.0);
        assert_eq!(rect.y, 110.0);
        // Opacity: 1.0 * 0.9^9 * 1.0
        let expected_opacity = 0.9_f32.powi(9);
        assert!((prims[0].opacity - expected_opacity).abs() < 1e-5);
    }

    #[test]
    fn empty_batch_emits_nothing() {
        let mut scene = Scene::new();
        let node = Node {
            parent: None,
            children: Vec::new(),
            offset: (0.0, 0.0),
            clip: None,
            z_order: 0,
            content: NodeContent::batch(vec![]),
            dirty: false,
            opacity: 1.0,
        };
        scene.insert(node).unwrap();
        assert!(scene.collect_primitives().is_empty());
    }

    // ── Security cap tests ──

    #[test]
    fn scene_insert_at_max_nodes_returns_error() {
        let mut scene = Scene::new();
        for i in 0..MAX_NODES {
            scene
                .insert(make_leaf((0.0, 0.0), None))
                .unwrap_or_else(|_| {
                    panic!("insert should succeed at index {i}");
                });
        }
        assert_eq!(scene.node_count(), MAX_NODES);
        let result = scene.insert(make_leaf((0.0, 0.0), None));
        assert!(
            matches!(result, Err(Error::SceneGraphFull(_))),
            "expected SceneGraphFull error, got: {result:?}"
        );
    }

    #[test]
    fn batch_oversized_truncates() {
        let p = Primitive::SolidRect {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
            color: crate::Color::WHITE,
            border_radius: crate::primitive::BorderRadius::ZERO,
        };
        let batch = NodeContent::batch(vec![p; MAX_BATCH_PRIMITIVES + 1000]);
        match batch {
            NodeContent::Batch(prims) => assert_eq!(prims.len(), MAX_BATCH_PRIMITIVES),
            _ => panic!("expected Batch"),
        }
    }

    #[test]
    fn batch_at_exact_limit_ok() {
        let p = Primitive::SolidRect {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
            color: crate::Color::WHITE,
            border_radius: crate::primitive::BorderRadius::ZERO,
        };
        let batch = NodeContent::batch(vec![p; MAX_BATCH_PRIMITIVES]);
        match batch {
            NodeContent::Batch(prims) => assert_eq!(prims.len(), MAX_BATCH_PRIMITIVES),
            _ => panic!("expected Batch"),
        }
    }

    #[test]
    fn offset_arc() {
        let p = Primitive::Arc {
            center_x: 10.0,
            center_y: 20.0,
            radius: 30.0,
            thickness: 4.0,
            angle_start: 0.0,
            angle_sweep: 1.5,
            color: crate::Color::WHITE,
        };
        let shifted = offset_primitive(&p, 100.0, 200.0);
        match shifted {
            Primitive::Arc {
                center_x,
                center_y,
                radius,
                thickness,
                angle_start,
                angle_sweep,
                ..
            } => {
                assert_eq!(center_x, 110.0);
                assert_eq!(center_y, 220.0);
                assert_eq!(radius, 30.0);
                assert_eq!(thickness, 4.0);
                assert_eq!(angle_start, 0.0);
                assert_eq!(angle_sweep, 1.5);
            }
            _ => panic!("expected Arc"),
        }
    }
}
