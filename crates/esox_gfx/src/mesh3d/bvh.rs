//! BVH (Bounding Volume Hierarchy) — spatial index for frustum culling acceleration.
//!
//! Flat-array BVH using median-split construction. Wins at >4K objects where
//! linear frustum scan becomes measurable.

use super::bounds::{Aabb, Containment, Frustum};

/// A BVH node — 32 bytes, cache-line friendly.
///
/// Internal nodes have `count == 0` and `offset_or_child` points to the left child.
/// Leaf nodes have `count > 0` and `offset_or_child` is the start index into `indices`.
#[allow(dead_code)]
pub(crate) struct BvhNode {
    pub aabb: Aabb,
    /// Leaf: index into `indices` array. Internal: index of left child in `nodes`.
    pub offset_or_child: u32,
    /// >0 = leaf node (number of objects). 0 = internal node.
    pub count: u16,
    /// Split axis (0=X, 1=Y, 2=Z). Only meaningful for internal nodes.
    pub axis: u8,
    pub _pad: u8,
}

/// Flat-array bounding volume hierarchy for broad-phase frustum culling.
pub struct Bvh {
    nodes: Vec<BvhNode>,
    /// Object indices — leaves reference contiguous ranges in this array.
    indices: Vec<u32>,
}

/// Maximum leaf size before splitting.
const LEAF_THRESHOLD: usize = 4;

/// Fixed-size traversal stack depth (enough for 2^64 objects in theory).
const STACK_DEPTH: usize = 64;

impl Bvh {
    /// Build a BVH from a set of AABBs.
    ///
    /// Returns a BVH that maps each input AABB index to its position.
    /// O(n log n) median-split construction.
    pub fn build(aabbs: &[Aabb]) -> Self {
        let n = aabbs.len();
        if n == 0 {
            return Self {
                nodes: Vec::new(),
                indices: Vec::new(),
            };
        }

        let mut indices: Vec<u32> = (0..n as u32).collect();
        let centroids: Vec<glam::Vec3> = aabbs.iter().map(|a| a.center()).collect();

        // Pre-allocate nodes (roughly 2n for a balanced tree)
        let mut nodes = Vec::with_capacity(2 * n);

        Self::build_recursive(aabbs, &centroids, &mut indices, &mut nodes, 0, n);

        Self { nodes, indices }
    }

    fn build_recursive(
        aabbs: &[Aabb],
        centroids: &[glam::Vec3],
        indices: &mut [u32],
        nodes: &mut Vec<BvhNode>,
        start: usize,
        end: usize,
    ) -> u32 {
        let node_idx = nodes.len() as u32;

        // Compute bounds of all objects in [start..end)
        let mut bounds = Aabb::EMPTY;
        for &idx in &indices[start..end] {
            bounds = bounds.union(aabbs[idx as usize]);
        }

        let count = end - start;

        if count <= LEAF_THRESHOLD {
            // Leaf node
            nodes.push(BvhNode {
                aabb: bounds,
                offset_or_child: start as u32,
                count: count as u16,
                axis: 0,
                _pad: 0,
            });
            return node_idx;
        }

        // Find longest axis of the centroid bounds
        let mut centroid_bounds = Aabb::EMPTY;
        for &idx in &indices[start..end] {
            let c = centroids[idx as usize];
            centroid_bounds.min = centroid_bounds.min.min(c);
            centroid_bounds.max = centroid_bounds.max.max(c);
        }
        let extent = centroid_bounds.max - centroid_bounds.min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };

        // Median split using select_nth_unstable_by (O(n) average)
        let mid = start + count / 2;
        indices[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
            let ca = match axis {
                0 => centroids[a as usize].x,
                1 => centroids[a as usize].y,
                _ => centroids[a as usize].z,
            };
            let cb = match axis {
                0 => centroids[b as usize].x,
                1 => centroids[b as usize].y,
                _ => centroids[b as usize].z,
            };
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Push placeholder for this internal node
        nodes.push(BvhNode {
            aabb: bounds,
            offset_or_child: 0, // will be filled
            count: 0,           // internal
            axis: axis as u8,
            _pad: 0,
        });

        // Recurse left and right
        let _left_idx = Self::build_recursive(aabbs, centroids, indices, nodes, start, mid);
        let right_idx = Self::build_recursive(aabbs, centroids, indices, nodes, mid, end);

        // The left child is at node_idx + 1 (pushed right after this node).
        // Store the right child index (left is implicit at node_idx + 1).
        // Actually, we store left child index in offset_or_child, but since
        // left is always node_idx+1, we store right_idx for the skip.
        // Convention: offset_or_child = left child index = node_idx + 1
        nodes[node_idx as usize].offset_or_child = right_idx;

        node_idx
    }

    /// Query the BVH for all objects visible in the given frustum.
    ///
    /// Pushes visible object indices into `out`. Uses a fixed-size stack (no allocation).
    pub fn query_frustum(&self, frustum: &Frustum, out: &mut Vec<u32>) {
        if self.nodes.is_empty() {
            return;
        }

        let mut stack = [0u32; STACK_DEPTH];
        let mut sp = 0;
        stack[0] = 0; // root
        sp += 1;

        while sp > 0 {
            sp -= 1;
            let node_idx = stack[sp] as usize;
            let node = &self.nodes[node_idx];

            let containment = frustum.test_aabb(&node.aabb);
            match containment {
                Containment::Outside => continue,
                Containment::Inside => {
                    // Entire subtree is visible — collect all descendants
                    self.collect_all(node_idx, out);
                }
                Containment::Intersect => {
                    if node.count > 0 {
                        // Leaf — add all objects (they need per-object test at caller level
                        // if desired, but for our use case the AABB is tight enough)
                        let start = node.offset_or_child as usize;
                        let end = start + node.count as usize;
                        for &idx in &self.indices[start..end] {
                            out.push(idx);
                        }
                    } else {
                        // Internal — push children
                        // Left child is at node_idx + 1
                        let left = node_idx as u32 + 1;
                        let right = node.offset_or_child;
                        if sp + 2 <= STACK_DEPTH {
                            stack[sp] = right;
                            sp += 1;
                            stack[sp] = left;
                            sp += 1;
                        }
                    }
                }
            }
        }
    }

    /// Collect all object indices in the subtree rooted at `node_idx`.
    fn collect_all(&self, node_idx: usize, out: &mut Vec<u32>) {
        let mut stack = [0u32; STACK_DEPTH];
        let mut sp = 0;
        stack[0] = node_idx as u32;
        sp += 1;

        while sp > 0 {
            sp -= 1;
            let idx = stack[sp] as usize;
            let node = &self.nodes[idx];

            if node.count > 0 {
                let start = node.offset_or_child as usize;
                let end = start + node.count as usize;
                for &obj_idx in &self.indices[start..end] {
                    out.push(obj_idx);
                }
            } else {
                let left = idx as u32 + 1;
                let right = node.offset_or_child;
                if sp + 2 <= STACK_DEPTH {
                    stack[sp] = right;
                    sp += 1;
                    stack[sp] = left;
                    sp += 1;
                }
            }
        }
    }

    /// Number of nodes in the BVH.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn aabb_at(center: Vec3, half: f32) -> Aabb {
        Aabb::new(center - Vec3::splat(half), center + Vec3::splat(half))
    }

    fn all_visible_frustum() -> Frustum {
        // Camera far away, looking at origin, huge frustum
        let view = glam::Mat4::look_at_rh(
            Vec3::new(0.0, 0.0, 1000.0),
            Vec3::ZERO,
            Vec3::Y,
        );
        let proj = glam::Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_2, // 90 degrees
            1.0,
            0.1,
            2000.0,
        );
        Frustum::from_view_projection(&(proj * view))
    }

    fn narrow_frustum() -> Frustum {
        // Camera at z=10 looking at origin, narrow FOV
        let view = glam::Mat4::look_at_rh(
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::ZERO,
            Vec3::Y,
        );
        let proj = glam::Mat4::perspective_rh(
            0.1, // very narrow FOV
            1.0,
            0.1,
            100.0,
        );
        Frustum::from_view_projection(&(proj * view))
    }

    #[test]
    fn build_empty() {
        let bvh = Bvh::build(&[]);
        assert_eq!(bvh.node_count(), 0);
    }

    #[test]
    fn build_single() {
        let aabbs = [aabb_at(Vec3::ZERO, 1.0)];
        let bvh = Bvh::build(&aabbs);
        assert!(bvh.node_count() > 0);
    }

    #[test]
    fn build_two() {
        let aabbs = [
            aabb_at(Vec3::new(-5.0, 0.0, 0.0), 1.0),
            aabb_at(Vec3::new(5.0, 0.0, 0.0), 1.0),
        ];
        let bvh = Bvh::build(&aabbs);
        assert!(bvh.node_count() > 0);
    }

    #[test]
    fn build_hundred() {
        let aabbs: Vec<Aabb> = (0..100)
            .map(|i| {
                let x = (i % 10) as f32 * 3.0;
                let z = (i / 10) as f32 * 3.0;
                aabb_at(Vec3::new(x, 0.0, z), 0.5)
            })
            .collect();
        let bvh = Bvh::build(&aabbs);
        assert!(bvh.node_count() > 0);
    }

    #[test]
    fn query_all_visible() {
        let aabbs: Vec<Aabb> = (0..20)
            .map(|i| {
                let x = (i % 5) as f32 * 2.0 - 4.0;
                let z = (i / 5) as f32 * 2.0 - 4.0;
                aabb_at(Vec3::new(x, 0.0, z), 0.5)
            })
            .collect();
        let bvh = Bvh::build(&aabbs);
        let frustum = all_visible_frustum();
        let mut visible = Vec::new();
        bvh.query_frustum(&frustum, &mut visible);
        visible.sort();
        visible.dedup();
        assert_eq!(visible.len(), 20);
    }

    #[test]
    fn query_all_culled() {
        // Objects at z = -500..
        let aabbs: Vec<Aabb> = (0..10)
            .map(|i| aabb_at(Vec3::new(0.0, 0.0, -500.0 - i as f32), 0.5))
            .collect();
        let bvh = Bvh::build(&aabbs);
        // Narrow frustum looking at origin from z=10 — objects at z=-500 are behind far plane
        let frustum = narrow_frustum();
        let mut visible = Vec::new();
        bvh.query_frustum(&frustum, &mut visible);
        assert_eq!(visible.len(), 0);
    }

    #[test]
    fn query_partial() {
        // Mix of visible and culled objects
        let mut aabbs = Vec::new();
        // 5 at origin (visible)
        for i in 0..5 {
            aabbs.push(aabb_at(Vec3::new(i as f32 * 0.5 - 1.0, 0.0, 0.0), 0.3));
        }
        // 5 far away (culled)
        for i in 0..5 {
            aabbs.push(aabb_at(Vec3::new(500.0 + i as f32, 0.0, 0.0), 0.3));
        }

        let bvh = Bvh::build(&aabbs);
        let frustum = narrow_frustum();
        let mut visible = Vec::new();
        bvh.query_frustum(&frustum, &mut visible);
        visible.sort();
        visible.dedup();

        // At least some should be visible, and not all
        assert!(!visible.is_empty());
        assert!(visible.len() < 10);
    }

    #[test]
    fn no_false_negatives() {
        // Every object that's individually visible must appear in BVH query results
        let aabbs: Vec<Aabb> = (0..50)
            .map(|i| {
                let x = (i % 10) as f32 - 5.0;
                let z = (i / 10) as f32 - 2.5;
                aabb_at(Vec3::new(x, 0.0, z), 0.4)
            })
            .collect();

        let bvh = Bvh::build(&aabbs);
        let frustum = all_visible_frustum();

        let mut bvh_visible = Vec::new();
        bvh.query_frustum(&frustum, &mut bvh_visible);
        bvh_visible.sort();
        bvh_visible.dedup();

        // Brute-force check
        let mut linear_visible = Vec::new();
        for (i, aabb) in aabbs.iter().enumerate() {
            if frustum.test_aabb_visible(aabb) {
                linear_visible.push(i as u32);
            }
        }

        // BVH must include all linearly-visible objects (no false negatives)
        for idx in &linear_visible {
            assert!(
                bvh_visible.contains(idx),
                "BVH missed visible object {idx}"
            );
        }
    }

    #[test]
    fn degenerate_aligned() {
        // All objects on the same X coordinate
        let aabbs: Vec<Aabb> = (0..10)
            .map(|i| aabb_at(Vec3::new(0.0, 0.0, i as f32), 0.2))
            .collect();
        let bvh = Bvh::build(&aabbs);
        let frustum = all_visible_frustum();
        let mut visible = Vec::new();
        bvh.query_frustum(&frustum, &mut visible);
        visible.sort();
        visible.dedup();
        assert_eq!(visible.len(), 10);
    }

    #[test]
    fn build_ten_thousand() {
        let aabbs: Vec<Aabb> = (0..10_000)
            .map(|i| {
                let x = (i % 100) as f32 * 2.0;
                let z = (i / 100) as f32 * 2.0;
                aabb_at(Vec3::new(x, 0.0, z), 0.5)
            })
            .collect();
        let bvh = Bvh::build(&aabbs);
        assert!(bvh.node_count() > 0);

        // Query with a frustum
        let frustum = all_visible_frustum();
        let mut visible = Vec::new();
        bvh.query_frustum(&frustum, &mut visible);
        visible.sort();
        visible.dedup();
        assert_eq!(visible.len(), 10_000);
    }
}
