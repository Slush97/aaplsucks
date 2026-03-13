//! Bounding volumes — AABB and frustum for culling.
//!
//! Pure math module with no wgpu imports. Fully unit-testable.

use glam::{Mat4, Vec3, Vec4};

// ── AABB ──

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// Identity element for union: min=MAX, max=MIN so any union produces the other operand.
    pub const EMPTY: Self = Self {
        min: Vec3::MAX,
        max: Vec3::MIN,
    };

    /// Construct from min/max corners.
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Compute the AABB enclosing a set of points.
    ///
    /// Returns `Aabb::EMPTY` for an empty slice.
    pub fn from_points(positions: &[Vec3]) -> Self {
        let mut aabb = Self::EMPTY;
        for &p in positions {
            aabb.min = aabb.min.min(p);
            aabb.max = aabb.max.max(p);
        }
        aabb
    }

    /// Center of the box.
    pub fn center(self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Half-extents (distance from center to each face).
    pub fn half_extents(self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Surface area (for SAH heuristic in BVH).
    pub fn surface_area(self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Union of two AABBs (smallest AABB enclosing both).
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Check if a point is inside the box (inclusive).
    pub fn contains_point(self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    /// Test whether this AABB overlaps another AABB.
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Transform the AABB by a 4x4 matrix using Arvo's fast method.
    ///
    /// Transforms center, then computes new half-extents from the absolute values
    /// of the matrix columns. O(1) branchless — no 8-corner expansion.
    pub fn transformed(self, m: &Mat4) -> Aabb {
        let center = self.center();
        let half = self.half_extents();

        let cols = m.to_cols_array_2d();

        // New center = M * center (affine)
        let new_center = Vec3::new(
            cols[0][0] * center.x + cols[1][0] * center.y + cols[2][0] * center.z + cols[3][0],
            cols[0][1] * center.x + cols[1][1] * center.y + cols[2][1] * center.z + cols[3][1],
            cols[0][2] * center.x + cols[1][2] * center.y + cols[2][2] * center.z + cols[3][2],
        );

        // New half-extents from absolute upper-left 3x3
        let new_half = Vec3::new(
            cols[0][0].abs() * half.x + cols[1][0].abs() * half.y + cols[2][0].abs() * half.z,
            cols[0][1].abs() * half.x + cols[1][1].abs() * half.y + cols[2][1].abs() * half.z,
            cols[0][2].abs() * half.x + cols[1][2].abs() * half.y + cols[2][2].abs() * half.z,
        );

        Aabb {
            min: new_center - new_half,
            max: new_center + new_half,
        }
    }
}

// ── Frustum ──

/// Result of testing an AABB against a frustum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Containment {
    /// Entirely outside — can be culled.
    Outside,
    /// Partially inside — must be rendered but children may be culled.
    Intersect,
    /// Entirely inside — render without further testing.
    Inside,
}

/// View frustum defined by 6 inward-pointing half-planes.
///
/// Plane order: left, right, bottom, top, near, far.
/// Each `Vec4(a, b, c, d)` represents the half-space `ax + by + cz + d >= 0`.
pub struct Frustum {
    pub planes: [Vec4; 6],
}

impl Frustum {
    /// Extract frustum planes from a combined view-projection matrix.
    ///
    /// Uses the Gribb-Hartmann method. wgpu uses depth [0,1], so the near plane
    /// is row3 (not row3+row2 as in OpenGL's [-1,1] depth).
    pub fn from_view_projection(vp: &Mat4) -> Self {
        let rows = vp.transpose().to_cols_array_2d();
        let row0 = Vec4::from(rows[0]);
        let row1 = Vec4::from(rows[1]);
        let row2 = Vec4::from(rows[2]);
        let row3 = Vec4::from(rows[3]);

        let mut planes = [
            row3 + row0, // left
            row3 - row0, // right
            row3 + row1, // bottom
            row3 - row1, // top
            row2,        // near  (depth [0,1])
            row3 - row2, // far
        ];

        // Normalize planes so that (a,b,c) is a unit normal.
        for p in &mut planes {
            let len = Vec3::new(p.x, p.y, p.z).length();
            if len > 1e-10 {
                *p /= len;
            }
        }

        Self { planes }
    }

    /// Test an AABB against the frustum with full containment info.
    ///
    /// Uses the p-vertex/n-vertex method: for each plane, find the corner
    /// most in the direction of the plane normal (p-vertex) and the opposite
    /// corner (n-vertex). Early-out on `Outside`.
    pub fn test_aabb(&self, aabb: &Aabb) -> Containment {
        let mut result = Containment::Inside;

        for plane in &self.planes {
            let normal = Vec3::new(plane.x, plane.y, plane.z);

            // p-vertex: corner most along the plane normal
            let p = Vec3::new(
                if normal.x >= 0.0 { aabb.max.x } else { aabb.min.x },
                if normal.y >= 0.0 { aabb.max.y } else { aabb.min.y },
                if normal.z >= 0.0 { aabb.max.z } else { aabb.min.z },
            );

            // n-vertex: opposite corner
            let n = Vec3::new(
                if normal.x >= 0.0 { aabb.min.x } else { aabb.max.x },
                if normal.y >= 0.0 { aabb.min.y } else { aabb.max.y },
                if normal.z >= 0.0 { aabb.min.z } else { aabb.max.z },
            );

            // If p-vertex is outside, the entire box is outside.
            if normal.dot(p) + plane.w < 0.0 {
                return Containment::Outside;
            }

            // If n-vertex is outside, the box intersects the plane.
            if normal.dot(n) + plane.w < 0.0 {
                result = Containment::Intersect;
            }
        }

        result
    }

    /// Fast visibility test — only distinguishes visible/invisible (no Inside vs Intersect).
    ///
    /// Slightly cheaper than `test_aabb` since it only tests the p-vertex.
    pub fn test_aabb_visible(&self, aabb: &Aabb) -> bool {
        for plane in &self.planes {
            let normal = Vec3::new(plane.x, plane.y, plane.z);

            let p = Vec3::new(
                if normal.x >= 0.0 { aabb.max.x } else { aabb.min.x },
                if normal.y >= 0.0 { aabb.max.y } else { aabb.min.y },
                if normal.z >= 0.0 { aabb.max.z } else { aabb.min.z },
            );

            if normal.dot(p) + plane.w < 0.0 {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_4;

    fn approx_eq(a: Vec3, b: Vec3, eps: f32) -> bool {
        (a - b).length() < eps
    }

    // ── AABB tests ──

    #[test]
    fn aabb_from_cube_vertices() {
        let cube = [
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.0, 0.5, -0.5),
        ];
        let aabb = Aabb::from_points(&cube);
        assert_eq!(aabb.min, Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(aabb.max, Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn aabb_from_empty() {
        let aabb = Aabb::from_points(&[]);
        assert_eq!(aabb, Aabb::EMPTY);
    }

    #[test]
    fn aabb_from_single_point() {
        let p = Vec3::new(3.0, 4.0, 5.0);
        let aabb = Aabb::from_points(&[p]);
        assert_eq!(aabb.min, p);
        assert_eq!(aabb.max, p);
    }

    #[test]
    fn aabb_center_and_half_extents() {
        let aabb = Aabb::new(Vec3::new(-2.0, -1.0, 0.0), Vec3::new(4.0, 3.0, 6.0));
        assert!(approx_eq(aabb.center(), Vec3::new(1.0, 1.0, 3.0), 1e-6));
        assert!(approx_eq(aabb.half_extents(), Vec3::new(3.0, 2.0, 3.0), 1e-6));
    }

    #[test]
    fn aabb_surface_area() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::new(2.0, 3.0, 4.0));
        // 2 * (2*3 + 3*4 + 4*2) = 2 * (6+12+8) = 52
        assert!((aabb.surface_area() - 52.0).abs() < 1e-6);
    }

    #[test]
    fn aabb_union() {
        let a = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let b = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(3.0, 3.0, 3.0));
        let u = a.union(b);
        assert_eq!(u.min, Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(u.max, Vec3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn aabb_union_with_empty() {
        let a = Aabb::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));
        let u = Aabb::EMPTY.union(a);
        assert_eq!(u.min, a.min);
        assert_eq!(u.max, a.max);
    }

    #[test]
    fn aabb_contains_point() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::new(2.0, 2.0, 2.0));
        assert!(aabb.contains_point(Vec3::new(1.0, 1.0, 1.0)));
        assert!(aabb.contains_point(Vec3::ZERO)); // edge
        assert!(!aabb.contains_point(Vec3::new(-0.1, 0.0, 0.0)));
        assert!(!aabb.contains_point(Vec3::new(3.0, 1.0, 1.0)));
    }

    #[test]
    fn aabb_intersects_overlap() {
        let a = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        let b = Aabb::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0));
        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
    }

    #[test]
    fn aabb_intersects_touching() {
        let a = Aabb::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        let b = Aabb::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 1.0, 1.0));
        assert!(a.intersects(&b)); // touching edge counts as intersection
    }

    #[test]
    fn aabb_intersects_disjoint() {
        let a = Aabb::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        let b = Aabb::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0));
        assert!(!a.intersects(&b));
        assert!(!b.intersects(&a));
    }

    #[test]
    fn aabb_intersects_contained() {
        let outer = Aabb::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0));
        let inner = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(outer.intersects(&inner));
        assert!(inner.intersects(&outer));
    }

    #[test]
    fn aabb_intersects_one_axis_miss() {
        let a = Aabb::new(Vec3::ZERO, Vec3::new(2.0, 2.0, 2.0));
        // Overlaps on X and Y but not Z
        let b = Aabb::new(Vec3::new(0.5, 0.5, 5.0), Vec3::new(1.5, 1.5, 6.0));
        assert!(!a.intersects(&b));
    }

    #[test]
    fn aabb_transformed_identity() {
        let aabb = Aabb::new(Vec3::new(-1.0, -2.0, -3.0), Vec3::new(1.0, 2.0, 3.0));
        let t = aabb.transformed(&Mat4::IDENTITY);
        assert!(approx_eq(t.min, aabb.min, 1e-5));
        assert!(approx_eq(t.max, aabb.max, 1e-5));
    }

    #[test]
    fn aabb_transformed_translation() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        let m = Mat4::from_translation(Vec3::new(10.0, 20.0, 30.0));
        let t = aabb.transformed(&m);
        assert!(approx_eq(t.min, Vec3::new(10.0, 20.0, 30.0), 1e-5));
        assert!(approx_eq(t.max, Vec3::new(11.0, 21.0, 31.0), 1e-5));
    }

    #[test]
    fn aabb_transformed_scale() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let m = Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0));
        let t = aabb.transformed(&m);
        assert!(approx_eq(t.min, Vec3::new(-2.0, -3.0, -4.0), 1e-5));
        assert!(approx_eq(t.max, Vec3::new(2.0, 3.0, 4.0), 1e-5));
    }

    #[test]
    fn aabb_transformed_rotation() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        // 90 degrees around Y
        let m = Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let t = aabb.transformed(&m);
        // A unit cube rotated 90° around Y should still have same extents (it's symmetric)
        assert!(approx_eq(t.min, Vec3::new(-1.0, -1.0, -1.0), 1e-4));
        assert!(approx_eq(t.max, Vec3::new(1.0, 1.0, 1.0), 1e-4));
    }

    #[test]
    fn aabb_transformed_rotation_nonsymmetric() {
        // Non-symmetric box: rotating should expand AABB
        let aabb = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.0, 1.0, 0.0));
        // 45 degrees around Z
        let m = Mat4::from_rotation_z(FRAC_PI_4);
        let t = aabb.transformed(&m);
        // The rotated box should be larger than the original on both X and Y
        assert!(t.max.x > 0.0);
        assert!(t.max.y > 1.0); // was 1, rotation should expand Y
    }

    // ── Frustum tests ──

    fn test_frustum() -> Frustum {
        let view = Mat4::look_at_rh(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
        );
        let proj = Mat4::perspective_rh(FRAC_PI_4, 1.0, 0.1, 100.0);
        Frustum::from_view_projection(&(proj * view))
    }

    #[test]
    fn frustum_test_inside() {
        let f = test_frustum();
        // Box at origin — should be inside the frustum
        let aabb = Aabb::new(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5));
        assert_eq!(f.test_aabb(&aabb), Containment::Inside);
        assert!(f.test_aabb_visible(&aabb));
    }

    #[test]
    fn frustum_test_outside_left() {
        let f = test_frustum();
        // Box far to the left — should be outside
        let aabb = Aabb::new(Vec3::new(-100.0, -0.5, -0.5), Vec3::new(-99.0, 0.5, 0.5));
        assert_eq!(f.test_aabb(&aabb), Containment::Outside);
        assert!(!f.test_aabb_visible(&aabb));
    }

    #[test]
    fn frustum_test_outside_right() {
        let f = test_frustum();
        let aabb = Aabb::new(Vec3::new(99.0, -0.5, -0.5), Vec3::new(100.0, 0.5, 0.5));
        assert_eq!(f.test_aabb(&aabb), Containment::Outside);
    }

    #[test]
    fn frustum_test_outside_behind() {
        let f = test_frustum();
        // Box behind the camera
        let aabb = Aabb::new(Vec3::new(-0.5, -0.5, 10.0), Vec3::new(0.5, 0.5, 11.0));
        assert_eq!(f.test_aabb(&aabb), Containment::Outside);
        assert!(!f.test_aabb_visible(&aabb));
    }

    #[test]
    fn frustum_test_outside_far() {
        let f = test_frustum();
        // Box beyond the far plane (far = 100, camera at z=5 looking at -z)
        let aabb = Aabb::new(Vec3::new(-0.5, -0.5, -200.0), Vec3::new(0.5, 0.5, -199.0));
        assert_eq!(f.test_aabb(&aabb), Containment::Outside);
    }

    #[test]
    fn frustum_test_intersect() {
        let f = test_frustum();
        // A large box straddling the frustum boundary
        let aabb = Aabb::new(Vec3::new(-50.0, -50.0, -50.0), Vec3::new(50.0, 50.0, 50.0));
        let result = f.test_aabb(&aabb);
        assert!(result == Containment::Intersect || result == Containment::Inside);
        assert!(f.test_aabb_visible(&aabb));
    }

    #[test]
    fn frustum_test_outside_above() {
        let f = test_frustum();
        let aabb = Aabb::new(Vec3::new(-0.5, 100.0, -0.5), Vec3::new(0.5, 101.0, 0.5));
        assert_eq!(f.test_aabb(&aabb), Containment::Outside);
    }

    #[test]
    fn frustum_test_outside_below() {
        let f = test_frustum();
        let aabb = Aabb::new(Vec3::new(-0.5, -101.0, -0.5), Vec3::new(0.5, -100.0, 0.5));
        assert_eq!(f.test_aabb(&aabb), Containment::Outside);
    }

    #[test]
    fn frustum_near_plane_culls() {
        let f = test_frustum();
        // Box right at the camera — but before the near plane won't exist in typical test
        // since near=0.1 and camera is at z=5, a box between z=5 and z=5.2 is behind
        let aabb = Aabb::new(Vec3::new(-0.1, -0.1, 5.05), Vec3::new(0.1, 0.1, 5.15));
        assert_eq!(f.test_aabb(&aabb), Containment::Outside);
    }

    #[test]
    fn frustum_visible_agrees_with_test() {
        let f = test_frustum();

        let test_boxes = [
            Aabb::new(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5)),
            Aabb::new(Vec3::new(-100.0, 0.0, 0.0), Vec3::new(-99.0, 1.0, 1.0)),
            Aabb::new(Vec3::new(0.0, 0.0, 10.0), Vec3::new(1.0, 1.0, 11.0)),
        ];

        for aabb in &test_boxes {
            let full = f.test_aabb(aabb);
            let quick = f.test_aabb_visible(aabb);
            assert_eq!(
                full != Containment::Outside,
                quick,
                "disagreement for {aabb:?}"
            );
        }
    }
}
