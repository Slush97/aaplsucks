//! LOD (Level of Detail) — distance-based mesh selection.

use glam::Vec3;

use super::bounds::Aabb;
use super::mesh::MeshHandle;

/// A single LOD level: use this mesh up to `max_distance`.
#[derive(Debug, Clone, Copy)]
pub struct LodLevel {
    /// Mesh to render at this detail level.
    pub mesh: MeshHandle,
    /// Maximum view distance for this level (in world units).
    pub max_distance: f32,
}

/// A group of LOD levels for the same logical object.
///
/// Levels are sorted by `max_distance` ascending (finest detail first).
pub struct LodGroup {
    levels: Vec<LodLevel>,
    /// AABB from the highest-detail mesh (used for culling at all levels).
    pub aabb: Aabb,
}

impl LodGroup {
    /// Create a new LOD group.
    ///
    /// `levels` must be sorted by `max_distance` ascending (finest first).
    /// `aabb` should come from the highest-detail mesh.
    pub fn new(levels: Vec<LodLevel>, aabb: Aabb) -> Self {
        debug_assert!(
            levels.windows(2).all(|w| w[0].max_distance <= w[1].max_distance),
            "LodGroup levels must be sorted by max_distance ascending"
        );
        Self { levels, aabb }
    }

    /// Select the appropriate mesh handle for a given distance.
    ///
    /// Returns the first level whose `max_distance >= distance`.
    /// If beyond all levels, returns the coarsest (last) mesh.
    pub fn select(&self, distance: f32) -> MeshHandle {
        for level in &self.levels {
            if distance <= level.max_distance {
                return level.mesh;
            }
        }
        // Beyond all levels — use coarsest
        self.levels
            .last()
            .map(|l| l.mesh)
            .unwrap_or(MeshHandle(0))
    }

    /// Number of LOD levels.
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }
}

/// Compute the projected screen-space radius of a bounding sphere.
///
/// Useful for screen-space LOD thresholds.
pub fn projected_radius(
    sphere_radius: f32,
    distance: f32,
    fov_y: f32,
    viewport_height: f32,
) -> f32 {
    if distance <= 0.0 {
        return viewport_height;
    }
    let cot_half_fov = 1.0 / (fov_y * 0.5).tan();
    sphere_radius * cot_half_fov * viewport_height / (2.0 * distance)
}

/// Compute the distance from a camera position to an instance (via model matrix translation).
pub fn instance_distance(camera_pos: Vec3, model_translation: Vec3) -> f32 {
    camera_pos.distance(model_translation)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_lod_group() -> LodGroup {
        LodGroup::new(
            vec![
                LodLevel {
                    mesh: MeshHandle(0),
                    max_distance: 10.0,
                },
                LodLevel {
                    mesh: MeshHandle(1),
                    max_distance: 50.0,
                },
                LodLevel {
                    mesh: MeshHandle(2),
                    max_distance: 200.0,
                },
            ],
            Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
        )
    }

    #[test]
    fn select_closest() {
        let lod = make_lod_group();
        assert_eq!(lod.select(5.0), MeshHandle(0));
    }

    #[test]
    fn select_mid() {
        let lod = make_lod_group();
        assert_eq!(lod.select(25.0), MeshHandle(1));
    }

    #[test]
    fn select_at_boundary() {
        let lod = make_lod_group();
        assert_eq!(lod.select(10.0), MeshHandle(0));
        assert_eq!(lod.select(50.0), MeshHandle(1));
    }

    #[test]
    fn select_farthest() {
        let lod = make_lod_group();
        assert_eq!(lod.select(150.0), MeshHandle(2));
    }

    #[test]
    fn select_beyond_all() {
        let lod = make_lod_group();
        assert_eq!(lod.select(500.0), MeshHandle(2));
    }

    #[test]
    fn projected_radius_basic() {
        let r = projected_radius(1.0, 10.0, std::f32::consts::FRAC_PI_4, 1080.0);
        assert!(r > 0.0);
        // Closer should give larger radius
        let r_close = projected_radius(1.0, 5.0, std::f32::consts::FRAC_PI_4, 1080.0);
        assert!(r_close > r);
    }

    #[test]
    fn projected_radius_zero_distance() {
        let r = projected_radius(1.0, 0.0, std::f32::consts::FRAC_PI_4, 1080.0);
        assert_eq!(r, 1080.0);
    }
}
