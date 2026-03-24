//! Level-of-detail (LOD) component for distance-based mesh selection.

use esox_gfx::mesh3d::MeshHandle;

/// A single LOD level: the mesh to use and the maximum camera distance for this level.
#[derive(Debug, Clone, Copy)]
pub struct LodLevel {
    pub mesh: MeshHandle,
    /// Maximum distance (not squared) from the camera at which this LOD is used.
    pub max_distance: f32,
}

/// Component for entities that support mesh LOD.
///
/// Levels must be sorted by ascending `max_distance`. The last level is used
/// for all distances beyond its threshold.
#[derive(Debug, Clone)]
pub struct LodMesh {
    pub levels: Vec<LodLevel>,
}

impl LodMesh {
    /// Select the appropriate mesh handle for the given squared distance.
    ///
    /// Returns `None` if the level list is empty (caller should fall back to
    /// the base [`MeshRenderer::mesh`]).
    pub fn select(&self, distance_sq: f32) -> Option<MeshHandle> {
        for level in &self.levels {
            if distance_sq <= level.max_distance * level.max_distance {
                return Some(level.mesh);
            }
        }
        self.levels.last().map(|l| l.mesh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a test MeshHandle. MeshHandle has a crate-private inner field,
    /// but it's a `#[repr(transparent)]` u32 newtype, so transmute is safe.
    fn h(id: u32) -> MeshHandle {
        unsafe { std::mem::transmute(id) }
    }

    #[test]
    fn select_nearest_lod() {
        let lod = LodMesh {
            levels: vec![
                LodLevel { mesh: h(1), max_distance: 10.0 },
                LodLevel { mesh: h(2), max_distance: 50.0 },
                LodLevel { mesh: h(3), max_distance: 200.0 },
            ],
        };
        assert_eq!(lod.select(5.0 * 5.0), Some(h(1)));
        assert_eq!(lod.select(30.0 * 30.0), Some(h(2)));
        assert_eq!(lod.select(100.0 * 100.0), Some(h(3)));
        assert_eq!(lod.select(999.0 * 999.0), Some(h(3)));
    }

    #[test]
    fn select_empty_returns_none() {
        let lod = LodMesh { levels: vec![] };
        assert_eq!(lod.select(10.0), None);
    }
}
