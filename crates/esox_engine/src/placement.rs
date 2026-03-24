//! Grid placement system — snap-to-grid building placement with ghost preview.

use glam::{Quat, Vec3};

use esox_gfx::mesh3d::{InstanceData, MaterialHandle, MeshHandle, Renderer3D, Transform};

use crate::physics::PhysicsBackend;

/// Grid configuration for the placement system.
#[derive(Debug, Clone, Copy)]
pub struct PlacementGrid {
    /// Size of one grid cell in world units.
    pub cell_size: f32,
    /// Y-level of the placement plane.
    pub plane_y: f32,
}

impl Default for PlacementGrid {
    fn default() -> Self {
        Self {
            cell_size: 1.0,
            plane_y: 0.0,
        }
    }
}

/// Rotation of a placed building in 90-degree increments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementRotation {
    North,
    East,
    South,
    West,
}

impl Default for PlacementRotation {
    fn default() -> Self {
        Self::North
    }
}

impl PlacementRotation {
    /// Rotate 90 degrees clockwise.
    pub fn rotate_cw(self) -> Self {
        match self {
            Self::North => Self::East,
            Self::East => Self::South,
            Self::South => Self::West,
            Self::West => Self::North,
        }
    }

    /// Rotate 90 degrees counter-clockwise.
    pub fn rotate_ccw(self) -> Self {
        match self {
            Self::North => Self::West,
            Self::West => Self::South,
            Self::South => Self::East,
            Self::East => Self::North,
        }
    }

    /// Convert to a Y-axis quaternion.
    pub fn to_quat(self) -> Quat {
        match self {
            Self::North => Quat::IDENTITY,
            Self::East => Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            Self::South => Quat::from_rotation_y(std::f32::consts::PI),
            Self::West => Quat::from_rotation_y(3.0 * std::f32::consts::FRAC_PI_2),
        }
    }
}

/// State of the ghost placement preview.
pub struct PlacementGhost {
    /// Mesh to render for the ghost.
    pub mesh: MeshHandle,
    /// Material for the ghost (should be translucent).
    pub ghost_material: MaterialHandle,
    /// Footprint in grid cells (width, depth) before rotation.
    pub footprint: (u32, u32),
    /// Current rotation.
    pub rotation: PlacementRotation,
    /// Snapped world position (set by [`update_ghost`]).
    pub snapped_position: Vec3,
    /// Whether the placement is valid (no overlap).
    pub valid: bool,
}

/// Snap a world-space XZ position to the grid, accounting for multi-cell footprint.
///
/// Odd-width footprints snap to cell centers; even-width snap to cell edges.
pub fn snap_to_grid(
    pos: Vec3,
    grid: &PlacementGrid,
    footprint: (u32, u32),
    rotation: PlacementRotation,
) -> Vec3 {
    let cell = grid.cell_size;
    let (fw, fd) = match rotation {
        PlacementRotation::North | PlacementRotation::South => footprint,
        PlacementRotation::East | PlacementRotation::West => (footprint.1, footprint.0),
    };

    let snap_x = if fw % 2 == 0 {
        (pos.x / cell).round() * cell
    } else {
        ((pos.x / cell).floor() + 0.5) * cell
    };
    let snap_z = if fd % 2 == 0 {
        (pos.z / cell).round() * cell
    } else {
        ((pos.z / cell).floor() + 0.5) * cell
    };

    Vec3::new(snap_x, grid.plane_y, snap_z)
}

/// Check whether a placement is valid using the physics overlap query.
///
/// Returns `true` if no existing colliders overlap the given box.
pub fn check_placement_valid(
    physics: &dyn PhysicsBackend,
    position: Vec3,
    half_extents: Vec3,
) -> bool {
    physics.overlap_box(position, half_extents).is_empty()
}

/// Update the ghost position and validity from a world-space cursor position.
pub fn update_ghost(
    ghost: &mut PlacementGhost,
    cursor_world_pos: Vec3,
    grid: &PlacementGrid,
    physics: &dyn PhysicsBackend,
) {
    ghost.snapped_position = snap_to_grid(cursor_world_pos, grid, ghost.footprint, ghost.rotation);

    let (fw, fd) = match ghost.rotation {
        PlacementRotation::North | PlacementRotation::South => ghost.footprint,
        PlacementRotation::East | PlacementRotation::West => {
            (ghost.footprint.1, ghost.footprint.0)
        }
    };
    let half_extents = Vec3::new(
        fw as f32 * grid.cell_size * 0.5,
        0.5, // thin vertical extent for overlap check
        fd as f32 * grid.cell_size * 0.5,
    );
    ghost.valid = check_placement_valid(physics, ghost.snapped_position, half_extents);
}

/// Draw the ghost preview as a translucent colored instance.
pub fn draw_ghost(ghost: &PlacementGhost, renderer: &mut Renderer3D) {
    let tint = if ghost.valid {
        [0.3, 1.0, 0.3, 0.5] // green translucent
    } else {
        [1.0, 0.3, 0.3, 0.5] // red translucent
    };
    let instance = InstanceData::with_color(
        &Transform {
            position: ghost.snapped_position,
            rotation: ghost.rotation.to_quat(),
            scale: Vec3::ONE,
        },
        tint,
    );
    renderer.draw_with_material(ghost.mesh, ghost.ghost_material, &[instance]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap_1x1_to_cell_center() {
        let grid = PlacementGrid {
            cell_size: 1.0,
            plane_y: 0.0,
        };
        let pos = Vec3::new(2.3, 5.0, 4.7);
        let snapped = snap_to_grid(pos, &grid, (1, 1), PlacementRotation::North);
        assert!((snapped.x - 2.5).abs() < 1e-5);
        assert!((snapped.z - 4.5).abs() < 1e-5);
        assert!((snapped.y - 0.0).abs() < 1e-5);
    }

    #[test]
    fn snap_2x2_to_cell_edge() {
        let grid = PlacementGrid {
            cell_size: 1.0,
            plane_y: 0.0,
        };
        let pos = Vec3::new(2.3, 0.0, 4.7);
        let snapped = snap_to_grid(pos, &grid, (2, 2), PlacementRotation::North);
        assert!((snapped.x - 2.0).abs() < 1e-5);
        assert!((snapped.z - 5.0).abs() < 1e-5);
    }

    #[test]
    fn snap_rotated_swaps_footprint() {
        let grid = PlacementGrid {
            cell_size: 1.0,
            plane_y: 0.0,
        };
        let pos = Vec3::new(2.3, 0.0, 4.7);
        // 1x2 rotated East becomes 2x1
        let north = snap_to_grid(pos, &grid, (1, 2), PlacementRotation::North);
        let east = snap_to_grid(pos, &grid, (1, 2), PlacementRotation::East);
        // North: 1-wide (odd→center), 2-deep (even→edge)
        assert!((north.x - 2.5).abs() < 1e-5);
        assert!((north.z - 5.0).abs() < 1e-5);
        // East: 2-wide (even→edge), 1-deep (odd→center)
        assert!((east.x - 2.0).abs() < 1e-5);
        assert!((east.z - 4.5).abs() < 1e-5);
    }

    #[test]
    fn rotation_cycle() {
        let r = PlacementRotation::North;
        assert_eq!(r.rotate_cw().rotate_cw().rotate_cw().rotate_cw(), r);
        assert_eq!(r.rotate_ccw().rotate_ccw().rotate_ccw().rotate_ccw(), r);
        assert_eq!(r.rotate_cw(), PlacementRotation::East);
        assert_eq!(r.rotate_ccw(), PlacementRotation::West);
    }
}
