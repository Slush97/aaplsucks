//! Ground overlay system — decals and highlights projected onto the Y=0 plane.
//!
//! Overlays are flat quads rendered slightly above the ground to avoid z-fighting.
//! They are queued per-frame and cleared after drawing.

use glam::Vec3;

use esox_gfx::mesh3d::{
    InstanceData, MaterialHandle, MeshData, MeshHandle, Renderer3D, Transform,
};
use esox_gfx::GpuContext;

/// Height offset above Y=0 to prevent z-fighting with the ground plane.
const OVERLAY_Y_OFFSET: f32 = 0.01;

/// A single ground overlay to be drawn this frame.
pub struct GroundOverlay {
    /// XZ world position (Y is ignored — uses [`OVERLAY_Y_OFFSET`]).
    pub position: Vec3,
    /// Width and depth in world units.
    pub size: (f32, f32),
    /// Rotation around the Y axis in radians.
    pub rotation: f32,
    /// Color/alpha tint.
    pub color: [f32; 4],
    /// Material to use (should be alpha-blended, `depth_write: false`).
    pub material: MaterialHandle,
}

/// Manages and batches ground overlays for rendering.
pub struct GroundOverlayRenderer {
    overlays: Vec<GroundOverlay>,
    quad_mesh: Option<MeshHandle>,
}

impl GroundOverlayRenderer {
    pub fn new() -> Self {
        Self {
            overlays: Vec::new(),
            quad_mesh: None,
        }
    }

    /// Ensure the shared unit quad mesh is uploaded (lazy init).
    pub fn ensure_mesh(&mut self, gpu: &GpuContext, renderer: &mut Renderer3D) {
        if self.quad_mesh.is_none() {
            let data = MeshData::plane(1.0, 1.0, 1);
            self.quad_mesh = Some(renderer.upload_mesh(gpu, &data));
        }
    }

    /// Queue an overlay for this frame.
    pub fn add(&mut self, overlay: GroundOverlay) {
        self.overlays.push(overlay);
    }

    /// Batch and draw all queued overlays, then clear.
    ///
    /// Call this during the render phase, after [`ensure_mesh`](Self::ensure_mesh).
    pub fn draw(&mut self, renderer: &mut Renderer3D) {
        if self.overlays.is_empty() {
            return;
        }

        let mesh = match self.quad_mesh {
            Some(m) => m,
            None => return,
        };

        // Sort by material for efficient batching.
        self.overlays.sort_unstable_by_key(|o| o.material.0);

        let mut batch_start = 0;
        while batch_start < self.overlays.len() {
            let mat = self.overlays[batch_start].material;
            let batch_end = self.overlays[batch_start..]
                .iter()
                .position(|o| o.material != mat)
                .map_or(self.overlays.len(), |p| batch_start + p);

            let instances: Vec<InstanceData> = self.overlays[batch_start..batch_end]
                .iter()
                .map(|o| {
                    InstanceData::with_color(
                        &Transform {
                            position: Vec3::new(o.position.x, OVERLAY_Y_OFFSET, o.position.z),
                            rotation: glam::Quat::from_rotation_y(o.rotation),
                            scale: Vec3::new(o.size.0, 1.0, o.size.1),
                        },
                        o.color,
                    )
                })
                .collect();

            renderer.draw_with_material(mesh, mat, &instances);
            batch_start = batch_end;
        }

        self.overlays.clear();
    }

    /// Number of overlays queued this frame.
    pub fn count(&self) -> usize {
        self.overlays.len()
    }
}

impl Default for GroundOverlayRenderer {
    fn default() -> Self {
        Self::new()
    }
}
