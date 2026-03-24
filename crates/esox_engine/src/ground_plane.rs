//! Infinite ground plane — tiled flat quads that follow the camera.

use glam::Vec3;

use esox_gfx::mesh3d::{
    InstanceData, MaterialHandle, MeshData, MeshHandle, Renderer3D, Transform,
};
use esox_gfx::GpuContext;

/// Configuration for the ground plane.
pub struct GroundPlaneConfig {
    /// Size of each ground tile in world units (tiles are square).
    pub tile_size: f32,
    /// How many tiles to render around the camera in each direction.
    pub render_radius: u32,
    /// Material to use for ground tiles.
    pub material: MaterialHandle,
}

/// Manages a pool of ground-plane tile meshes that follow the camera.
pub struct GroundPlane {
    tile_size: f32,
    render_radius: i32,
    material: MaterialHandle,
    tile_mesh: MeshHandle,
}

impl GroundPlane {
    /// Create the ground plane. Uploads a single flat quad mesh.
    pub fn new(gpu: &GpuContext, renderer: &mut Renderer3D, config: GroundPlaneConfig) -> Self {
        let mesh_data = MeshData::plane(config.tile_size, config.tile_size, 1);
        let tile_mesh = renderer.upload_mesh(gpu, &mesh_data);
        Self {
            tile_size: config.tile_size,
            render_radius: config.render_radius as i32,
            material: config.material,
            tile_mesh,
        }
    }

    /// Issue draw calls for visible ground tiles around the camera.
    ///
    /// Tiles snap to the tile grid so the ground appears infinite as the
    /// camera moves. Cost: `(2*radius+1)^2` instances per frame.
    pub fn draw(&self, renderer: &mut Renderer3D, camera_pos: Vec3) {
        let ts = self.tile_size;
        let cam_tile_x = (camera_pos.x / ts).floor() as i32;
        let cam_tile_z = (camera_pos.z / ts).floor() as i32;
        let r = self.render_radius;

        let side = (2 * r + 1) as usize;
        let mut instances = Vec::with_capacity(side * side);

        for dx in -r..=r {
            for dz in -r..=r {
                let wx = (cam_tile_x + dx) as f32 * ts + ts * 0.5;
                let wz = (cam_tile_z + dz) as f32 * ts + ts * 0.5;
                instances.push(InstanceData::from_transform(&Transform {
                    position: Vec3::new(wx, 0.0, wz),
                    ..Transform::default()
                }));
            }
        }

        renderer.draw_with_material(self.tile_mesh, self.material, &instances);
    }

    /// The mesh handle for the ground tile (useful for custom rendering).
    pub fn tile_mesh(&self) -> MeshHandle {
        self.tile_mesh
    }

    /// The material used for ground tiles.
    pub fn material(&self) -> MaterialHandle {
        self.material
    }
}
