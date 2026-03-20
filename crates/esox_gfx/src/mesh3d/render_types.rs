//! Core render types and constants for the 3D renderer.

use super::instance::InstanceData;
use super::material::{BlendMode3D, MaterialHandle, MaterialType, PipelineKey};
use super::mesh::MeshHandle;

// ── Uniforms ──

/// GPU uniform data: view-projection matrix, camera position, viewport, time.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct Uniforms {
    /// Combined view-projection matrix (column-major).
    pub(super) view_projection: [[f32; 4]; 4],
    /// Camera world-space position (w unused).
    pub(super) camera_position: [f32; 4],
    /// Viewport: [width, height, 1/width, 1/height].
    pub(super) viewport: [f32; 4],
    /// Time: [elapsed_seconds, delta_seconds, 0, 0].
    pub(super) time: [f32; 4],
    /// Camera forward direction (xyz), w unused.
    pub(super) camera_forward: [f32; 4],
}

// ── Draw command ──

/// A queued draw command (mesh + material + instances).
pub(super) struct DrawCmd {
    pub(super) mesh: MeshHandle,
    pub(super) material: MaterialHandle,
    pub(super) instance_offset: u32,
    pub(super) instance_count: u32,
}

// ── Batch stats ──

/// Statistics from a single frame's draw batching.
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchStats3D {
    /// Number of draw calls issued.
    pub draw_calls: u32,
    /// Number of pipeline switches.
    pub pipeline_switches: u32,
    /// Number of material bind group switches.
    pub material_switches: u32,
    /// Total instances rendered.
    pub total_instances: u32,
    /// Total triangles rendered (instances * triangles-per-mesh).
    pub total_triangles: u32,
    /// Draw commands culled by frustum.
    pub culled_draws: u32,
    /// Instances culled by frustum.
    pub culled_instances: u32,
}

// ── Constants ──

/// Initial instance buffer capacity.
pub(super) const INITIAL_INSTANCE_CAPACITY: u64 = 4096;

/// Maximum instances per frame (safety limit).
pub(super) const MAX_INSTANCES: u32 = 500_000;

/// Depth texture format.
pub(super) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// HDR format for offscreen rendering (when post-processing is enabled).
pub(super) const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Bit flag to distinguish skinned (standalone-buffer) mesh handles from mega-buffer handles.
pub(crate) const SKINNED_MESH_BIT: u32 = 0x8000_0000;

/// Size of `DrawIndexedIndirectArgs` (5 × u32 = 20 bytes).
pub(super) const INDIRECT_ARGS_SIZE: u64 = 20;

/// Initial indirect buffer capacity.
pub(super) const INITIAL_INDIRECT_CAPACITY: u32 = 1024;

// ── Free functions ──

/// Sort key for draw commands: (pipeline key hash, material index, mesh index).
pub(super) fn pipeline_key_sort_tuple(
    key: &PipelineKey,
    material_idx: u32,
    mesh_idx: u32,
) -> (u8, u8, bool, u32, u32) {
    let mt = match key.material_type {
        MaterialType::Unlit => 0,
        MaterialType::Lit => 1,
        MaterialType::PBR => 2,
        MaterialType::Toon => 3,
    };
    let bm = match key.blend_mode {
        BlendMode3D::Opaque => 0,
        BlendMode3D::AlphaBlend => 1,
        BlendMode3D::Additive => 2,
    };
    (mt, bm, key.depth_write, material_idx, mesh_idx)
}

/// Extract the translation (column 3) from the first instance of a draw command.
pub(super) fn instance_translation(staging: &[InstanceData], offset: u32) -> glam::Vec3 {
    let inst = &staging[offset as usize];
    glam::Vec3::new(inst.model[3][0], inst.model[3][1], inst.model[3][2])
}

#[cfg(test)]
mod tests {
    use super::super::material::{CullMode3D};
    use super::*;

    #[test]
    fn draw_cmd_sort_order() {
        let keys = vec![
            PipelineKey {
                material_type: MaterialType::Lit,
                blend_mode: BlendMode3D::Opaque,
                cull_mode: CullMode3D::Back,
                depth_write: true,
            },
            PipelineKey {
                material_type: MaterialType::PBR,
                blend_mode: BlendMode3D::Opaque,
                cull_mode: CullMode3D::Back,
                depth_write: true,
            },
        ];

        let mut cmds = vec![
            DrawCmd {
                mesh: MeshHandle(1),
                material: MaterialHandle(1),
                instance_offset: 0,
                instance_count: 1,
            },
            DrawCmd {
                mesh: MeshHandle(0),
                material: MaterialHandle(0),
                instance_offset: 1,
                instance_count: 1,
            },
        ];

        cmds.sort_by(|a, b| {
            let key_a = &keys[a.material.0 as usize];
            let key_b = &keys[b.material.0 as usize];
            pipeline_key_sort_tuple(key_a, a.material.0, a.mesh.0).cmp(
                &pipeline_key_sort_tuple(key_b, b.material.0, b.mesh.0),
            )
        });

        assert_eq!(cmds[0].material.0, 0); // Lit
        assert_eq!(cmds[1].material.0, 1); // PBR
    }

    #[test]
    fn draw_cmd_merge_adjacent() {
        let cmds = vec![
            DrawCmd {
                mesh: MeshHandle(0),
                material: MaterialHandle(0),
                instance_offset: 0,
                instance_count: 5,
            },
            DrawCmd {
                mesh: MeshHandle(0),
                material: MaterialHandle(0),
                instance_offset: 5,
                instance_count: 3,
            },
            DrawCmd {
                mesh: MeshHandle(1),
                material: MaterialHandle(0),
                instance_offset: 8,
                instance_count: 2,
            },
        ];

        let mut merged = Vec::new();
        let mut i = 0;
        while i < cmds.len() {
            let mut count = cmds[i].instance_count;
            let mut j = i + 1;
            while j < cmds.len()
                && cmds[j].material.0 == cmds[i].material.0
                && cmds[j].mesh.0 == cmds[i].mesh.0
                && cmds[j].instance_offset == cmds[i].instance_offset + count
            {
                count += cmds[j].instance_count;
                j += 1;
            }
            merged.push((
                cmds[i].mesh.0,
                cmds[i].material.0,
                cmds[i].instance_offset,
                count,
            ));
            i = j;
        }

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0], (0, 0, 0, 8));
        assert_eq!(merged[1], (1, 0, 8, 2));
    }

    #[test]
    fn batch_stats_default() {
        let stats = BatchStats3D::default();
        assert_eq!(stats.draw_calls, 0);
        assert_eq!(stats.total_triangles, 0);
    }

    #[test]
    fn transparency_sort_back_to_front() {
        let cam_pos = glam::Vec3::new(0.0, 0.0, 0.0);

        let staging = vec![
            InstanceData {
                model: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, -5.0, 1.0],
                ],
                color: [1.0; 4],
                params: [0.0; 4],
            },
            InstanceData {
                model: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, -20.0, 1.0],
                ],
                color: [1.0; 4],
                params: [0.0; 4],
            },
            InstanceData {
                model: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, -10.0, 1.0],
                ],
                color: [1.0; 4],
                params: [0.0; 4],
            },
        ];

        let mut indices: Vec<usize> = vec![0, 1, 2];
        indices.sort_by(|&a, &b| {
            let pos_a = instance_translation(&staging, a as u32);
            let pos_b = instance_translation(&staging, b as u32);
            let dist_a = cam_pos.distance_squared(pos_a);
            let dist_b = cam_pos.distance_squared(pos_b);
            dist_b
                .partial_cmp(&dist_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        assert_eq!(indices, vec![1, 2, 0]);
    }
}
