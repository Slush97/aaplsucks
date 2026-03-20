//! GPU compute shader skinning — transforms vertices on the GPU using joint matrices.

use glam::Mat4;
use wgpu::util::DeviceExt;

use super::gltf_loader::SkinningData;
use super::mesh::{Mesh, MeshData, MeshHandle};
use super::vertex::Vertex3D;

/// Compute pipeline for GPU skinning.
pub(crate) struct SkinningPipeline {
    compute_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// A mesh with GPU skinning buffers.
pub(crate) struct SkinnedMesh {
    /// Original vertex data (STORAGE, immutable). Kept alive for GPU references.
    #[allow(dead_code)]
    source_buffer: wgpu::Buffer,
    /// Skinned vertex data (STORAGE | VERTEX, written by compute, read by render).
    output_buffer: wgpu::Buffer,
    /// Per-vertex joint indices + weights (STORAGE). Kept alive for GPU references.
    #[allow(dead_code)]
    skin_data_buffer: wgpu::Buffer,
    /// Joint matrices (STORAGE, updated each frame).
    joint_buffer: wgpu::Buffer,
    /// Bind group for the compute dispatch.
    bind_group: wgpu::BindGroup,
    /// Number of vertices.
    vertex_count: u32,
    /// Maximum number of joints this buffer supports.
    max_joints: u32,
}

/// Per-vertex skinning data packed for GPU upload (32 bytes).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GpuSkinVertex {
    /// Joint indices (4 u32s).
    joints: [u32; 4],
    /// Blend weights (4 f32s).
    weights: [f32; 4],
}

impl SkinningPipeline {
    /// Create the compute pipeline for skinning.
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("esox_3d_skinning_layout"),
            entries: &[
                // binding 0: source vertices (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: output vertices (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: per-vertex skin data (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: joint matrices (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("esox_3d_skinning_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("esox_3d_skinning_shader"),
            source: wgpu::ShaderSource::Wgsl(SKINNING_SHADER.into()),
        });

        let compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("esox_3d_skinning_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("skin_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            compute_pipeline,
            bind_group_layout,
        }
    }
}

impl SkinningPipeline {
    /// Rebuild the compute pipeline with new shader source.
    #[cfg(feature = "hot-reload")]
    pub fn rebuild_pipeline(&mut self, device: &wgpu::Device, src: &str) {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("esox_3d_skinning_pipeline_layout"),
            bind_group_layouts: &[&self.bind_group_layout],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("esox_3d_skinning_shader"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });

        self.compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("esox_3d_skinning_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("skin_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
    }
}

impl SkinnedMesh {
    /// Create a skinned mesh from vertex data and skinning weights.
    pub fn new(
        device: &wgpu::Device,
        pipeline: &SkinningPipeline,
        mesh_data: &MeshData,
        skin_data: &SkinningData,
        max_joints: u32,
    ) -> Self {
        let vertex_count = mesh_data.vertices.len() as u32;

        // Source vertex buffer (STORAGE).
        let source_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("esox_3d_skin_source"),
            contents: bytemuck::cast_slice(&mesh_data.vertices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Output vertex buffer (STORAGE | VERTEX).
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_3d_skin_output"),
            size: (vertex_count as u64) * (size_of::<Vertex3D>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        // Pack skinning data.
        let gpu_skin: Vec<GpuSkinVertex> = skin_data
            .joints
            .iter()
            .zip(skin_data.weights.iter())
            .map(|(j, w)| GpuSkinVertex {
                joints: *j,
                weights: *w,
            })
            .collect();

        let skin_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("esox_3d_skin_data"),
            contents: bytemuck::cast_slice(&gpu_skin),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Joint matrices buffer.
        let joint_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_3d_skin_joints"),
            size: (max_joints as u64) * (size_of::<Mat4>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("esox_3d_skinning_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: source_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: skin_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: joint_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            source_buffer,
            output_buffer,
            skin_data_buffer,
            joint_buffer,
            bind_group,
            vertex_count,
            max_joints,
        }
    }
}

// ── Integration with Renderer3D ──

impl super::renderer::Renderer3D {
    /// Upload a skinned mesh and return (MeshHandle, skinned_mesh_index).
    ///
    /// The MeshHandle's vertex buffer will be the output of the compute skinning pass.
    pub fn upload_skinned_mesh(
        &mut self,
        gpu: &crate::pipeline::GpuContext,
        mesh_data: &MeshData,
        skin_data: &SkinningData,
        joint_count: u32,
    ) -> (MeshHandle, usize) {
        // Lazily create the skinning pipeline.
        if self.skinning_pipeline.is_none() {
            self.skinning_pipeline = Some(SkinningPipeline::new(&gpu.device));
        }
        let pipeline = self.skinning_pipeline.as_ref().unwrap();

        let skinned = SkinnedMesh::new(&gpu.device, pipeline, mesh_data, skin_data, joint_count);

        // Create a Mesh that uses the skinned output buffer as its vertex buffer.
        let index_buffer =
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("esox_3d_skinned_index_buffer"),
                    contents: bytemuck::cast_slice(&mesh_data.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

        let mesh = Mesh {
            vertex_buffer: skinned.output_buffer.clone(),
            index_buffer,
            index_count: mesh_data.indices.len() as u32,
        };

        let mesh_handle = MeshHandle(self.meshes.len() as u32 | super::render_types::SKINNED_MESH_BIT);
        self.meshes.push(mesh);

        let skinned_index = self.skinned_meshes.len();
        self.skinned_meshes.push(skinned);

        (mesh_handle, skinned_index)
    }

    /// Update joint matrices for a skinned mesh.
    pub fn update_joints(
        &mut self,
        gpu: &crate::pipeline::GpuContext,
        skinned_index: usize,
        matrices: &[Mat4],
    ) {
        if skinned_index >= self.skinned_meshes.len() {
            return;
        }
        let skinned = &self.skinned_meshes[skinned_index];
        let count = matrices.len().min(skinned.max_joints as usize);
        let data: Vec<[f32; 16]> = matrices[..count]
            .iter()
            .map(|m| m.to_cols_array())
            .collect();
        gpu.queue
            .write_buffer(&skinned.joint_buffer, 0, bytemuck::cast_slice(&data));
    }

    /// Dispatch compute skinning for all skinned meshes.
    ///
    /// Returns a command buffer to submit before the render pass, or `None` if no skinned meshes.
    pub fn dispatch_skinning(
        &self,
        gpu: &crate::pipeline::GpuContext,
    ) -> Option<wgpu::CommandBuffer> {
        let pipeline = self.skinning_pipeline.as_ref()?;
        if self.skinned_meshes.is_empty() {
            return None;
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("esox_3d_skinning_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("esox_3d_skinning_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.compute_pipeline);

            for skinned in &self.skinned_meshes {
                pass.set_bind_group(0, Some(&skinned.bind_group), &[]);
                let workgroups = (skinned.vertex_count + 63) / 64;
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        Some(encoder.finish())
    }
}

// ── Compute shader ──

pub(crate) const SKINNING_SHADER: &str = r"
// Vertex3D: position(3f), normal(3f), uv(2f), color(4f), tangent(4f) = 16 floats = 64 bytes
// Stored as array<vec4<f32>, 4> for alignment.

struct SkinVertex {
    joints: vec4<u32>,
    weights: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> source: array<array<vec4<f32>, 4>>;
@group(0) @binding(1) var<storage, read_write> output: array<array<vec4<f32>, 4>>;
@group(0) @binding(2) var<storage, read> skin_data: array<SkinVertex>;
@group(0) @binding(3) var<storage, read> joints: array<mat4x4<f32>>;

@compute @workgroup_size(64)
fn skin_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let vertex_count = arrayLength(&source);
    if idx >= vertex_count {
        return;
    }

    let src = source[idx];
    let skin = skin_data[idx];

    // Build skin matrix from 4 joint influences.
    var skin_mat = joints[skin.joints.x] * skin.weights.x
                 + joints[skin.joints.y] * skin.weights.y
                 + joints[skin.joints.z] * skin.weights.z
                 + joints[skin.joints.w] * skin.weights.w;

    // Extract position (vec4<f32>[0].xyz) and normal (vec4<f32>[0].w, vec4<f32>[1].xy).
    // Layout: [px py pz nx] [ny nz u v] [cr cg cb ca] [tx ty tz tw]
    let position = vec4<f32>(src[0].xyz, 1.0);
    let normal = vec3<f32>(src[0].w, src[1].x, src[1].y);
    let tangent_xyz = src[3].xyz;
    let tangent_w = src[3].w;

    // Transform position.
    let skinned_pos = skin_mat * position;

    // Transform normal using adjugate (cofactor) for correctness with non-uniform scale.
    let m0 = skin_mat[0].xyz;
    let m1 = skin_mat[1].xyz;
    let m2 = skin_mat[2].xyz;
    let adj = mat3x3<f32>(cross(m1, m2), cross(m2, m0), cross(m0, m1));
    let skinned_normal = normalize(normal * adj);

    // Transform tangent direction (tangents use the regular linear transform).
    let linear_mat = mat3x3<f32>(m0, m1, m2);
    let skinned_tangent = normalize(linear_mat * tangent_xyz);

    // Write output — preserve UV, color, tangent handedness.
    var out: array<vec4<f32>, 4>;
    out[0] = vec4<f32>(skinned_pos.xyz, skinned_normal.x);
    out[1] = vec4<f32>(skinned_normal.y, skinned_normal.z, src[1].z, src[1].w);
    out[2] = src[2]; // color unchanged
    out[3] = vec4<f32>(skinned_tangent, tangent_w);

    output[idx] = out;
}
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_skin_vertex_size() {
        assert_eq!(size_of::<GpuSkinVertex>(), 32);
    }

    #[test]
    fn gpu_skin_vertex_is_pod() {
        let v = GpuSkinVertex {
            joints: [0, 1, 2, 3],
            weights: [0.5, 0.3, 0.1, 0.1],
        };
        let _bytes: &[u8] = bytemuck::bytes_of(&v);
    }
}
