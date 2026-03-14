//! GPU compute-driven particle system.
//!
//! Particle state lives entirely on the GPU; the CPU only writes emitter
//! parameters each frame. Rendering uses indirect draw with the existing
//! instanced mesh pipeline.

use wgpu::util::DeviceExt;

use super::instance::InstanceData;
use super::material::MaterialHandle;

/// Handle to a particle pool (index into `Renderer3D::particle_pools`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParticlePoolHandle(pub u32);

/// GPU-side particle (64 bytes, storage buffer).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GpuParticle {
    position: [f32; 3],
    age: f32,       // 16B
    velocity: [f32; 3],
    lifetime: f32,  // 16B
    color_start: [f32; 4], // 16B
    color_end: [f32; 4],   // 16B
}

/// Emitter parameters uploaded by CPU each frame (128 bytes, uniform buffer).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct EmitterParams {
    pub origin: [f32; 3],
    pub spawn_count: u32,
    pub velocity_min: [f32; 3],
    pub size_start: f32,
    pub velocity_max: [f32; 3],
    pub size_end: f32,
    pub gravity: [f32; 3],
    pub lifetime_min: f32,
    pub color_start: [f32; 4],
    pub color_end: [f32; 4],
    pub dt: f32,
    pub lifetime_max: f32,
    pub seed: u32,
    pub _pad: u32,
}

/// Draw command queued for particle rendering.
pub(crate) struct ParticleDrawCmd {
    pub pool: ParticlePoolHandle,
    pub material: MaterialHandle,
}

/// Compute pipeline for particle simulation.
pub(crate) struct ParticlePipeline {
    update_pipeline: wgpu::ComputePipeline,
    finalize_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// Per-pool GPU resources.
pub(crate) struct ParticlePool {
    /// GpuParticle array (STORAGE rw).
    #[allow(dead_code)]
    particle_buffer: wgpu::Buffer,
    /// InstanceData array (STORAGE rw | VERTEX) — written by compute, read by draw.
    pub(crate) instance_output: wgpu::Buffer,
    /// [alive_count, remaining_spawns] (STORAGE rw | COPY_DST).
    counters_buffer: wgpu::Buffer,
    /// DrawIndexedIndirect args (INDIRECT | STORAGE rw).
    pub(crate) indirect_args_buffer: wgpu::Buffer,
    /// EmitterParams (UNIFORM | COPY_DST).
    emitter_buffer: wgpu::Buffer,
    /// Bind group for compute dispatch.
    bind_group: wgpu::BindGroup,
    /// Pool capacity (max particles).
    capacity: u32,
    /// Whether this pool has been activated this frame.
    active: bool,
}

impl ParticlePipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("esox_3d_particle_layout"),
                entries: &[
                    // binding 0: particle buffer (rw storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: instance output (rw storage)
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
                    // binding 2: counters (rw storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 3: emitter params (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 4: indirect args (rw storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("esox_3d_particle_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("esox_3d_particle_shader"),
            source: wgpu::ShaderSource::Wgsl(PARTICLE_SHADER.into()),
        });

        let update_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("esox_3d_particle_update"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("update_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let finalize_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("esox_3d_particle_finalize"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("finalize_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            update_pipeline,
            finalize_pipeline,
            bind_group_layout,
        }
    }
}

impl ParticlePool {
    fn new(device: &wgpu::Device, pipeline: &ParticlePipeline, capacity: u32) -> Self {
        let particle_size = size_of::<GpuParticle>() as u64;
        let instance_size = size_of::<InstanceData>() as u64;

        // All particles start dead (age >= lifetime, both zero).
        let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_3d_particle_buf"),
            size: capacity as u64 * particle_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let instance_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_3d_particle_instances"),
            size: capacity as u64 * instance_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        // counters: [alive_count: u32, remaining_spawns: u32]
        let counters_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("esox_3d_particle_counters"),
                contents: bytemuck::cast_slice(&[0u32, 0u32]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // DrawIndexedIndirect: [index_count, instance_count, first_index, base_vertex, first_instance]
        let indirect_args_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("esox_3d_particle_indirect"),
                contents: bytemuck::cast_slice(&[6u32, 0u32, 0u32, 0u32, 0u32]),
                usage: wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
            });

        let emitter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_3d_particle_emitter"),
            size: size_of::<EmitterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("esox_3d_particle_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instance_output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: counters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: emitter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: indirect_args_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            particle_buffer,
            instance_output,
            counters_buffer,
            indirect_args_buffer,
            emitter_buffer,
            bind_group,
            capacity,
            active: false,
        }
    }
}

// ── Integration with Renderer3D ──

impl super::renderer::Renderer3D {
    /// Create a particle pool with the given capacity. Returns a handle used to
    /// reference this pool when updating emitters and drawing particles.
    pub fn create_particle_pool(
        &mut self,
        gpu: &crate::pipeline::GpuContext,
        capacity: u32,
    ) -> ParticlePoolHandle {
        if self.particle_pipeline.is_none() {
            self.particle_pipeline = Some(ParticlePipeline::new(&gpu.device));
        }
        let pipeline = self.particle_pipeline.as_ref().unwrap();
        let pool = ParticlePool::new(&gpu.device, pipeline, capacity);
        let handle = ParticlePoolHandle(self.particle_pools.len() as u32);
        self.particle_pools.push(pool);

        // Ensure the quad mesh exists for billboard rendering.
        if self.particle_quad_mesh.is_none() {
            self.particle_quad_mesh = Some(self.upload_particle_quad(gpu));
        }

        // Patch indirect args with the quad mesh's actual offsets in the mega-buffer.
        if let Some(quad_handle) = self.particle_quad_mesh {
            let quad_idx = quad_handle.0 as usize;
            if quad_idx < self.mesh_regions.len() {
                let r = &self.mesh_regions[quad_idx];
                let pool = &self.particle_pools[handle.0 as usize];
                // DrawIndexedIndirect: [index_count, instance_count, first_index, base_vertex (i32), first_instance]
                let args: [u32; 5] = [
                    r.index_count,
                    0,
                    r.index_offset,
                    r.vertex_offset, // safe: vertex_offset fits in i32 range
                    0,
                ];
                gpu.queue.write_buffer(
                    &pool.indirect_args_buffer,
                    0,
                    bytemuck::cast_slice(&args),
                );
            }
        }

        handle
    }

    /// Update emitter parameters for a particle pool. Call once per frame per
    /// active emitter before `dispatch_particles`.
    pub fn update_particle_emitter(
        &mut self,
        gpu: &crate::pipeline::GpuContext,
        pool: ParticlePoolHandle,
        params: &EmitterParams,
    ) {
        let idx = pool.0 as usize;
        if idx >= self.particle_pools.len() {
            return;
        }
        let pool = &mut self.particle_pools[idx];
        pool.active = true;

        // Write emitter params.
        gpu.queue
            .write_buffer(&pool.emitter_buffer, 0, bytemuck::bytes_of(params));

        // Reset counters: alive_count=0, remaining_spawns=spawn_count.
        gpu.queue.write_buffer(
            &pool.counters_buffer,
            0,
            bytemuck::cast_slice(&[0u32, params.spawn_count]),
        );
    }

    /// Queue a particle pool for rendering with the given material.
    pub fn draw_particles(&mut self, pool: ParticlePoolHandle, material: MaterialHandle) {
        self.particle_draw_cmds.push(ParticleDrawCmd { pool, material });
    }

    /// Run the compute pass for all active particle pools.
    /// Returns a command buffer to submit before the render pass.
    pub fn dispatch_particles(
        &mut self,
        gpu: &crate::pipeline::GpuContext,
    ) -> Option<wgpu::CommandBuffer> {
        let pipeline = self.particle_pipeline.as_ref()?;
        let any_active = self.particle_pools.iter().any(|p| p.active);
        if !any_active {
            return None;
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("esox_3d_particle_encoder"),
            });

        // Update pass — simulate all active pools.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("esox_3d_particle_update_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.update_pipeline);

            for pool in &self.particle_pools {
                if !pool.active {
                    continue;
                }
                pass.set_bind_group(0, Some(&pool.bind_group), &[]);
                let workgroups = (pool.capacity + 63) / 64;
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        // Finalize pass — copy alive_count into indirect args.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("esox_3d_particle_finalize_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.finalize_pipeline);

            for pool in &self.particle_pools {
                if !pool.active {
                    continue;
                }
                pass.set_bind_group(0, Some(&pool.bind_group), &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }

        // Reset active flags for next frame.
        for pool in &mut self.particle_pools {
            pool.active = false;
        }

        Some(encoder.finish())
    }

    /// Upload a unit quad mesh for particle billboards. Returns a MeshHandle
    /// from the mega-buffer.
    fn upload_particle_quad(
        &mut self,
        gpu: &crate::pipeline::GpuContext,
    ) -> super::mesh::MeshHandle {
        use super::mesh::MeshData;
        use super::vertex::Vertex3D;

        let vertices = vec![
            Vertex3D {
                position: [-0.5, -0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex3D {
                position: [0.5, -0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [1.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex3D {
                position: [0.5, 0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [1.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex3D {
                position: [-0.5, 0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];
        let data = MeshData { vertices, indices };
        self.upload_mesh(gpu, &data)
    }
}

// ── WGSL compute shader ──

const PARTICLE_SHADER: &str = r#"
// Particle simulation compute shader.
//
// Two entry points:
// - update_main: simulates + spawns particles (1 thread per slot, workgroup 64)
// - finalize_main: copies alive_count into indirect args (1 thread)

struct GpuParticle {
    position: vec3<f32>,
    age: f32,
    velocity: vec3<f32>,
    lifetime: f32,
    color_start: vec4<f32>,
    color_end: vec4<f32>,
}

struct InstanceData {
    model_col0: vec4<f32>,
    model_col1: vec4<f32>,
    model_col2: vec4<f32>,
    model_col3: vec4<f32>,
    color: vec4<f32>,
    params: vec4<f32>,
}

struct EmitterParams {
    origin: vec3<f32>,
    spawn_count: u32,
    velocity_min: vec3<f32>,
    size_start: f32,
    velocity_max: vec3<f32>,
    size_end: f32,
    gravity: vec3<f32>,
    lifetime_min: f32,
    color_start: vec4<f32>,
    color_end: vec4<f32>,
    dt: f32,
    lifetime_max: f32,
    seed: u32,
    _pad: u32,
}

// Counters: [0] = alive_count, [1] = remaining_spawns
struct Counters {
    alive_count: atomic<u32>,
    remaining_spawns: atomic<u32>,
}

// DrawIndexedIndirect: [index_count, instance_count, first_index, base_vertex, first_instance]
struct IndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: u32,
    first_instance: u32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<GpuParticle>;
@group(0) @binding(1) var<storage, read_write> instances: array<InstanceData>;
@group(0) @binding(2) var<storage, read_write> counters: Counters;
@group(0) @binding(3) var<uniform> emitter: EmitterParams;
@group(0) @binding(4) var<storage, read_write> indirect: IndirectArgs;

// Simple hash-based PRNG.
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / 4294967295.0;
}

fn rand_range(seed: u32, lo: f32, hi: f32) -> f32 {
    return lo + rand_f32(seed) * (hi - lo);
}

@compute @workgroup_size(64)
fn update_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let count = arrayLength(&particles);
    if idx >= count {
        return;
    }

    var p = particles[idx];
    let dt = emitter.dt;
    let alive = p.age < p.lifetime && p.lifetime > 0.0;

    if alive {
        // Integrate.
        p.velocity += emitter.gravity * dt;
        p.position += p.velocity * dt;
        p.age += dt;

        if p.age < p.lifetime {
            // Still alive — write instance data.
            let t = p.age / p.lifetime;
            let color = mix(p.color_start, p.color_end, vec4<f32>(t));
            let size = mix(emitter.size_start, emitter.size_end, t);

            let out_idx = atomicAdd(&counters.alive_count, 1u);
            instances[out_idx] = InstanceData(
                vec4<f32>(size, 0.0, 0.0, 0.0),
                vec4<f32>(0.0, size, 0.0, 0.0),
                vec4<f32>(0.0, 0.0, size, 0.0),
                vec4<f32>(p.position, 1.0),
                color,
                vec4<f32>(0.0),
            );
        }

        particles[idx] = p;
    } else {
        // Dead slot — try to claim a spawn.
        let prev = atomicSub(&counters.remaining_spawns, 1u);
        if prev > 0u {
            // Spawn new particle.
            let s = emitter.seed + idx * 7u + prev * 13u;
            let vx = rand_range(s, emitter.velocity_min.x, emitter.velocity_max.x);
            let vy = rand_range(s + 1u, emitter.velocity_min.y, emitter.velocity_max.y);
            let vz = rand_range(s + 2u, emitter.velocity_min.z, emitter.velocity_max.z);
            let lt = rand_range(s + 3u, emitter.lifetime_min, emitter.lifetime_max);

            p.position = emitter.origin;
            p.velocity = vec3<f32>(vx, vy, vz);
            p.age = 0.0;
            p.lifetime = lt;
            p.color_start = emitter.color_start;
            p.color_end = emitter.color_end;
            particles[idx] = p;

            // Newly spawned particle is alive — write instance.
            let size = emitter.size_start;
            let out_idx = atomicAdd(&counters.alive_count, 1u);
            instances[out_idx] = InstanceData(
                vec4<f32>(size, 0.0, 0.0, 0.0),
                vec4<f32>(0.0, size, 0.0, 0.0),
                vec4<f32>(0.0, 0.0, size, 0.0),
                vec4<f32>(p.position, 1.0),
                emitter.color_start,
                vec4<f32>(0.0),
            );
        } else {
            // No spawns left, undo the subtract (we went negative).
            atomicAdd(&counters.remaining_spawns, 1u);
        }
    }
}

@compute @workgroup_size(1)
fn finalize_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let alive = atomicLoad(&counters.alive_count);
    indirect.instance_count = alive;
}
"#;
