//! Mesh data — CPU-side geometry, GPU-resident buffers, and mega-buffer.

use wgpu::util::DeviceExt;

use super::bounds::Aabb;
use super::vertex::Vertex3D;

/// CPU-side mesh geometry ready for GPU upload.
pub struct MeshData {
    /// Vertex data.
    pub vertices: Vec<Vertex3D>,
    /// Triangle indices (3 per triangle).
    pub indices: Vec<u32>,
}

impl MeshData {
    /// Create from raw vertices and indices.
    pub fn new(vertices: Vec<Vertex3D>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    /// Compute the axis-aligned bounding box of this mesh's vertices.
    pub fn compute_aabb(&self) -> Aabb {
        let positions: Vec<glam::Vec3> = self
            .vertices
            .iter()
            .map(|v| glam::Vec3::new(v.position[0], v.position[1], v.position[2]))
            .collect();
        Aabb::from_points(&positions)
    }

    /// Generate a unit cube (side length 1, centered at origin).
    ///
    /// 24 vertices (4 per face for correct normals), 36 indices.
    pub fn cube(size: f32) -> Self {
        let h = size * 0.5;

        type CubeFace = ([f32; 3], [[f32; 3]; 4], [[f32; 2]; 4]);
        let faces: [CubeFace; 6] = [
            // +X
            (
                [1.0, 0.0, 0.0],
                [[h, -h, -h], [h, h, -h], [h, h, h], [h, -h, h]],
                [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            ),
            // -X
            (
                [-1.0, 0.0, 0.0],
                [[-h, -h, h], [-h, h, h], [-h, h, -h], [-h, -h, -h]],
                [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            ),
            // +Y
            (
                [0.0, 1.0, 0.0],
                [[-h, h, -h], [-h, h, h], [h, h, h], [h, h, -h]],
                [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            ),
            // -Y
            (
                [0.0, -1.0, 0.0],
                [[-h, -h, h], [-h, -h, -h], [h, -h, -h], [h, -h, h]],
                [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            ),
            // +Z
            (
                [0.0, 0.0, 1.0],
                [[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]],
                [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            ),
            // -Z
            (
                [0.0, 0.0, -1.0],
                [[h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h]],
                [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            ),
        ];

        let mut vertices = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);

        for (normal, positions, uvs) in &faces {
            let base = vertices.len() as u32;
            for i in 0..4 {
                vertices.push(Vertex3D::new(positions[i], *normal, uvs[i]));
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        Self { vertices, indices }
    }

    /// Generate a UV sphere centered at origin.
    ///
    /// `segments` = longitude divisions (around Y), `rings` = latitude divisions.
    /// Minimum 3 segments and 2 rings.
    pub fn sphere(radius: f32, segments: u32, rings: u32) -> Self {
        let segments = segments.max(3);
        let rings = rings.max(2);

        let mut vertices = Vec::with_capacity(((segments + 1) * (rings + 1)) as usize);
        let mut indices = Vec::with_capacity((segments * rings * 6) as usize);

        for ring in 0..=rings {
            let phi = std::f32::consts::PI * ring as f32 / rings as f32;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            for seg in 0..=segments {
                let theta = std::f32::consts::TAU * seg as f32 / segments as f32;
                let sin_theta = theta.sin();
                let cos_theta = theta.cos();

                let nx = sin_phi * cos_theta;
                let ny = cos_phi;
                let nz = sin_phi * sin_theta;

                let u = seg as f32 / segments as f32;
                let v = ring as f32 / rings as f32;

                vertices.push(Vertex3D::new(
                    [radius * nx, radius * ny, radius * nz],
                    [nx, ny, nz],
                    [u, v],
                ));
            }
        }

        let stride = segments + 1;
        for ring in 0..rings {
            for seg in 0..segments {
                let tl = ring * stride + seg;
                let tr = tl + 1;
                let bl = tl + stride;
                let br = bl + 1;

                if ring != 0 {
                    indices.extend_from_slice(&[tl, tr, bl]);
                }
                if ring != rings - 1 {
                    indices.extend_from_slice(&[tr, br, bl]);
                }
            }
        }

        Self { vertices, indices }
    }

    /// Generate a flat plane in the XZ plane (Y = 0), centered at origin.
    pub fn plane(width: f32, depth: f32, subdivisions: u32) -> Self {
        let subdivisions = subdivisions.max(1);
        let hw = width * 0.5;
        let hd = depth * 0.5;

        let cols = subdivisions + 1;
        let rows = subdivisions + 1;

        let mut vertices = Vec::with_capacity((cols * rows) as usize);
        let mut indices = Vec::with_capacity((subdivisions * subdivisions * 6) as usize);

        for row in 0..rows {
            for col in 0..cols {
                let u = col as f32 / subdivisions as f32;
                let v = row as f32 / subdivisions as f32;
                let x = -hw + u * width;
                let z = -hd + v * depth;
                vertices.push(Vertex3D::new([x, 0.0, z], [0.0, 1.0, 0.0], [u, v]));
            }
        }

        for row in 0..subdivisions {
            for col in 0..subdivisions {
                let tl = row * cols + col;
                let tr = tl + 1;
                let bl = tl + cols;
                let br = bl + 1;
                indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
            }
        }

        Self { vertices, indices }
    }

    /// Generate a closed cylinder centered at origin, aligned along Y axis.
    pub fn cylinder(radius: f32, height: f32, segments: u32) -> Self {
        let segments = segments.max(3);
        let half_h = height * 0.5;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Tube.
        for i in 0..=segments {
            let angle = std::f32::consts::TAU * i as f32 / segments as f32;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let u = i as f32 / segments as f32;

            vertices.push(Vertex3D::new(
                [radius * cos_a, -half_h, radius * sin_a],
                [cos_a, 0.0, sin_a],
                [u, 1.0],
            ));
            vertices.push(Vertex3D::new(
                [radius * cos_a, half_h, radius * sin_a],
                [cos_a, 0.0, sin_a],
                [u, 0.0],
            ));
        }

        for i in 0..segments {
            let bl = i * 2;
            let br = (i + 1) * 2;
            let tl = bl + 1;
            let tr = br + 1;
            indices.extend_from_slice(&[bl, tl, br, tl, tr, br]);
        }

        // Top cap.
        let top_center = vertices.len() as u32;
        vertices.push(Vertex3D::new(
            [0.0, half_h, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5],
        ));
        for i in 0..=segments {
            let angle = std::f32::consts::TAU * i as f32 / segments as f32;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            vertices.push(Vertex3D::new(
                [radius * cos_a, half_h, radius * sin_a],
                [0.0, 1.0, 0.0],
                [0.5 + 0.5 * cos_a, 0.5 - 0.5 * sin_a],
            ));
        }
        for i in 0..segments {
            let rim = top_center + 1 + i;
            indices.extend_from_slice(&[top_center, rim + 1, rim]);
        }

        // Bottom cap.
        let bot_center = vertices.len() as u32;
        vertices.push(Vertex3D::new(
            [0.0, -half_h, 0.0],
            [0.0, -1.0, 0.0],
            [0.5, 0.5],
        ));
        for i in 0..=segments {
            let angle = std::f32::consts::TAU * i as f32 / segments as f32;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            vertices.push(Vertex3D::new(
                [radius * cos_a, -half_h, radius * sin_a],
                [0.0, -1.0, 0.0],
                [0.5 + 0.5 * cos_a, 0.5 + 0.5 * sin_a],
            ));
        }
        for i in 0..segments {
            let rim = bot_center + 1 + i;
            indices.extend_from_slice(&[bot_center, rim, rim + 1]);
        }

        Self { vertices, indices }
    }

    /// Generate a cone centered at origin, aligned along Y axis.
    ///
    /// Base at -height/2, apex at +height/2. Includes base cap.
    pub fn cone(radius: f32, height: f32, segments: u32) -> Self {
        let segments = segments.max(3);
        let half_h = height * 0.5;

        let slope = radius / height;
        let ny = 1.0 / (1.0 + slope * slope).sqrt();
        let nxz = slope * ny;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Tube (cone surface).
        for i in 0..=segments {
            let angle = std::f32::consts::TAU * i as f32 / segments as f32;
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            vertices.push(Vertex3D::new(
                [radius * cos_a, -half_h, radius * sin_a],
                [nxz * cos_a, ny, nxz * sin_a],
                [i as f32 / segments as f32, 1.0],
            ));
            vertices.push(Vertex3D::new(
                [0.0, half_h, 0.0],
                [nxz * cos_a, ny, nxz * sin_a],
                [i as f32 / segments as f32, 0.0],
            ));
        }

        for i in 0..segments {
            let base = i * 2;
            let next_base = (i + 1) * 2;
            let apex = base + 1;
            indices.extend_from_slice(&[base, apex, next_base]);
        }

        // Base cap.
        let bot_center = vertices.len() as u32;
        vertices.push(Vertex3D::new(
            [0.0, -half_h, 0.0],
            [0.0, -1.0, 0.0],
            [0.5, 0.5],
        ));
        for i in 0..=segments {
            let angle = std::f32::consts::TAU * i as f32 / segments as f32;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            vertices.push(Vertex3D::new(
                [radius * cos_a, -half_h, radius * sin_a],
                [0.0, -1.0, 0.0],
                [0.5 + 0.5 * cos_a, 0.5 + 0.5 * sin_a],
            ));
        }
        for i in 0..segments {
            let rim = bot_center + 1 + i;
            indices.extend_from_slice(&[bot_center, rim, rim + 1]);
        }

        Self { vertices, indices }
    }

    /// Generate a torus centered at origin, lying in the XZ plane.
    ///
    /// `major_radius` — distance from center to tube center.
    /// `minor_radius` — tube radius.
    pub fn torus(
        major_radius: f32,
        minor_radius: f32,
        major_segments: u32,
        minor_segments: u32,
    ) -> Self {
        let major_segments = major_segments.max(3);
        let minor_segments = minor_segments.max(3);

        let vert_count = ((major_segments + 1) * (minor_segments + 1)) as usize;
        let idx_count = (major_segments * minor_segments * 6) as usize;
        let mut vertices = Vec::with_capacity(vert_count);
        let mut indices = Vec::with_capacity(idx_count);

        for i in 0..=major_segments {
            let u = i as f32 / major_segments as f32;
            let theta = std::f32::consts::TAU * u;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            for j in 0..=minor_segments {
                let v = j as f32 / minor_segments as f32;
                let phi = std::f32::consts::TAU * v;
                let cos_phi = phi.cos();
                let sin_phi = phi.sin();

                let x = (major_radius + minor_radius * cos_phi) * cos_theta;
                let y = minor_radius * sin_phi;
                let z = (major_radius + minor_radius * cos_phi) * sin_theta;

                let nx = cos_phi * cos_theta;
                let ny = sin_phi;
                let nz = cos_phi * sin_theta;

                vertices.push(Vertex3D::new([x, y, z], [nx, ny, nz], [u, v]));
            }
        }

        let stride = minor_segments + 1;
        for i in 0..major_segments {
            for j in 0..minor_segments {
                let tl = i * stride + j;
                let tr = tl + 1;
                let bl = tl + stride;
                let br = bl + 1;
                indices.extend_from_slice(&[tl, tr, bl, tr, br, bl]);
            }
        }

        Self { vertices, indices }
    }
}

/// Handle to a GPU-resident mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub(crate) u32);

/// GPU-resident mesh (vertex buffer + index buffer).
pub(crate) struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

impl Mesh {
    /// Upload [`MeshData`] to the GPU.
    pub fn upload(device: &wgpu::Device, data: &MeshData) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("esox_3d_vertex_buffer"),
            contents: bytemuck::cast_slice(&data.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("esox_3d_index_buffer"),
            contents: bytemuck::cast_slice(&data.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            index_count: data.indices.len() as u32,
        }
    }
}

/// A mesh's location within the shared mega-buffers.
#[allow(dead_code)]
pub(crate) struct MeshRegion {
    /// Offset in the mega vertex buffer (in vertices, not bytes).
    pub vertex_offset: u32,
    /// Number of vertices.
    pub vertex_count: u32,
    /// Offset in the mega index buffer (in indices, not bytes).
    pub index_offset: u32,
    /// Number of indices.
    pub index_count: u32,
    /// Object-space AABB for culling.
    pub aabb: Aabb,
}

/// Initial mega-buffer vertex capacity (64K vertices ≈ 4MB at 64 bytes/vertex).
const INITIAL_VERTEX_CAPACITY: u32 = 65_536;

/// Initial mega-buffer index capacity (256K indices ≈ 1MB at 4 bytes/index).
const INITIAL_INDEX_CAPACITY: u32 = 262_144;

/// Shared vertex and index buffers for all 3D meshes.
///
/// Eliminates per-mesh buffer switches — bind once per frame instead of per draw.
pub(crate) struct MegaBuffer {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub vertex_capacity: u32,
    pub index_capacity: u32,
    pub vertex_used: u32,
    pub index_used: u32,
}

impl MegaBuffer {
    /// Create a new mega-buffer with default initial sizes.
    pub fn new(device: &wgpu::Device) -> Self {
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_3d_mega_vertex"),
            size: INITIAL_VERTEX_CAPACITY as u64 * size_of::<Vertex3D>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_3d_mega_index"),
            size: INITIAL_INDEX_CAPACITY as u64 * 4,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            vertex_buffer,
            index_buffer,
            vertex_capacity: INITIAL_VERTEX_CAPACITY,
            index_capacity: INITIAL_INDEX_CAPACITY,
            vertex_used: 0,
            index_used: 0,
        }
    }

    /// Append mesh data and return a `MeshRegion` describing its location.
    ///
    /// Grows buffers by power-of-two with GPU copy if needed.
    pub fn append(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        data: &MeshData,
        aabb: Aabb,
    ) -> MeshRegion {
        let vert_count = data.vertices.len() as u32;
        let idx_count = data.indices.len() as u32;

        // Grow vertex buffer if needed.
        if self.vertex_used + vert_count > self.vertex_capacity {
            let new_cap = ((self.vertex_used + vert_count) as u64)
                .next_power_of_two()
                .max(INITIAL_VERTEX_CAPACITY as u64) as u32;
            let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("esox_3d_mega_vertex"),
                size: new_cap as u64 * size_of::<Vertex3D>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            if self.vertex_used > 0 {
                encoder.copy_buffer_to_buffer(
                    &self.vertex_buffer,
                    0,
                    &new_buffer,
                    0,
                    self.vertex_used as u64 * size_of::<Vertex3D>() as u64,
                );
            }
            self.vertex_buffer = new_buffer;
            self.vertex_capacity = new_cap;
            tracing::debug!("grew mega vertex buffer to {new_cap} vertices");
        }

        // Grow index buffer if needed.
        if self.index_used + idx_count > self.index_capacity {
            let new_cap = ((self.index_used + idx_count) as u64)
                .next_power_of_two()
                .max(INITIAL_INDEX_CAPACITY as u64) as u32;
            let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("esox_3d_mega_index"),
                size: new_cap as u64 * 4,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            if self.index_used > 0 {
                encoder.copy_buffer_to_buffer(
                    &self.index_buffer,
                    0,
                    &new_buffer,
                    0,
                    self.index_used as u64 * 4,
                );
            }
            self.index_buffer = new_buffer;
            self.index_capacity = new_cap;
            tracing::debug!("grew mega index buffer to {new_cap} indices");
        }

        let region = MeshRegion {
            vertex_offset: self.vertex_used,
            vertex_count: vert_count,
            index_offset: self.index_used,
            index_count: idx_count,
            aabb,
        };

        // Write data via queue.
        queue.write_buffer(
            &self.vertex_buffer,
            self.vertex_used as u64 * size_of::<Vertex3D>() as u64,
            bytemuck::cast_slice(&data.vertices),
        );
        queue.write_buffer(
            &self.index_buffer,
            self.index_used as u64 * 4,
            bytemuck::cast_slice(&data.indices),
        );

        self.vertex_used += vert_count;
        self.index_used += idx_count;

        region
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_geometry() {
        let cube = MeshData::cube(1.0);
        assert_eq!(cube.vertices.len(), 24);
        assert_eq!(cube.indices.len(), 36);
        assert!(cube.indices.iter().all(|&i| i < 24));
    }

    #[test]
    fn cube_normals_are_unit() {
        let cube = MeshData::cube(2.0);
        for v in &cube.vertices {
            let len = (v.normal[0].powi(2) + v.normal[1].powi(2) + v.normal[2].powi(2)).sqrt();
            assert!((len - 1.0).abs() < 1e-6, "normal length {len}");
        }
    }

    #[test]
    fn sphere_geometry() {
        let sphere = MeshData::sphere(1.0, 16, 8);
        assert!(!sphere.vertices.is_empty());
        assert!(!sphere.indices.is_empty());
        let max_idx = sphere.vertices.len() as u32;
        assert!(sphere.indices.iter().all(|&i| i < max_idx));
    }

    #[test]
    fn sphere_normals_are_unit() {
        let sphere = MeshData::sphere(1.0, 16, 8);
        for v in &sphere.vertices {
            let len = (v.normal[0].powi(2) + v.normal[1].powi(2) + v.normal[2].powi(2)).sqrt();
            assert!((len - 1.0).abs() < 1e-6, "normal length {len}");
        }
    }

    #[test]
    fn sphere_vertices_on_surface() {
        let r = 2.5;
        let sphere = MeshData::sphere(r, 32, 16);
        for v in &sphere.vertices {
            let dist =
                (v.position[0].powi(2) + v.position[1].powi(2) + v.position[2].powi(2)).sqrt();
            assert!(
                (dist - r).abs() < 1e-5,
                "vertex distance {dist} != radius {r}"
            );
        }
    }

    #[test]
    fn plane_geometry() {
        let plane = MeshData::plane(2.0, 2.0, 4);
        assert_eq!(plane.vertices.len(), 25);
        assert_eq!(plane.indices.len(), 96);
    }

    #[test]
    fn plane_is_flat() {
        let plane = MeshData::plane(10.0, 10.0, 2);
        for v in &plane.vertices {
            assert!((v.position[1]).abs() < 1e-6, "Y should be 0");
            assert_eq!(v.normal, [0.0, 1.0, 0.0]);
        }
    }

    #[test]
    fn minimum_sphere_segments() {
        let sphere = MeshData::sphere(1.0, 1, 1);
        assert!(!sphere.vertices.is_empty());
        assert!(!sphere.indices.is_empty());
    }

    #[test]
    fn cylinder_geometry() {
        let cyl = MeshData::cylinder(1.0, 2.0, 16);
        assert!(!cyl.vertices.is_empty());
        assert!(!cyl.indices.is_empty());
        let max_idx = cyl.vertices.len() as u32;
        assert!(cyl.indices.iter().all(|&i| i < max_idx));
    }

    #[test]
    fn cylinder_normals_are_unit() {
        let cyl = MeshData::cylinder(1.0, 2.0, 8);
        for v in &cyl.vertices {
            let len = (v.normal[0].powi(2) + v.normal[1].powi(2) + v.normal[2].powi(2)).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "normal length {len}");
        }
    }

    #[test]
    fn cylinder_minimum_segments() {
        let cyl = MeshData::cylinder(1.0, 1.0, 1);
        assert!(!cyl.vertices.is_empty());
        assert!(!cyl.indices.is_empty());
        let max_idx = cyl.vertices.len() as u32;
        assert!(cyl.indices.iter().all(|&i| i < max_idx));
    }

    #[test]
    fn cone_geometry() {
        let cone = MeshData::cone(1.0, 2.0, 16);
        assert!(!cone.vertices.is_empty());
        assert!(!cone.indices.is_empty());
        let max_idx = cone.vertices.len() as u32;
        assert!(cone.indices.iter().all(|&i| i < max_idx));
    }

    #[test]
    fn cone_normals_are_unit() {
        let cone = MeshData::cone(1.0, 2.0, 8);
        for v in &cone.vertices {
            let len = (v.normal[0].powi(2) + v.normal[1].powi(2) + v.normal[2].powi(2)).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "normal length {len}");
        }
    }

    #[test]
    fn cone_minimum_segments() {
        let cone = MeshData::cone(1.0, 1.0, 1);
        assert!(!cone.vertices.is_empty());
        assert!(!cone.indices.is_empty());
        let max_idx = cone.vertices.len() as u32;
        assert!(cone.indices.iter().all(|&i| i < max_idx));
    }

    #[test]
    fn torus_geometry() {
        let torus = MeshData::torus(2.0, 0.5, 16, 8);
        let expected_verts = (16 + 1) * (8 + 1);
        assert_eq!(torus.vertices.len(), expected_verts);
        let expected_indices = 16 * 8 * 6;
        assert_eq!(torus.indices.len(), expected_indices);
        let max_idx = torus.vertices.len() as u32;
        assert!(torus.indices.iter().all(|&i| i < max_idx));
    }

    #[test]
    fn torus_normals_are_unit() {
        let torus = MeshData::torus(2.0, 0.5, 12, 6);
        for v in &torus.vertices {
            let len = (v.normal[0].powi(2) + v.normal[1].powi(2) + v.normal[2].powi(2)).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "normal length {len}");
        }
    }

    #[test]
    fn torus_minimum_segments() {
        let torus = MeshData::torus(1.0, 0.3, 1, 1);
        assert!(!torus.vertices.is_empty());
        assert!(!torus.indices.is_empty());
        let max_idx = torus.vertices.len() as u32;
        assert!(torus.indices.iter().all(|&i| i < max_idx));
    }

    #[test]
    fn compute_aabb_cube() {
        let cube = MeshData::cube(2.0);
        let aabb = cube.compute_aabb();
        assert!((aabb.min.x - (-1.0)).abs() < 1e-5);
        assert!((aabb.max.x - 1.0).abs() < 1e-5);
        assert!((aabb.min.y - (-1.0)).abs() < 1e-5);
        assert!((aabb.max.y - 1.0).abs() < 1e-5);
    }

    #[test]
    fn compute_aabb_sphere() {
        let sphere = MeshData::sphere(3.0, 32, 16);
        let aabb = sphere.compute_aabb();
        assert!((aabb.min.x - (-3.0)).abs() < 0.01);
        assert!((aabb.max.x - 3.0).abs() < 0.01);
    }
}
