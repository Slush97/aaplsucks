//! GPU vertex format and buffer layout descriptors for 3D meshes.

/// A 3D vertex with position, normal, texture coordinates, per-vertex color, and tangent.
///
/// 64 bytes, 4-byte aligned. Suitable for both loaded and procedural meshes.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Vertex3D {
    /// Object-space position.
    pub position: [f32; 3],
    /// Object-space normal (should be unit length).
    pub normal: [f32; 3],
    /// Texture coordinates.
    pub uv: [f32; 2],
    /// Per-vertex RGBA color (linear, premultiplied alpha).
    pub color: [f32; 4],
    /// Tangent vector (xyz = tangent direction, w = bitangent handedness ±1).
    /// Zero tangent signals the shader to skip normal mapping.
    pub tangent: [f32; 4],
}

impl Vertex3D {
    /// Create a vertex with the given position, normal, white color, and zero tangent.
    pub fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> Self {
        Self {
            position,
            normal,
            uv,
            color: [1.0, 1.0, 1.0, 1.0],
            tangent: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create a vertex with explicit color.
    pub fn with_color(
        position: [f32; 3],
        normal: [f32; 3],
        uv: [f32; 2],
        color: [f32; 4],
    ) -> Self {
        Self {
            position,
            normal,
            uv,
            color,
            tangent: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create a vertex with explicit tangent.
    pub fn with_tangent(
        position: [f32; 3],
        normal: [f32; 3],
        uv: [f32; 2],
        tangent: [f32; 4],
    ) -> Self {
        Self {
            position,
            normal,
            uv,
            color: [1.0, 1.0, 1.0, 1.0],
            tangent,
        }
    }
}

/// Vertex buffer attributes for [`Vertex3D`] (locations 0–4).
pub(crate) const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 5] = [
    wgpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: wgpu::VertexFormat::Float32x3,
    },
    wgpu::VertexAttribute {
        offset: 12,
        shader_location: 1,
        format: wgpu::VertexFormat::Float32x3,
    },
    wgpu::VertexAttribute {
        offset: 24,
        shader_location: 2,
        format: wgpu::VertexFormat::Float32x2,
    },
    wgpu::VertexAttribute {
        offset: 32,
        shader_location: 3,
        format: wgpu::VertexFormat::Float32x4,
    },
    wgpu::VertexAttribute {
        offset: 48,
        shader_location: 4,
        format: wgpu::VertexFormat::Float32x4,
    },
];

/// Vertex buffer layout for [`Vertex3D`] (slot 0, per-vertex).
pub(crate) fn vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: size_of::<Vertex3D>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &VERTEX_ATTRIBUTES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_is_64_bytes() {
        assert_eq!(size_of::<Vertex3D>(), 64);
    }

    #[test]
    fn vertex_is_4_byte_aligned() {
        assert_eq!(align_of::<Vertex3D>(), 4);
    }

    #[test]
    fn default_tangent_is_zero_with_positive_handedness() {
        let v = Vertex3D::new([0.0; 3], [0.0, 1.0, 0.0], [0.0; 2]);
        assert_eq!(v.tangent, [0.0, 0.0, 0.0, 1.0]);
    }
}
