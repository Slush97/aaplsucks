//! Per-instance GPU data for instanced mesh drawing.

use super::transform::Transform;

/// Per-instance data uploaded to the GPU for instanced drawing.
///
/// 96 bytes: model matrix (64) + color (16) + params (16).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct InstanceData {
    /// Column-major 4x4 model matrix.
    pub model: [[f32; 4]; 4],
    /// Instance color/tint (multiplied with vertex color in the shader).
    pub color: [f32; 4],
    /// User-defined shader parameters.
    pub params: [f32; 4],
}

impl InstanceData {
    /// Create an instance from a transform with white color and zero params.
    pub fn from_transform(transform: &Transform) -> Self {
        Self {
            model: transform.matrix_cols(),
            color: [1.0, 1.0, 1.0, 1.0],
            params: [0.0; 4],
        }
    }

    /// Create an instance with a transform and color.
    pub fn with_color(transform: &Transform, color: [f32; 4]) -> Self {
        Self {
            model: transform.matrix_cols(),
            color,
            params: [0.0; 4],
        }
    }

    /// Create an instance with transform, color, and custom params.
    pub fn new(transform: &Transform, color: [f32; 4], params: [f32; 4]) -> Self {
        Self {
            model: transform.matrix_cols(),
            color,
            params,
        }
    }
}

/// Instance buffer attributes (locations 5–10).
pub(crate) const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 6] = [
    // model column 0
    wgpu::VertexAttribute {
        offset: 0,
        shader_location: 5,
        format: wgpu::VertexFormat::Float32x4,
    },
    // model column 1
    wgpu::VertexAttribute {
        offset: 16,
        shader_location: 6,
        format: wgpu::VertexFormat::Float32x4,
    },
    // model column 2
    wgpu::VertexAttribute {
        offset: 32,
        shader_location: 7,
        format: wgpu::VertexFormat::Float32x4,
    },
    // model column 3
    wgpu::VertexAttribute {
        offset: 48,
        shader_location: 8,
        format: wgpu::VertexFormat::Float32x4,
    },
    // color
    wgpu::VertexAttribute {
        offset: 64,
        shader_location: 9,
        format: wgpu::VertexFormat::Float32x4,
    },
    // params
    wgpu::VertexAttribute {
        offset: 80,
        shader_location: 10,
        format: wgpu::VertexFormat::Float32x4,
    },
];

/// Vertex buffer layout for [`InstanceData`] (slot 1, per-instance).
pub(crate) fn instance_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: size_of::<InstanceData>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &INSTANCE_ATTRIBUTES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_data_is_96_bytes() {
        assert_eq!(size_of::<InstanceData>(), 96);
    }

    #[test]
    fn instance_data_is_4_byte_aligned() {
        assert_eq!(align_of::<InstanceData>(), 4);
    }

    #[test]
    fn from_identity_transform() {
        let inst = InstanceData::from_transform(&Transform::IDENTITY);
        assert_eq!(inst.model[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(inst.model[1], [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(inst.model[2], [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(inst.model[3], [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(inst.color, [1.0, 1.0, 1.0, 1.0]);
    }
}
