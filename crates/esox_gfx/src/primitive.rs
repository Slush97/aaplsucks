use crate::color::Color;

/// Axis-aligned rectangle in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    /// X position of the top-left corner.
    pub x: f32,
    /// Y position of the top-left corner.
    pub y: f32,
    /// Width of the rectangle.
    pub width: f32,
    /// Height of the rectangle.
    pub height: f32,
}

/// UV-space rectangle for texture sampling.
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct UvRect {
    /// Left U coordinate.
    pub u0: f32,
    /// Top V coordinate.
    pub v0: f32,
    /// Right U coordinate.
    pub u1: f32,
    /// Bottom V coordinate.
    pub v1: f32,
}

impl UvRect {
    /// A zero-area UV rect (used for untextured quads).
    pub const ZERO: Self = Self {
        u0: 0.0,
        v0: 0.0,
        u1: 0.0,
        v1: 0.0,
    };
}

/// Per-corner border radius.
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BorderRadius {
    /// Top-left corner radius.
    pub top_left: f32,
    /// Top-right corner radius.
    pub top_right: f32,
    /// Bottom-left corner radius.
    pub bottom_left: f32,
    /// Bottom-right corner radius.
    pub bottom_right: f32,
}

impl BorderRadius {
    /// No rounding on any corner.
    pub const ZERO: Self = Self {
        top_left: 0.0,
        top_right: 0.0,
        bottom_left: 0.0,
        bottom_right: 0.0,
    };

    /// Uniform radius on all corners.
    pub const fn uniform(r: f32) -> Self {
        Self {
            top_left: r,
            top_right: r,
            bottom_left: r,
            bottom_right: r,
        }
    }
}

/// Pipeline ID for 2D SDF shapes (Rect, Circle, Ellipse, etc.).
pub const PIPELINE_SDF_2D: ShaderId = ShaderId(0);
/// Pipeline ID for textured/glyph quads (95%+ of terminal pixels).
pub const PIPELINE_TEXT: ShaderId = ShaderId(1);
/// Pipeline ID for raymarched 3D shapes (Sphere3D, Torus3D, etc.).
pub const PIPELINE_3D: ShaderId = ShaderId(2);
/// Pipeline ID for opaque 2D SDF shapes (no alpha blending — cell backgrounds).
pub const PIPELINE_SDF_2D_OPAQUE: ShaderId = ShaderId(3);
/// Pipeline ID for additive-blend 2D SDF shapes (src + dst — glowing particles).
pub const PIPELINE_SDF_2D_ADDITIVE: ShaderId = ShaderId(4);
/// Pipeline ID for screen-blend 2D SDF shapes (1 - (1-src)*(1-dst) — brightens).
pub const PIPELINE_SDF_2D_SCREEN: ShaderId = ShaderId(5);
/// Pipeline ID for multiply-blend 2D SDF shapes (src * dst — darkens).
pub const PIPELINE_SDF_2D_MULTIPLY: ShaderId = ShaderId(6);

/// Offset added to scene pipeline IDs to get the non-MSAA (sample_count=1)
/// variant. Used when the 2D render pass falls back to sample_count=1 (e.g.
/// compositing 2D UI on top of a 3D pre-render pass).
pub const NO_MSAA_PIPELINE_OFFSET: u32 = 50;

/// Minimum shader ID for user-registered pipelines. IDs 0–99 are reserved for
/// built-in pipelines.
pub const USER_SHADER_ID_MIN: u32 = 10;

/// Identifier for a user-registered shader pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderId(pub u32);

/// Blend mode for shape rendering.
///
/// Each variant maps to a distinct GPU pipeline with a different
/// [`wgpu::BlendState`]. The mode is encoded into bits 1–2 of `flags[3]`
/// for non-Shader SDF shapes (bit 0 is the rainbow flag).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum BlendMode {
    /// Standard premultiplied alpha blending (src over dst).
    #[default]
    Normal = 0,
    /// Additive blending (src + dst). Good for glowing particles.
    Additive = 1,
    /// Screen blending (1 - (1-src)*(1-dst)). Brightens without washing out.
    Screen = 2,
    /// Multiply blending (src * dst). Darkens.
    Multiply = 3,
}

impl BlendMode {
    /// Decode blend mode from the 2-bit field (bits 1–2 of `flags[3]`).
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0x3 {
            0 => Self::Normal,
            1 => Self::Additive,
            2 => Self::Screen,
            3 => Self::Multiply,
            _ => unreachable!(),
        }
    }

    /// Encode as a 2-bit value for storage in `flags[3]`.
    pub fn to_bits(self) -> u8 {
        self as u8
    }

    /// Return the SDF pipeline ID for this blend mode.
    pub fn sdf_pipeline_id(self) -> u32 {
        match self {
            Self::Normal => PIPELINE_SDF_2D.0,
            Self::Additive => PIPELINE_SDF_2D_ADDITIVE.0,
            Self::Screen => PIPELINE_SDF_2D_SCREEN.0,
            Self::Multiply => PIPELINE_SDF_2D_MULTIPLY.0,
        }
    }
}

/// User-supplied parameters passed to a custom shader (16 floats).
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ShaderParams {
    /// Arbitrary float parameters for the shader.
    pub values: [f32; 16],
}

/// SDF shape type encoded as a `u32` for the GPU uber-shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ShapeType {
    /// Rounded rectangle (default).
    Rect = 0,
    /// Circle (uses `sdf_params[0]` as radius).
    Circle = 1,
    /// Ellipse (uses `sdf_params[0..2]` as rx, ry).
    Ellipse = 2,
    /// Ring / annulus (uses `sdf_params[0..2]` as outer_r, inner_r).
    Ring = 3,
    /// Line / capsule (uses `sdf_params[0..4]` as x1, y1, x2, y2).
    Line = 4,
    /// Circular arc.
    Arc = 5,
    /// Equilateral triangle SDF.
    Triangle = 6,
    /// Textured quad (atlas sample, no SDF).
    Textured = 7,
    /// Custom shader pipeline.
    Shader = 8,
    /// Raymarched 3D sphere (sdf_params: [radius, 0, 0, 0]).
    Sphere3D = 9,
    /// Raymarched 3D torus (sdf_params: [major_radius, minor_radius, 0, 0]).
    Torus3D = 10,
    /// Raymarched 3D rounded box (sdf_params: [half_x, half_y, half_z, corner_radius]).
    RoundedBox3D = 11,
    /// Regular N-gon (sdf_params: [sides, 0, 0, 0]; border_radius[0] = corner rounding).
    Polygon = 12,
    /// N-pointed star (sdf_params: [points, inner_r, outer_r, 0]).
    Star = 13,
    /// Pie/sector slice (sdf_params: [radius, angle_start, angle_sweep, 0]).
    Sector = 14,
    /// Stadium/capsule from rect dimensions.
    Capsule = 15,
    /// Axis-aligned cross/plus sign (sdf_params: [arm_width, arm_length, 0, 0]).
    Cross = 16,
    /// Quadratic bezier stroke (sdf_params: [cx, cy, 0, 0]; extra: [x2, y2, 0, 0]).
    Bezier = 17,
    /// Triangle with 3 arbitrary vertices (sdf_params: [x1, y1, x2, y2]; extra: [x3, y3, 0, 0]).
    ArbitraryTriangle = 18,
    /// Symmetric trapezoid (sdf_params: [top_half_w, bottom_half_w, half_h, 0]).
    Trapezoid = 19,
    /// Raymarched 3D sliced torus — a torus fragment clipped by two cut planes.
    ///
    /// `sdf_params`: `[major_r, minor_r, 0, 0]` (same as [`Torus3D`]).
    /// `border_radius`: `[a_min, a_max, b_min, b_max]` — dot-product acceptance
    /// ranges against two fixed cut-plane normals (shader constants).
    /// `extra[0..2]`: piece translation in object space.
    SlicedTorus3D = 20,
    /// Raymarched 3D morphing shape — blends between two SDF primitives.
    ///
    /// `sdf_params`: `[params_a0, params_a1, params_b0, params_b1]`.
    /// `border_radius`: `[morph_factor, shape_a_type, shape_b_type, 0]`.
    /// `extra`: `[extra_b0, extra_b1, extra_b2, 0]` — additional shape B params.
    Morph3D = 21,
    /// SDF curly underline (sine wave).
    ///
    /// `sdf_params`: `[frequency, amplitude, thickness, 0]`.
    UnderlineCurly = 22,
    /// SDF dotted underline (repeating circles).
    ///
    /// `sdf_params`: `[dot_radius, spacing, 0, 0]`.
    UnderlineDotted = 23,
    /// SDF dashed underline (repeating rounded rectangles).
    ///
    /// `sdf_params`: `[dash_width, gap_width, thickness, 0]`.
    UnderlineDashed = 24,
    /// 2D heart SDF (Inigo Quilez parameterization).
    ///
    /// `sdf_params`: `[scale, 0, 0, 0]`. The bounding rect is square with side = scale * 2.
    Heart = 25,
}

impl ShapeType {
    /// Return the pipeline ID for this shape type.
    ///
    /// - 2D SDF shapes → [`PIPELINE_SDF_2D`] (0)
    /// - Textured quads → [`PIPELINE_TEXT`] (1)
    /// - Raymarched 3D → [`PIPELINE_3D`] (2)
    /// - Shader → handled separately (uses `flags[3]`)
    pub fn pipeline_id(self) -> u32 {
        match self {
            Self::Textured => PIPELINE_TEXT.0,
            Self::Sphere3D
            | Self::Torus3D
            | Self::RoundedBox3D
            | Self::SlicedTorus3D
            | Self::Morph3D => PIPELINE_3D.0,
            // Shader is special-cased by the caller (pipeline ID from flags[3]).
            // All other 2D SDF shapes use pipeline 0.
            _ => PIPELINE_SDF_2D.0,
        }
    }

    /// Encode as `f32` for GPU upload.
    pub fn to_f32(self) -> f32 {
        (self as u32) as f32
    }

    /// Decode from `f32` (returns `None` if out of range).
    pub fn from_f32(v: f32) -> Option<Self> {
        if v < 0.0 || v > u32::MAX as f32 {
            return None;
        }
        let i = v as u32;
        match i {
            0 => Some(Self::Rect),
            1 => Some(Self::Circle),
            2 => Some(Self::Ellipse),
            3 => Some(Self::Ring),
            4 => Some(Self::Line),
            5 => Some(Self::Arc),
            6 => Some(Self::Triangle),
            7 => Some(Self::Textured),
            8 => Some(Self::Shader),
            9 => Some(Self::Sphere3D),
            10 => Some(Self::Torus3D),
            11 => Some(Self::RoundedBox3D),
            12 => Some(Self::Polygon),
            13 => Some(Self::Star),
            14 => Some(Self::Sector),
            15 => Some(Self::Capsule),
            16 => Some(Self::Cross),
            17 => Some(Self::Bezier),
            18 => Some(Self::ArbitraryTriangle),
            19 => Some(Self::Trapezoid),
            20 => Some(Self::SlicedTorus3D),
            21 => Some(Self::Morph3D),
            22 => Some(Self::UnderlineCurly),
            23 => Some(Self::UnderlineDotted),
            24 => Some(Self::UnderlineDashed),
            25 => Some(Self::Heart),
            _ => None,
        }
    }
}

/// A drawable primitive in the scene.
#[derive(Debug, Clone, Copy)]
pub enum Primitive {
    /// A solid-colored rectangle.
    SolidRect {
        /// Destination rectangle.
        rect: Rect,
        /// Fill color.
        color: Color,
        /// Corner radii.
        border_radius: BorderRadius,
    },
    /// A textured rectangle (atlas glyph, image, etc.).
    TexturedRect {
        /// Destination rectangle.
        rect: Rect,
        /// Source UV coordinates in the atlas.
        uv: UvRect,
        /// Tint color (multiplied with texture).
        color: Color,
        /// Atlas array layer index.
        layer: u32,
    },
    /// A rectangle rendered by a custom shader pipeline.
    ShaderRect {
        /// Destination rectangle.
        rect: Rect,
        /// Which shader pipeline to use.
        shader: ShaderId,
        /// Parameters passed to the shader.
        params: ShaderParams,
    },
    /// A filled circle.
    Circle {
        /// Center X coordinate.
        center_x: f32,
        /// Center Y coordinate.
        center_y: f32,
        /// Radius in pixels.
        radius: f32,
        /// Fill color.
        color: Color,
    },
    /// A filled ellipse.
    Ellipse {
        /// Center X coordinate.
        center_x: f32,
        /// Center Y coordinate.
        center_y: f32,
        /// Horizontal radius.
        rx: f32,
        /// Vertical radius.
        ry: f32,
        /// Fill color.
        color: Color,
    },
    /// A ring (annulus).
    Ring {
        /// Center X coordinate.
        center_x: f32,
        /// Center Y coordinate.
        center_y: f32,
        /// Outer radius.
        outer_r: f32,
        /// Inner radius.
        inner_r: f32,
        /// Fill color.
        color: Color,
    },
    /// A line segment (rendered as a capsule with thickness).
    Line {
        /// Start X.
        x1: f32,
        /// Start Y.
        y1: f32,
        /// End X.
        x2: f32,
        /// End Y.
        y2: f32,
        /// Line thickness in pixels.
        thickness: f32,
        /// Line color.
        color: Color,
    },
    /// A circular arc.
    Arc {
        /// Center X coordinate.
        center_x: f32,
        /// Center Y coordinate.
        center_y: f32,
        /// Arc radius (to the centerline of the stroke).
        radius: f32,
        /// Stroke thickness.
        thickness: f32,
        /// Start angle in radians.
        angle_start: f32,
        /// Sweep angle in radians.
        angle_sweep: f32,
        /// Fill color.
        color: Color,
    },
    /// An equilateral triangle inscribed in the given rect.
    Triangle {
        /// Bounding rectangle.
        rect: Rect,
        /// Fill color.
        color: Color,
    },
    /// A regular N-sided polygon.
    Polygon {
        /// Center X coordinate.
        center_x: f32,
        /// Center Y coordinate.
        center_y: f32,
        /// Circumscribed radius.
        radius: f32,
        /// Number of sides.
        sides: u32,
        /// Fill color.
        color: Color,
    },
    /// An N-pointed star.
    Star {
        /// Center X coordinate.
        center_x: f32,
        /// Center Y coordinate.
        center_y: f32,
        /// Number of points.
        points: u32,
        /// Inner radius.
        inner_r: f32,
        /// Outer radius.
        outer_r: f32,
        /// Fill color.
        color: Color,
    },
    /// A pie/sector slice (filled arc).
    Sector {
        /// Center X coordinate.
        center_x: f32,
        /// Center Y coordinate.
        center_y: f32,
        /// Radius.
        radius: f32,
        /// Start angle in radians.
        angle_start: f32,
        /// Sweep angle in radians.
        angle_sweep: f32,
        /// Fill color.
        color: Color,
    },
    /// A stadium/capsule shape from rect dimensions.
    Capsule {
        /// Bounding rectangle.
        rect: Rect,
        /// Fill color.
        color: Color,
    },
    /// An axis-aligned cross/plus sign.
    CrossShape {
        /// Center X coordinate.
        center_x: f32,
        /// Center Y coordinate.
        center_y: f32,
        /// Arm width.
        arm_width: f32,
        /// Arm length.
        arm_length: f32,
        /// Fill color.
        color: Color,
    },
    /// A quadratic bezier curve stroke.
    Bezier {
        /// Start X.
        x0: f32,
        /// Start Y.
        y0: f32,
        /// Control point X.
        cx: f32,
        /// Control point Y.
        cy: f32,
        /// End X.
        x1: f32,
        /// End Y.
        y1: f32,
        /// Stroke thickness.
        thickness: f32,
        /// Stroke color.
        color: Color,
    },
    /// A triangle with 3 arbitrary vertex positions.
    ArbitraryTriangle {
        /// Vertex 1 X.
        x1: f32,
        /// Vertex 1 Y.
        y1: f32,
        /// Vertex 2 X.
        x2: f32,
        /// Vertex 2 Y.
        y2: f32,
        /// Vertex 3 X.
        x3: f32,
        /// Vertex 3 Y.
        y3: f32,
        /// Fill color.
        color: Color,
    },
    /// A symmetric trapezoid.
    Trapezoid {
        /// Center X coordinate.
        center_x: f32,
        /// Center Y coordinate.
        center_y: f32,
        /// Top edge half-width.
        top_half_w: f32,
        /// Bottom edge half-width.
        bottom_half_w: f32,
        /// Half-height.
        half_h: f32,
        /// Fill color.
        color: Color,
    },
    /// A 2D heart shape (Inigo Quilez SDF).
    Heart {
        /// Center X coordinate.
        center_x: f32,
        /// Center Y coordinate.
        center_y: f32,
        /// Scale factor (half-size of the bounding square).
        scale: f32,
        /// Fill color.
        color: Color,
    },
}

/// A GPU-ready quad instance, laid out for instanced rendering (144 bytes).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct QuadInstance {
    /// Position and size: [x, y, width, height].
    pub rect: [f32; 4],
    /// UV coordinates: [u0, v0, u1, v1].
    pub uv: [f32; 4],
    /// RGBA color (linear, premultiplied).
    pub color: [f32; 4],
    /// Border radii: [top_left, top_right, bottom_left, bottom_right].
    pub border_radius: [f32; 4],
    /// Shape-specific SDF parameters (radius, thickness, etc.).
    pub sdf_params: [f32; 4],
    /// Flags: [shape_type, stroke_width, opacity, bit_flags].
    pub flags: [f32; 4],
    /// Clip rectangle: [x, y, width, height]. All zeros = no clip.
    pub clip_rect: [f32; 4],
    /// Secondary color: gradient end, border/stroke color, or shadow color.
    /// All zeros = disabled.
    pub color2: [f32; 4],
    /// Extra parameters: [gradient_type, gradient_param, shadow_blur, shadow_offset_packed].
    /// All zeros = disabled.
    ///
    /// **Dual-use warning:** For [`ShapeType::Bezier`] and [`ShapeType::ArbitraryTriangle`],
    /// `extra[0..1]` stores shape geometry (endpoint / third vertex coordinates) instead of
    /// gradient data. Gradients and shadows must not be applied to these shape types.
    pub extra: [f32; 4],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_instance_is_144_bytes() {
        assert_eq!(size_of::<QuadInstance>(), 144);
    }

    #[test]
    fn shape_type_f32_roundtrip() {
        let variants = [
            ShapeType::Rect,
            ShapeType::Circle,
            ShapeType::Ellipse,
            ShapeType::Ring,
            ShapeType::Line,
            ShapeType::Arc,
            ShapeType::Triangle,
            ShapeType::Textured,
            ShapeType::Shader,
            ShapeType::Sphere3D,
            ShapeType::Torus3D,
            ShapeType::RoundedBox3D,
            ShapeType::Polygon,
            ShapeType::Star,
            ShapeType::Sector,
            ShapeType::Capsule,
            ShapeType::Cross,
            ShapeType::Bezier,
            ShapeType::ArbitraryTriangle,
            ShapeType::Trapezoid,
            ShapeType::SlicedTorus3D,
            ShapeType::Morph3D,
            ShapeType::UnderlineCurly,
            ShapeType::UnderlineDotted,
            ShapeType::UnderlineDashed,
            ShapeType::Heart,
        ];
        for variant in variants {
            let f = variant.to_f32();
            let back = ShapeType::from_f32(f).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn shape_type_from_f32_invalid() {
        assert!(ShapeType::from_f32(99.0).is_none());
        assert!(ShapeType::from_f32(-1.0).is_none());
    }

    #[test]
    fn quad_instance_size_is_16_byte_aligned() {
        // Instance stride must be a multiple of 16 for GPU vertex buffers.
        assert_eq!(size_of::<QuadInstance>() % 16, 0);
    }

    #[test]
    fn blend_mode_round_trip() {
        for mode in [
            BlendMode::Normal,
            BlendMode::Additive,
            BlendMode::Screen,
            BlendMode::Multiply,
        ] {
            let bits = mode.to_bits();
            let back = BlendMode::from_bits(bits);
            assert_eq!(mode, back);
        }
    }

    #[test]
    fn blend_mode_default_is_normal() {
        assert_eq!(BlendMode::default(), BlendMode::Normal);
    }

    #[test]
    fn blend_mode_sdf_pipeline_ids() {
        assert_eq!(BlendMode::Normal.sdf_pipeline_id(), PIPELINE_SDF_2D.0);
        assert_eq!(
            BlendMode::Additive.sdf_pipeline_id(),
            PIPELINE_SDF_2D_ADDITIVE.0
        );
        assert_eq!(
            BlendMode::Screen.sdf_pipeline_id(),
            PIPELINE_SDF_2D_SCREEN.0
        );
        assert_eq!(
            BlendMode::Multiply.sdf_pipeline_id(),
            PIPELINE_SDF_2D_MULTIPLY.0
        );
    }
}
