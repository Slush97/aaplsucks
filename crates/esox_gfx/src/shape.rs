//! Ergonomic shape builder and primitive-to-instance conversion.

use crate::color::Color;
use crate::primitive::{
    BlendMode, BorderRadius, Primitive, QuadInstance, Rect, ShaderId, ShaderParams, ShapeType,
    UvRect,
};

/// Ergonomic builder for constructing [`QuadInstance`] values.
pub struct ShapeBuilder {
    rect: [f32; 4],
    uv: [f32; 4],
    color: [f32; 4],
    border_radius: [f32; 4],
    sdf_params: [f32; 4],
    shape_type: ShapeType,
    stroke_width: f32,
    opacity: f32,
    bit_flags: f32,
    clip_rect: [f32; 4],
    color2: [f32; 4],
    extra: [f32; 4],
    /// When `true`, `extra[0..1]` stores shape geometry (Bezier endpoints,
    /// ArbitraryTriangle third vertex) and must not be overwritten by
    /// gradient or shadow encoding.
    uses_extra_for_geometry: bool,
    /// User shader pipeline ID. When `Some`, `flags[3]` is set to this ID
    /// instead of `bit_flags`.
    shader_id: Option<ShaderId>,
    /// Blend mode for SDF shapes. Ignored for `ShapeType::Shader`.
    blend_mode: BlendMode,
}

impl ShapeBuilder {
    /// Start building a rounded rectangle.
    pub fn rect(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self {
            rect: [x, y, w, h],
            uv: [0.0; 4],
            color: [1.0, 1.0, 1.0, 1.0],
            border_radius: [0.0; 4],
            sdf_params: [0.0; 4],
            shape_type: ShapeType::Rect,
            stroke_width: 0.0,
            opacity: 1.0,
            bit_flags: 0.0,
            clip_rect: [0.0; 4],
            color2: [0.0; 4],
            extra: [0.0; 4],
            uses_extra_for_geometry: false,
            shader_id: None,
            blend_mode: BlendMode::Normal,
        }
    }

    /// Start building a custom shader quad.
    ///
    /// The quad will be routed through the pipeline registered with `shader_id`.
    /// Use [`shader_params`](Self::shader_params) to pass up to 16 floats.
    pub fn shader(x: f32, y: f32, w: f32, h: f32, shader_id: ShaderId) -> Self {
        Self {
            rect: [x, y, w, h],
            shape_type: ShapeType::Shader,
            shader_id: Some(shader_id),
            uses_extra_for_geometry: true,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Pack 16 user floats into the instance fields read by a custom shader.
    ///
    /// Layout matches [`Primitive::ShaderRect`] packing:
    /// - `sdf_params` ← `params.values[0..4]`
    /// - `extra` ← `params.values[4..8]`
    /// - `border_radius` ← `params.values[8..12]`
    /// - `color2` ← `params.values[12..16]`
    pub fn shader_params(mut self, params: ShaderParams) -> Self {
        self.sdf_params = [
            params.values[0],
            params.values[1],
            params.values[2],
            params.values[3],
        ];
        self.extra = [
            params.values[4],
            params.values[5],
            params.values[6],
            params.values[7],
        ];
        self.border_radius = [
            params.values[8],
            params.values[9],
            params.values[10],
            params.values[11],
        ];
        self.color2 = [
            params.values[12],
            params.values[13],
            params.values[14],
            params.values[15],
        ];
        self
    }

    /// Start building a rounded rectangle with uniform corner radius.
    pub fn rounded_rect(x: f32, y: f32, w: f32, h: f32, radius: f32) -> Self {
        Self {
            border_radius: [radius, radius, radius, radius],
            ..Self::rect(x, y, w, h)
        }
    }

    /// Start building a circle.
    pub fn circle(cx: f32, cy: f32, radius: f32) -> Self {
        let diameter = radius * 2.0;
        Self {
            rect: [cx - radius, cy - radius, diameter, diameter],
            sdf_params: [radius, 0.0, 0.0, 0.0],
            shape_type: ShapeType::Circle,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building an ellipse.
    pub fn ellipse(cx: f32, cy: f32, rx: f32, ry: f32) -> Self {
        Self {
            rect: [cx - rx, cy - ry, rx * 2.0, ry * 2.0],
            sdf_params: [rx, ry, 0.0, 0.0],
            shape_type: ShapeType::Ellipse,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building a ring (annulus).
    pub fn ring(cx: f32, cy: f32, outer_r: f32, inner_r: f32) -> Self {
        let diameter = outer_r * 2.0;
        Self {
            rect: [cx - outer_r, cy - outer_r, diameter, diameter],
            sdf_params: [outer_r, inner_r, 0.0, 0.0],
            shape_type: ShapeType::Ring,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building a line (capsule).
    pub fn line(x1: f32, y1: f32, x2: f32, y2: f32, thickness: f32) -> Self {
        let half = thickness / 2.0;
        let min_x = x1.min(x2) - half;
        let min_y = y1.min(y2) - half;
        let max_x = x1.max(x2) + half;
        let max_y = y1.max(y2) + half;
        Self {
            rect: [min_x, min_y, max_x - min_x, max_y - min_y],
            sdf_params: [x1, y1, x2, y2],
            shape_type: ShapeType::Line,
            stroke_width: thickness,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building a circular arc.
    pub fn arc(
        cx: f32,
        cy: f32,
        radius: f32,
        thickness: f32,
        angle_start: f32,
        angle_sweep: f32,
    ) -> Self {
        let extent = radius + thickness / 2.0;
        let size = extent * 2.0;
        Self {
            rect: [cx - extent, cy - extent, size, size],
            sdf_params: [radius, thickness, angle_start, angle_sweep],
            shape_type: ShapeType::Arc,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building a triangle inscribed in the given rect.
    pub fn triangle(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self {
            rect: [x, y, w, h],
            shape_type: ShapeType::Triangle,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building a regular N-sided polygon.
    pub fn polygon(cx: f32, cy: f32, radius: f32, sides: u32) -> Self {
        let diameter = radius * 2.0;
        Self {
            rect: [cx - radius, cy - radius, diameter, diameter],
            sdf_params: [sides as f32, 0.0, 0.0, 0.0],
            shape_type: ShapeType::Polygon,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building an N-pointed star.
    pub fn star(cx: f32, cy: f32, points: u32, inner_r: f32, outer_r: f32) -> Self {
        let diameter = outer_r * 2.0;
        Self {
            rect: [cx - outer_r, cy - outer_r, diameter, diameter],
            sdf_params: [points as f32, inner_r, outer_r, 0.0],
            shape_type: ShapeType::Star,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building a pie/sector slice.
    pub fn sector(cx: f32, cy: f32, radius: f32, angle_start: f32, angle_sweep: f32) -> Self {
        let diameter = radius * 2.0;
        Self {
            rect: [cx - radius, cy - radius, diameter, diameter],
            sdf_params: [radius, angle_start, angle_sweep, 0.0],
            shape_type: ShapeType::Sector,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building a capsule (stadium) shape.
    pub fn capsule(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self {
            rect: [x, y, w, h],
            shape_type: ShapeType::Capsule,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building an axis-aligned cross/plus sign.
    pub fn cross(cx: f32, cy: f32, arm_width: f32, arm_length: f32) -> Self {
        let size = arm_length * 2.0;
        Self {
            rect: [cx - arm_length, cy - arm_length, size, size],
            sdf_params: [arm_width, arm_length, 0.0, 0.0],
            shape_type: ShapeType::Cross,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building a quadratic bezier curve stroke.
    ///
    /// Gradients and shadows are not supported on this shape type because
    /// `extra` is used for geometry data (endpoint coordinates).
    pub fn bezier(x0: f32, y0: f32, cx: f32, cy: f32, x1: f32, y1: f32, thickness: f32) -> Self {
        let half = thickness / 2.0;
        let min_x = x0.min(cx).min(x1) - half;
        let min_y = y0.min(cy).min(y1) - half;
        let max_x = x0.max(cx).max(x1) + half;
        let max_y = y0.max(cy).max(y1) + half;
        Self {
            rect: [min_x, min_y, max_x - min_x, max_y - min_y],
            sdf_params: [cx, cy, 0.0, 0.0],
            extra: [x1, y1, 0.0, 0.0],
            shape_type: ShapeType::Bezier,
            stroke_width: thickness,
            // Store start point in uv (repurposed for non-textured shapes).
            uv: [x0, y0, 0.0, 0.0],
            uses_extra_for_geometry: true,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building a triangle with 3 arbitrary vertices.
    ///
    /// Gradients and shadows are not supported on this shape type because
    /// `extra` is used for geometry data (third vertex coordinates).
    pub fn arbitrary_triangle(x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) -> Self {
        let min_x = x1.min(x2).min(x3);
        let min_y = y1.min(y2).min(y3);
        let max_x = x1.max(x2).max(x3);
        let max_y = y1.max(y2).max(y3);
        Self {
            rect: [min_x, min_y, max_x - min_x, max_y - min_y],
            sdf_params: [x1, y1, x2, y2],
            extra: [x3, y3, 0.0, 0.0],
            shape_type: ShapeType::ArbitraryTriangle,
            uses_extra_for_geometry: true,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building a symmetric trapezoid.
    pub fn trapezoid(cx: f32, cy: f32, top_w: f32, bottom_w: f32, h: f32) -> Self {
        let half_w = top_w.max(bottom_w);
        let half_h = h / 2.0;
        Self {
            rect: [cx - half_w, cy - half_h, half_w * 2.0, h],
            sdf_params: [top_w / 2.0, bottom_w / 2.0, half_h, 0.0],
            shape_type: ShapeType::Trapezoid,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building a 2D heart shape (Inigo Quilez SDF).
    pub fn heart(cx: f32, cy: f32, scale: f32) -> Self {
        let size = scale * 2.0;
        Self {
            rect: [cx - scale, cy - scale, size, size],
            sdf_params: [scale, 0.0, 0.0, 0.0],
            shape_type: ShapeType::Heart,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building a raymarched 3D sphere.
    pub fn sphere_3d(cx: f32, cy: f32, size: f32, radius: f32) -> Self {
        let half = size / 2.0;
        Self {
            rect: [cx - half, cy - half, size, size],
            sdf_params: [radius, 0.0, 0.0, 0.0],
            shape_type: ShapeType::Sphere3D,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building a raymarched 3D torus.
    pub fn torus_3d(cx: f32, cy: f32, size: f32, major_r: f32, minor_r: f32) -> Self {
        let half = size / 2.0;
        Self {
            rect: [cx - half, cy - half, size, size],
            sdf_params: [major_r, minor_r, 0.0, 0.0],
            shape_type: ShapeType::Torus3D,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building a raymarched 3D sliced torus piece.
    ///
    /// Each piece is a torus fragment clipped by two cut planes. The clip bounds
    /// are packed into `border_radius` as `[a_min, a_max, b_min, b_max]` and the
    /// piece translation into `extra[0..2]`.
    ///
    /// Gradients and shadows are not supported (extra field is used for geometry).
    pub fn sliced_torus_3d(
        cx: f32,
        cy: f32,
        size: f32,
        major_r: f32,
        minor_r: f32,
        clip_bounds: [f32; 4],
        translation: [f32; 3],
    ) -> Self {
        let half = size / 2.0;
        Self {
            rect: [cx - half, cy - half, size, size],
            sdf_params: [major_r, minor_r, 0.0, 0.0],
            border_radius: clip_bounds,
            extra: [translation[0], translation[1], translation[2], 0.0],
            shape_type: ShapeType::SlicedTorus3D,
            uses_extra_for_geometry: true,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Override the time used for rotation and rainbow hue in the shader.
    ///
    /// When non-zero, the shader uses this value instead of the global
    /// `uniforms.time.x`, allowing the animation to freeze at a specific
    /// time. Stored in `extra[3]`.
    pub fn time_override(mut self, t: f32) -> Self {
        self.extra[3] = t;
        self
    }

    /// Start building a raymarched 3D morphing shape that blends between two SDF primitives.
    ///
    /// Shape type IDs: 0 = sphere, 1 = torus, 2 = rounded box.
    ///
    /// Gradients and shadows are not supported (extra field is used for geometry).
    #[allow(clippy::too_many_arguments)]
    pub fn morph_3d(
        cx: f32,
        cy: f32,
        size: f32,
        shape_a: u32,
        shape_b: u32,
        morph_factor: f32,
        params_a: [f32; 2],
        params_b: [f32; 2],
        extra_b: [f32; 3],
    ) -> Self {
        let half = size / 2.0;
        Self {
            rect: [cx - half, cy - half, size, size],
            sdf_params: [params_a[0], params_a[1], params_b[0], params_b[1]],
            border_radius: [morph_factor, shape_a as f32, shape_b as f32, 0.0],
            extra: [extra_b[0], extra_b[1], extra_b[2], 0.0],
            shape_type: ShapeType::Morph3D,
            uses_extra_for_geometry: true,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building a raymarched 3D rounded box.
    pub fn rounded_box_3d(
        cx: f32,
        cy: f32,
        size: f32,
        hx: f32,
        hy: f32,
        hz: f32,
        corner_r: f32,
    ) -> Self {
        let half = size / 2.0;
        Self {
            rect: [cx - half, cy - half, size, size],
            sdf_params: [hx, hy, hz, corner_r],
            shape_type: ShapeType::RoundedBox3D,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building a textured quad (layer 0).
    pub fn textured(x: f32, y: f32, w: f32, h: f32, uv: UvRect) -> Self {
        Self {
            rect: [x, y, w, h],
            uv: [uv.u0, uv.v0, uv.u1, uv.v1],
            shape_type: ShapeType::Textured,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Start building a textured quad on a specific atlas layer.
    pub fn textured_layered(x: f32, y: f32, w: f32, h: f32, uv: UvRect, layer: u32) -> Self {
        Self {
            rect: [x, y, w, h],
            uv: [uv.u0, uv.v0, uv.u1, uv.v1],
            shape_type: ShapeType::Textured,
            bit_flags: layer as f32,
            ..Self::rect(0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Set the fill color.
    pub fn color(mut self, c: Color) -> Self {
        self.color = [c.r, c.g, c.b, c.a];
        self
    }

    /// Set per-corner border radii.
    pub fn border_radius(mut self, br: BorderRadius) -> Self {
        self.border_radius = [br.top_left, br.top_right, br.bottom_left, br.bottom_right];
        self
    }

    /// Set stroke width (0 = filled).
    pub fn stroke(mut self, width: f32) -> Self {
        self.stroke_width = width;
        self
    }

    /// Set opacity (0.0–1.0).
    pub fn opacity(mut self, o: f32) -> Self {
        self.opacity = o;
        self
    }

    /// Enable rainbow normal-based coloring for 3D shapes.
    ///
    /// When set, the shader computes base color from surface normals using HSV
    /// instead of using the instance color.
    ///
    /// No-op on `ShapeType::Shader` (flags[3] carries the pipeline ID).
    pub fn rainbow(mut self) -> Self {
        if self.shader_id.is_some() {
            tracing::warn!("rainbow() ignored on ShapeType::Shader (flags[3] is pipeline ID)");
            return self;
        }
        self.bit_flags = 1.0;
        self
    }

    /// Set the blend mode for this shape.
    ///
    /// Controls how the shape's color is composited with the framebuffer.
    /// No-op on `ShapeType::Shader` (they define their own pipeline).
    pub fn blend(mut self, mode: BlendMode) -> Self {
        if self.shader_id.is_some() {
            tracing::warn!("blend() ignored on ShapeType::Shader (flags[3] is pipeline ID)");
            return self;
        }
        self.blend_mode = mode;
        self
    }

    /// Set the secondary color (gradient end, border color, or shadow color).
    pub fn color2(mut self, c: Color) -> Self {
        self.color2 = [c.r, c.g, c.b, c.a];
        self
    }

    /// Apply a linear gradient from the primary color to `end_color`.
    ///
    /// `angle` is in radians (0 = left-to-right, π/2 = top-to-bottom).
    ///
    /// No-op on Bezier and ArbitraryTriangle shapes (extra field conflict).
    pub fn linear_gradient(mut self, end_color: Color, angle: f32) -> Self {
        if self.uses_extra_for_geometry {
            tracing::warn!("linear_gradient() ignored: extra field is used for shape geometry");
            return self;
        }
        self.color2 = [end_color.r, end_color.g, end_color.b, end_color.a];
        self.extra[0] = 1.0;
        self.extra[1] = angle;
        self
    }

    /// Apply a radial gradient from the primary color (center) to `end_color` (edge).
    ///
    /// No-op on Bezier and ArbitraryTriangle shapes (extra field conflict).
    pub fn radial_gradient(mut self, end_color: Color) -> Self {
        if self.uses_extra_for_geometry {
            tracing::warn!("radial_gradient() ignored: extra field is used for shape geometry");
            return self;
        }
        self.color2 = [end_color.r, end_color.g, end_color.b, end_color.a];
        self.extra[0] = 2.0;
        self
    }

    /// Apply a conic (angular) gradient from the primary color to `end_color`.
    ///
    /// `start_angle` is in radians.
    ///
    /// No-op on Bezier and ArbitraryTriangle shapes (extra field conflict).
    pub fn conic_gradient(mut self, end_color: Color, start_angle: f32) -> Self {
        if self.uses_extra_for_geometry {
            tracing::warn!("conic_gradient() ignored: extra field is used for shape geometry");
            return self;
        }
        self.color2 = [end_color.r, end_color.g, end_color.b, end_color.a];
        self.extra[0] = 3.0;
        self.extra[1] = start_angle;
        self
    }

    /// Add a drop shadow behind the shape.
    ///
    /// Inflates the bounding rect so shadow fragments are not clipped.
    /// Shadow color defaults to `color2` if set, otherwise semi-transparent black.
    ///
    /// No-op on Bezier and ArbitraryTriangle shapes (extra field conflict).
    pub fn shadow(mut self, blur: f32, dx: f32, dy: f32) -> Self {
        if self.uses_extra_for_geometry {
            tracing::warn!("shadow() ignored: extra field is used for shape geometry");
            return self;
        }
        self.extra[2] = blur;
        self.extra[3] = (dx + 128.0) * 256.0 + (dy + 128.0);
        // Inflate bounding rect to accommodate shadow extent.
        let inflate = blur + dx.abs().max(dy.abs());
        self.rect[0] -= inflate;
        self.rect[1] -= inflate;
        self.rect[2] += inflate * 2.0;
        self.rect[3] += inflate * 2.0;
        self
    }

    /// Add a glow effect (shadow with zero offset).
    ///
    /// No-op on Bezier and ArbitraryTriangle shapes (extra field conflict).
    pub fn glow(self, blur: f32) -> Self {
        self.shadow(blur, 0.0, 0.0)
    }

    /// Set the clip rectangle. `None` means no clipping (sentinel `[0,0,0,0]`).
    pub fn clip(mut self, rect: Option<Rect>) -> Self {
        self.clip_rect = match rect {
            Some(r) => [r.x, r.y, r.width, r.height],
            None => [0.0; 4],
        };
        self
    }

    /// Build the final [`QuadInstance`].
    pub fn build(self) -> QuadInstance {
        let flags_3 = match self.shader_id {
            Some(id) => id.0 as f32,
            None if self.shape_type.pipeline_id() == crate::primitive::PIPELINE_SDF_2D.0 => {
                // SDF 2D shapes: encode blend mode in bits 1–2, rainbow flag in bit 0.
                let rainbow_bit = self.bit_flags as u32 & 1;
                let blend_bits = (self.blend_mode.to_bits() as u32) << 1;
                (blend_bits | rainbow_bit) as f32
            }
            None => self.bit_flags,
        };
        QuadInstance {
            rect: self.rect,
            uv: self.uv,
            color: self.color,
            border_radius: self.border_radius,
            sdf_params: self.sdf_params,
            flags: [
                self.shape_type.to_f32(),
                self.stroke_width,
                self.opacity,
                flags_3,
            ],
            clip_rect: self.clip_rect,
            color2: self.color2,
            extra: self.extra,
        }
    }
}

/// Convert a resolved [`Primitive`] into a [`QuadInstance`] for GPU upload.
pub fn primitive_to_instance(p: &Primitive, opacity: f32, clip: Option<Rect>) -> QuadInstance {
    match *p {
        Primitive::SolidRect {
            rect,
            color,
            border_radius,
        } => ShapeBuilder::rect(rect.x, rect.y, rect.width, rect.height)
            .color(color)
            .border_radius(border_radius)
            .opacity(opacity)
            .clip(clip)
            .build(),

        Primitive::TexturedRect {
            rect,
            uv,
            color,
            layer,
        } => ShapeBuilder::textured_layered(rect.x, rect.y, rect.width, rect.height, uv, layer)
            .color(color)
            .opacity(opacity)
            .clip(clip)
            .build(),

        Primitive::ShaderRect {
            rect,
            shader,
            params,
        } => {
            // Pack 16 shader params into existing QuadInstance fields:
            // sdf_params[0..4] = params[0..4]
            // extra[0..4]      = params[4..8]
            // border_radius[0..4] = params[8..12]
            // color2[0..4]     = params[12..16]
            let clip_rect = match clip {
                Some(r) => [r.x, r.y, r.width, r.height],
                None => [0.0; 4],
            };
            QuadInstance {
                rect: [rect.x, rect.y, rect.width, rect.height],
                uv: [0.0; 4],
                color: [1.0, 1.0, 1.0, 1.0],
                border_radius: [
                    params.values[8],
                    params.values[9],
                    params.values[10],
                    params.values[11],
                ],
                sdf_params: [
                    params.values[0],
                    params.values[1],
                    params.values[2],
                    params.values[3],
                ],
                flags: [ShapeType::Shader.to_f32(), 0.0, opacity, shader.0 as f32],
                clip_rect,
                color2: [
                    params.values[12],
                    params.values[13],
                    params.values[14],
                    params.values[15],
                ],
                extra: [
                    params.values[4],
                    params.values[5],
                    params.values[6],
                    params.values[7],
                ],
            }
        }

        Primitive::Circle {
            center_x,
            center_y,
            radius,
            color,
        } => ShapeBuilder::circle(center_x, center_y, radius)
            .color(color)
            .opacity(opacity)
            .clip(clip)
            .build(),

        Primitive::Ellipse {
            center_x,
            center_y,
            rx,
            ry,
            color,
        } => ShapeBuilder::ellipse(center_x, center_y, rx, ry)
            .color(color)
            .opacity(opacity)
            .clip(clip)
            .build(),

        Primitive::Ring {
            center_x,
            center_y,
            outer_r,
            inner_r,
            color,
        } => ShapeBuilder::ring(center_x, center_y, outer_r, inner_r)
            .color(color)
            .opacity(opacity)
            .clip(clip)
            .build(),

        Primitive::Line {
            x1,
            y1,
            x2,
            y2,
            thickness,
            color,
        } => ShapeBuilder::line(x1, y1, x2, y2, thickness)
            .color(color)
            .opacity(opacity)
            .clip(clip)
            .build(),

        Primitive::Arc {
            center_x,
            center_y,
            radius,
            thickness,
            angle_start,
            angle_sweep,
            color,
        } => ShapeBuilder::arc(
            center_x,
            center_y,
            radius,
            thickness,
            angle_start,
            angle_sweep,
        )
        .color(color)
        .opacity(opacity)
        .clip(clip)
        .build(),

        Primitive::Triangle { rect, color } => {
            ShapeBuilder::triangle(rect.x, rect.y, rect.width, rect.height)
                .color(color)
                .opacity(opacity)
                .clip(clip)
                .build()
        }

        Primitive::Polygon {
            center_x,
            center_y,
            radius,
            sides,
            color,
        } => ShapeBuilder::polygon(center_x, center_y, radius, sides)
            .color(color)
            .opacity(opacity)
            .clip(clip)
            .build(),

        Primitive::Star {
            center_x,
            center_y,
            points,
            inner_r,
            outer_r,
            color,
        } => ShapeBuilder::star(center_x, center_y, points, inner_r, outer_r)
            .color(color)
            .opacity(opacity)
            .clip(clip)
            .build(),

        Primitive::Sector {
            center_x,
            center_y,
            radius,
            angle_start,
            angle_sweep,
            color,
        } => ShapeBuilder::sector(center_x, center_y, radius, angle_start, angle_sweep)
            .color(color)
            .opacity(opacity)
            .clip(clip)
            .build(),

        Primitive::Capsule { rect, color } => {
            ShapeBuilder::capsule(rect.x, rect.y, rect.width, rect.height)
                .color(color)
                .opacity(opacity)
                .clip(clip)
                .build()
        }

        Primitive::CrossShape {
            center_x,
            center_y,
            arm_width,
            arm_length,
            color,
        } => ShapeBuilder::cross(center_x, center_y, arm_width, arm_length)
            .color(color)
            .opacity(opacity)
            .clip(clip)
            .build(),

        Primitive::Bezier {
            x0,
            y0,
            cx,
            cy,
            x1,
            y1,
            thickness,
            color,
        } => ShapeBuilder::bezier(x0, y0, cx, cy, x1, y1, thickness)
            .color(color)
            .opacity(opacity)
            .clip(clip)
            .build(),

        Primitive::ArbitraryTriangle {
            x1,
            y1,
            x2,
            y2,
            x3,
            y3,
            color,
        } => ShapeBuilder::arbitrary_triangle(x1, y1, x2, y2, x3, y3)
            .color(color)
            .opacity(opacity)
            .clip(clip)
            .build(),

        Primitive::Trapezoid {
            center_x,
            center_y,
            top_half_w,
            bottom_half_w,
            half_h,
            color,
        } => ShapeBuilder::trapezoid(
            center_x,
            center_y,
            top_half_w * 2.0,
            bottom_half_w * 2.0,
            half_h * 2.0,
        )
        .color(color)
        .opacity(opacity)
        .clip(clip)
        .build(),

        Primitive::Heart {
            center_x,
            center_y,
            scale,
            color,
        } => ShapeBuilder::heart(center_x, center_y, scale)
            .color(color)
            .opacity(opacity)
            .clip(clip)
            .build(),
    }
}

/// Get the [`ShaderId`] from a shader primitive's instance, if applicable.
pub fn shader_id_from_instance(instance: &QuadInstance) -> Option<ShaderId> {
    let shape = ShapeType::from_f32(instance.flags[0])?;
    if shape == ShapeType::Shader {
        Some(ShaderId(instance.flags[3] as u32))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::Rect;

    #[test]
    fn builder_circle() {
        let q = ShapeBuilder::circle(100.0, 200.0, 50.0)
            .color(Color::new(1.0, 0.0, 0.0, 1.0))
            .build();
        assert_eq!(q.rect, [50.0, 150.0, 100.0, 100.0]);
        assert_eq!(q.sdf_params[0], 50.0);
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Circle));
    }

    #[test]
    fn builder_rounded_rect() {
        let q = ShapeBuilder::rounded_rect(10.0, 20.0, 100.0, 50.0, 8.0)
            .stroke(2.0)
            .color(Color::WHITE)
            .build();
        assert_eq!(q.rect, [10.0, 20.0, 100.0, 50.0]);
        assert_eq!(q.border_radius, [8.0, 8.0, 8.0, 8.0]);
        assert_eq!(q.flags[1], 2.0); // stroke_width
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Rect));
    }

    #[test]
    fn builder_ellipse() {
        let q = ShapeBuilder::ellipse(50.0, 50.0, 30.0, 20.0)
            .color(Color::WHITE)
            .build();
        assert_eq!(q.rect, [20.0, 30.0, 60.0, 40.0]);
        assert_eq!(q.sdf_params[0], 30.0);
        assert_eq!(q.sdf_params[1], 20.0);
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Ellipse));
    }

    #[test]
    fn builder_ring() {
        let q = ShapeBuilder::ring(100.0, 100.0, 50.0, 40.0)
            .color(Color::WHITE)
            .build();
        assert_eq!(q.rect, [50.0, 50.0, 100.0, 100.0]);
        assert_eq!(q.sdf_params[0], 50.0);
        assert_eq!(q.sdf_params[1], 40.0);
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Ring));
    }

    #[test]
    fn builder_line() {
        let q = ShapeBuilder::line(10.0, 20.0, 100.0, 200.0, 4.0)
            .color(Color::WHITE)
            .build();
        assert_eq!(q.sdf_params, [10.0, 20.0, 100.0, 200.0]);
        assert_eq!(q.flags[1], 4.0); // stroke_width = thickness
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Line));
    }

    #[test]
    fn builder_triangle() {
        let q = ShapeBuilder::triangle(10.0, 20.0, 30.0, 40.0)
            .color(Color::WHITE)
            .build();
        assert_eq!(q.rect, [10.0, 20.0, 30.0, 40.0]);
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Triangle));
    }

    #[test]
    fn builder_textured() {
        let uv = UvRect {
            u0: 0.0,
            v0: 0.0,
            u1: 1.0,
            v1: 1.0,
        };
        let q = ShapeBuilder::textured(0.0, 0.0, 64.0, 64.0, uv)
            .color(Color::WHITE)
            .build();
        assert_eq!(q.uv, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Textured));
    }

    #[test]
    fn builder_sphere_3d() {
        let q = ShapeBuilder::sphere_3d(200.0, 200.0, 300.0, 0.8)
            .color(Color::WHITE)
            .build();
        assert_eq!(q.rect, [50.0, 50.0, 300.0, 300.0]);
        assert_eq!(q.sdf_params[0], 0.8);
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Sphere3D));
    }

    #[test]
    fn builder_torus_3d() {
        let q = ShapeBuilder::torus_3d(200.0, 200.0, 300.0, 0.6, 0.2)
            .color(Color::WHITE)
            .build();
        assert_eq!(q.rect, [50.0, 50.0, 300.0, 300.0]);
        assert_eq!(q.sdf_params[0], 0.6);
        assert_eq!(q.sdf_params[1], 0.2);
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Torus3D));
    }

    #[test]
    fn builder_rounded_box_3d() {
        let q = ShapeBuilder::rounded_box_3d(200.0, 200.0, 300.0, 0.5, 0.5, 0.5, 0.1)
            .color(Color::WHITE)
            .build();
        assert_eq!(q.rect, [50.0, 50.0, 300.0, 300.0]);
        assert_eq!(q.sdf_params, [0.5, 0.5, 0.5, 0.1]);
        assert_eq!(
            ShapeType::from_f32(q.flags[0]),
            Some(ShapeType::RoundedBox3D)
        );
    }

    #[test]
    fn primitive_to_instance_solid_rect() {
        let p = Primitive::SolidRect {
            rect: Rect {
                x: 10.0,
                y: 20.0,
                width: 100.0,
                height: 50.0,
            },
            color: Color::new(1.0, 0.0, 0.0, 1.0),
            border_radius: BorderRadius::uniform(4.0),
        };
        let q = primitive_to_instance(&p, 0.8, None);
        assert_eq!(q.rect, [10.0, 20.0, 100.0, 50.0]);
        assert_eq!(q.color[0], 1.0);
        assert_eq!(q.border_radius, [4.0, 4.0, 4.0, 4.0]);
        assert_eq!(q.flags[2], 0.8); // opacity
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Rect));
        assert_eq!(q.clip_rect, [0.0; 4]); // no clip
    }

    #[test]
    fn primitive_to_instance_circle() {
        let p = Primitive::Circle {
            center_x: 50.0,
            center_y: 50.0,
            radius: 25.0,
            color: Color::WHITE,
        };
        let q = primitive_to_instance(&p, 1.0, None);
        assert_eq!(q.rect, [25.0, 25.0, 50.0, 50.0]);
        assert_eq!(q.sdf_params[0], 25.0);
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Circle));
    }

    #[test]
    fn primitive_to_instance_line() {
        let p = Primitive::Line {
            x1: 0.0,
            y1: 0.0,
            x2: 100.0,
            y2: 0.0,
            thickness: 2.0,
            color: Color::WHITE,
        };
        let q = primitive_to_instance(&p, 1.0, None);
        assert_eq!(q.sdf_params, [0.0, 0.0, 100.0, 0.0]);
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Line));
    }

    #[test]
    fn primitive_to_instance_textured_rect() {
        let p = Primitive::TexturedRect {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                width: 64.0,
                height: 64.0,
            },
            uv: UvRect {
                u0: 0.0,
                v0: 0.0,
                u1: 0.5,
                v1: 0.5,
            },
            color: Color::WHITE,
            layer: 0,
        };
        let q = primitive_to_instance(&p, 1.0, None);
        assert_eq!(q.uv, [0.0, 0.0, 0.5, 0.5]);
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Textured));
    }

    #[test]
    fn shader_id_extraction() {
        let p = Primitive::ShaderRect {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 100.0,
            },
            shader: ShaderId(42),
            params: crate::primitive::ShaderParams { values: [0.0; 16] },
        };
        let q = primitive_to_instance(&p, 1.0, None);
        assert_eq!(shader_id_from_instance(&q), Some(ShaderId(42)));
    }

    #[test]
    fn builder_arc() {
        let q = ShapeBuilder::arc(100.0, 100.0, 40.0, 6.0, 0.0, std::f32::consts::PI)
            .color(Color::WHITE)
            .build();
        assert_eq!(q.sdf_params, [40.0, 6.0, 0.0, std::f32::consts::PI]);
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Arc));
        // Bounding quad: extent = 40 + 3 = 43, size = 86
        assert_eq!(q.rect[2], 86.0);
        assert_eq!(q.rect[3], 86.0);
    }

    #[test]
    fn builder_clip_rect() {
        let clip = Rect {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 50.0,
        };
        let q = ShapeBuilder::rect(0.0, 0.0, 200.0, 200.0)
            .clip(Some(clip))
            .build();
        assert_eq!(q.clip_rect, [10.0, 20.0, 100.0, 50.0]);
    }

    #[test]
    fn builder_rainbow_sets_bit_flags() {
        let q = ShapeBuilder::rounded_box_3d(200.0, 200.0, 300.0, 0.45, 0.45, 0.45, 0.05)
            .rainbow()
            .color(Color::WHITE)
            .build();
        assert_eq!(q.flags[3], 1.0); // bit_flags = rainbow
        assert_eq!(
            ShapeType::from_f32(q.flags[0]),
            Some(ShapeType::RoundedBox3D)
        );
    }

    #[test]
    fn builder_no_clip_is_zeroed() {
        let q = ShapeBuilder::rect(0.0, 0.0, 200.0, 200.0).build();
        assert_eq!(q.clip_rect, [0.0; 4]);
    }

    #[test]
    fn builder_polygon() {
        let q = ShapeBuilder::polygon(100.0, 100.0, 50.0, 6)
            .color(Color::WHITE)
            .build();
        assert_eq!(q.rect, [50.0, 50.0, 100.0, 100.0]);
        assert_eq!(q.sdf_params[0], 6.0);
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Polygon));
    }

    #[test]
    fn builder_star() {
        let q = ShapeBuilder::star(100.0, 100.0, 5, 20.0, 50.0)
            .color(Color::WHITE)
            .build();
        assert_eq!(q.rect, [50.0, 50.0, 100.0, 100.0]);
        assert_eq!(q.sdf_params[0], 5.0);
        assert_eq!(q.sdf_params[1], 20.0);
        assert_eq!(q.sdf_params[2], 50.0);
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Star));
    }

    #[test]
    fn builder_sector() {
        let q = ShapeBuilder::sector(100.0, 100.0, 50.0, 0.0, 1.5)
            .color(Color::WHITE)
            .build();
        assert_eq!(q.sdf_params, [50.0, 0.0, 1.5, 0.0]);
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Sector));
    }

    #[test]
    fn builder_capsule() {
        let q = ShapeBuilder::capsule(10.0, 20.0, 100.0, 30.0)
            .color(Color::WHITE)
            .build();
        assert_eq!(q.rect, [10.0, 20.0, 100.0, 30.0]);
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Capsule));
    }

    #[test]
    fn builder_cross() {
        let q = ShapeBuilder::cross(100.0, 100.0, 10.0, 40.0)
            .color(Color::WHITE)
            .build();
        assert_eq!(q.sdf_params[0], 10.0);
        assert_eq!(q.sdf_params[1], 40.0);
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Cross));
    }

    #[test]
    fn builder_bezier() {
        let q = ShapeBuilder::bezier(0.0, 0.0, 50.0, 100.0, 100.0, 0.0, 4.0)
            .color(Color::WHITE)
            .build();
        assert_eq!(q.sdf_params[0], 50.0); // cx
        assert_eq!(q.sdf_params[1], 100.0); // cy
        assert_eq!(q.extra[0], 100.0); // x1
        assert_eq!(q.extra[1], 0.0); // y1
        assert_eq!(q.flags[1], 4.0); // thickness
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Bezier));
    }

    #[test]
    fn builder_arbitrary_triangle() {
        let q = ShapeBuilder::arbitrary_triangle(0.0, 0.0, 100.0, 0.0, 50.0, 80.0)
            .color(Color::WHITE)
            .build();
        assert_eq!(q.sdf_params, [0.0, 0.0, 100.0, 0.0]);
        assert_eq!(q.extra[0], 50.0);
        assert_eq!(q.extra[1], 80.0);
        assert_eq!(
            ShapeType::from_f32(q.flags[0]),
            Some(ShapeType::ArbitraryTriangle)
        );
    }

    #[test]
    fn builder_trapezoid() {
        let q = ShapeBuilder::trapezoid(100.0, 100.0, 40.0, 80.0, 60.0)
            .color(Color::WHITE)
            .build();
        assert_eq!(q.sdf_params[0], 20.0); // top_half_w
        assert_eq!(q.sdf_params[1], 40.0); // bottom_half_w
        assert_eq!(q.sdf_params[2], 30.0); // half_h
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Trapezoid));
    }

    #[test]
    fn builder_linear_gradient() {
        let q = ShapeBuilder::rect(0.0, 0.0, 100.0, 100.0)
            .color(Color::new(1.0, 0.0, 0.0, 1.0))
            .linear_gradient(Color::new(0.0, 0.0, 1.0, 1.0), 1.57)
            .build();
        assert_eq!(q.extra[0], 1.0); // gradient_type = linear
        assert!((q.extra[1] - 1.57).abs() < 1e-5);
        assert_eq!(q.color2[2], 1.0); // blue end color
    }

    #[test]
    fn builder_radial_gradient() {
        let q = ShapeBuilder::circle(50.0, 50.0, 50.0)
            .color(Color::WHITE)
            .radial_gradient(Color::new(0.0, 0.0, 0.0, 1.0))
            .build();
        assert_eq!(q.extra[0], 2.0); // gradient_type = radial
    }

    #[test]
    fn builder_conic_gradient() {
        let q = ShapeBuilder::rect(0.0, 0.0, 100.0, 100.0)
            .color(Color::WHITE)
            .conic_gradient(Color::new(1.0, 0.0, 0.0, 1.0), 0.5)
            .build();
        assert_eq!(q.extra[0], 3.0); // gradient_type = conic
        assert_eq!(q.extra[1], 0.5); // start_angle
    }

    #[test]
    fn builder_shadow_packing() {
        let q = ShapeBuilder::rect(100.0, 100.0, 50.0, 50.0)
            .color(Color::WHITE)
            .shadow(8.0, 4.0, 4.0)
            .build();
        assert_eq!(q.extra[2], 8.0); // blur
        // Unpack offset: dx = floor(packed/256) - 128, dy = packed%256 - 128
        let packed = q.extra[3];
        let dx = (packed / 256.0).floor() - 128.0;
        let dy = packed - (packed / 256.0).floor() * 256.0 - 128.0;
        assert!((dx - 4.0).abs() < 1.0);
        assert!((dy - 4.0).abs() < 1.0);
        // Rect should be inflated by blur + max(|dx|, |dy|) = 8 + 4 = 12
        assert!(q.rect[0] < 100.0);
        assert!(q.rect[2] > 50.0);
    }

    #[test]
    fn primitive_to_instance_arc() {
        let p = Primitive::Arc {
            center_x: 50.0,
            center_y: 50.0,
            radius: 30.0,
            thickness: 4.0,
            angle_start: 0.0,
            angle_sweep: 1.5,
            color: Color::WHITE,
        };
        let q = primitive_to_instance(&p, 0.9, None);
        assert_eq!(q.sdf_params, [30.0, 4.0, 0.0, 1.5]);
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Arc));
        assert_eq!(q.flags[2], 0.9); // opacity
    }

    #[test]
    fn bezier_gradient_is_noop() {
        let q = ShapeBuilder::bezier(0.0, 0.0, 50.0, 100.0, 100.0, 0.0, 4.0)
            .linear_gradient(Color::new(0.0, 0.0, 1.0, 1.0), 1.0)
            .build();
        // extra[0..1] must still hold geometry, not gradient encoding.
        assert_eq!(q.extra[0], 100.0); // x1 endpoint
        assert_eq!(q.extra[1], 0.0); // y1 endpoint
    }

    #[test]
    fn bezier_shadow_is_noop() {
        let q = ShapeBuilder::bezier(0.0, 0.0, 50.0, 100.0, 100.0, 0.0, 4.0)
            .shadow(8.0, 4.0, 4.0)
            .build();
        // extra[2..3] must remain zero (shadow not applied).
        assert_eq!(q.extra[2], 0.0);
        assert_eq!(q.extra[3], 0.0);
    }

    #[test]
    fn shader_params_packing_roundtrip() {
        use crate::primitive::ShaderParams;
        let mut values = [0.0f32; 16];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i + 1) as f32 * 0.1;
        }
        let p = Primitive::ShaderRect {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 100.0,
            },
            shader: crate::primitive::ShaderId(7),
            params: ShaderParams { values },
        };
        let q = primitive_to_instance(&p, 1.0, None);
        // sdf_params = params[0..4]
        assert!((q.sdf_params[0] - 0.1).abs() < 1e-5);
        assert!((q.sdf_params[1] - 0.2).abs() < 1e-5);
        assert!((q.sdf_params[2] - 0.3).abs() < 1e-5);
        assert!((q.sdf_params[3] - 0.4).abs() < 1e-5);
        // extra = params[4..8]
        assert!((q.extra[0] - 0.5).abs() < 1e-5);
        assert!((q.extra[1] - 0.6).abs() < 1e-5);
        assert!((q.extra[2] - 0.7).abs() < 1e-5);
        assert!((q.extra[3] - 0.8).abs() < 1e-5);
        // border_radius = params[8..12]
        assert!((q.border_radius[0] - 0.9).abs() < 1e-5);
        assert!((q.border_radius[1] - 1.0).abs() < 1e-5);
        assert!((q.border_radius[2] - 1.1).abs() < 1e-5);
        assert!((q.border_radius[3] - 1.2).abs() < 1e-5);
        // color2 = params[12..16]
        assert!((q.color2[0] - 1.3).abs() < 1e-5);
        assert!((q.color2[1] - 1.4).abs() < 1e-5);
        assert!((q.color2[2] - 1.5).abs() < 1e-5);
        assert!((q.color2[3] - 1.6).abs() < 1e-5);
        // Shader ID in flags[3]
        assert_eq!(q.flags[3], 7.0);
    }

    #[test]
    fn builder_morph_3d() {
        let q = ShapeBuilder::morph_3d(
            200.0,
            200.0,
            300.0,
            0, // sphere
            1, // torus
            0.5,
            [0.65, 0.0],
            [0.55, 0.20],
            [0.0, 0.0, 0.0],
        )
        .rainbow()
        .color(Color::WHITE)
        .build();
        assert_eq!(q.rect, [50.0, 50.0, 300.0, 300.0]);
        assert_eq!(q.sdf_params, [0.65, 0.0, 0.55, 0.20]);
        assert_eq!(q.border_radius[0], 0.5); // morph factor
        assert_eq!(q.border_radius[1], 0.0); // shape A = sphere
        assert_eq!(q.border_radius[2], 1.0); // shape B = torus
        assert_eq!(ShapeType::from_f32(q.flags[0]), Some(ShapeType::Morph3D));
        assert_eq!(q.flags[3], 1.0); // rainbow flag
    }

    #[test]
    fn arbitrary_triangle_gradient_is_noop() {
        let q = ShapeBuilder::arbitrary_triangle(0.0, 0.0, 100.0, 0.0, 50.0, 80.0)
            .radial_gradient(Color::new(0.0, 0.0, 0.0, 1.0))
            .build();
        // extra[0..1] must still hold third vertex.
        assert_eq!(q.extra[0], 50.0);
        assert_eq!(q.extra[1], 80.0);
    }

    #[test]
    fn builder_shader_shape_type() {
        let q = ShapeBuilder::shader(10.0, 20.0, 100.0, 50.0, ShaderId(42)).build();
        assert_eq!(q.flags[0], ShapeType::Shader.to_f32()); // 8.0
    }

    #[test]
    fn builder_shader_pipeline_id() {
        let q = ShapeBuilder::shader(10.0, 20.0, 100.0, 50.0, ShaderId(42)).build();
        assert_eq!(q.flags[3], 42.0);
    }

    #[test]
    fn builder_shader_params_packing() {
        let mut values = [0.0f32; 16];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i + 1) as f32;
        }
        let q = ShapeBuilder::shader(0.0, 0.0, 100.0, 100.0, ShaderId(10))
            .shader_params(ShaderParams { values })
            .build();
        // sdf_params = values[0..4]
        assert_eq!(q.sdf_params, [1.0, 2.0, 3.0, 4.0]);
        // extra = values[4..8]
        assert_eq!(q.extra, [5.0, 6.0, 7.0, 8.0]);
        // border_radius = values[8..12]
        assert_eq!(q.border_radius, [9.0, 10.0, 11.0, 12.0]);
        // color2 = values[12..16]
        assert_eq!(q.color2, [13.0, 14.0, 15.0, 16.0]);
    }

    #[test]
    fn builder_shader_rainbow_noop() {
        let q = ShapeBuilder::shader(0.0, 0.0, 100.0, 100.0, ShaderId(42))
            .rainbow()
            .build();
        // rainbow() should be a no-op; flags[3] must still be the shader ID.
        assert_eq!(q.flags[3], 42.0);
    }

    #[test]
    fn builder_shader_opacity() {
        let q = ShapeBuilder::shader(0.0, 0.0, 100.0, 100.0, ShaderId(10))
            .opacity(0.5)
            .build();
        assert_eq!(q.flags[2], 0.5);
    }

    #[test]
    fn blend_mode_sets_flags() {
        let q = ShapeBuilder::circle(50.0, 50.0, 10.0)
            .blend(BlendMode::Additive)
            .build();
        // Additive = 1, shifted left by 1 bit → 0b10 = 2.
        assert_eq!(q.flags[3], 2.0);
    }

    #[test]
    fn blend_mode_preserves_rainbow() {
        // Use a 2D SDF shape so blend encoding is active (3D shapes skip it).
        let q = ShapeBuilder::circle(50.0, 50.0, 10.0)
            .rainbow()
            .blend(BlendMode::Screen)
            .build();
        // Screen = 2, shifted left → 0b100, plus rainbow bit 0 → 0b101 = 5.
        assert_eq!(q.flags[3], 5.0);
    }

    #[test]
    fn blend_mode_ignored_for_shader() {
        let q = ShapeBuilder::shader(0.0, 0.0, 100.0, 100.0, ShaderId(42))
            .blend(BlendMode::Additive)
            .build();
        // blend() should be a no-op; flags[3] must still be the shader ID.
        assert_eq!(q.flags[3], 42.0);
    }
}
