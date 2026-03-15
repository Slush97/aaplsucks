//! `esox_gfx` — GPU rendering engine for esox applications.
//!
//! Provides GPU-accelerated rendering primitives: atlas allocation, pipeline
//! management, damage tracking, and frame submission.
//!
//! ## Architecture
//!
//! `esox_gfx` uses a **frame-based** architecture rather than a retained scene graph.
//! Each frame the application pushes [`Primitive`]s into a [`Frame`], which batches
//! them by pipeline/clip key and submits a single draw call per batch via
//! [`FrameEncoder`]. A [`Scene`] (arena-based node graph) exists for optional
//! retained-mode use, but the immediate-mode [`Frame`] path is the primary API
//! used by `esox_ui`.
//!
//! ## Key types
//!
//! - [`GpuContext`] — wgpu device/queue/surface lifecycle
//! - [`Frame`] / [`FrameEncoder`] — per-frame draw command buffer
//! - [`ShapeBuilder`] — ergonomic primitive construction
//! - [`DamageTracker`] — tracks dirty regions for frame-skip optimization
//! - [`ShelfAllocator`] / [`SlabAllocator`] — glyph/image atlas packing
//! - [`BloomPass`] / [`OffscreenTarget`] — post-processing pipeline

pub mod atlas;
pub mod bloom;
pub mod color;
pub mod damage;
pub mod error;
pub mod frame;
#[cfg(feature = "mesh3d")]
pub mod mesh3d;
pub mod offscreen;
pub mod pipeline;
pub mod primitive;
pub mod scene;
pub mod screenshot;
pub mod shape;

// Re-exports for convenience.
pub use atlas::{
    AllocationId, AtlasAllocator, AtlasId, AtlasManager, AtlasRegion, AtlasTexture, ShelfAllocator,
    SlabAllocator,
};
pub use bloom::{BloomPass, PIPELINE_BLOOM_DOWNSAMPLE, PIPELINE_BLOOM_UPSAMPLE};
pub use color::{Color, srgb_to_linear};
pub use damage::{DamageRect, DamageTracker, TileGrid, TileIndex, TILE_SIZE};
pub use error::Error;
pub use frame::{
    ClipKey, ColorLoadOp, DrawBatch, Frame, FrameEncoder, FrameUniforms, PhaseRange,
    PostProcessPass, RenderPhase, SurfaceFrame,
};
pub use offscreen::{
    OffscreenTarget, PIPELINE_POST_PROCESS, POST_PROCESS_FRAGMENT, POST_PROCESS_IDENTITY_FRAGMENT,
    POST_PROCESS_PREAMBLE, POST_PROCESS_VERTEX, POST_PROCESS_VERTEX_SOURCE, PostProcessParams,
    compose_user_shader, post_process_bind_group_layout, validate_user_shader,
};
pub use pipeline::{
    GpuContext, PipelineCompileConfig, PipelineHandle, PipelineReceiver, PipelineRegistry,
    ReadyPipeline, RenderResources, SHADER_PREAMBLE, spawn_pipeline_compilation,
    validate_scene_shader,
};
pub use primitive::{
    BlendMode, BorderRadius, PIPELINE_3D, PIPELINE_SDF_2D, PIPELINE_SDF_2D_ADDITIVE,
    PIPELINE_SDF_2D_MULTIPLY, PIPELINE_SDF_2D_OPAQUE, PIPELINE_SDF_2D_SCREEN, PIPELINE_TEXT,
    Primitive, QuadInstance, Rect, ShaderId, ShaderParams, ShapeType, USER_SHADER_ID_MIN, UvRect,
};
pub use scene::{
    MAX_BATCH_PRIMITIVES, MAX_NODES, Node, NodeContent, NodeId, ResolvedPrimitive, Scene,
};
pub use screenshot::ScreenshotCapture;
pub use shape::{ShapeBuilder, primitive_to_instance};
