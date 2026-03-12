//! `esox_gfx` — Foundation graphics library for the Esocidae terminal emulator.
//!
//! Provides GPU-accelerated rendering primitives: a scene graph, atlas allocation,
//! pipeline management, damage tracking, and frame submission.

pub mod atlas;
pub mod bloom;
pub mod color;
pub mod damage;
pub mod error;
pub mod frame;
pub mod offscreen;
pub mod pipeline;
pub mod primitive;
pub mod scene;
pub mod shape;

// Re-exports for convenience.
pub use atlas::{
    AllocationId, AtlasAllocator, AtlasId, AtlasManager, AtlasRegion, AtlasTexture, ShelfAllocator,
    SlabAllocator,
};
pub use bloom::{BloomPass, PIPELINE_BLOOM_DOWNSAMPLE, PIPELINE_BLOOM_UPSAMPLE};
pub use color::{Color, srgb_to_linear};
pub use damage::{DamageRect, DamageTracker};
pub use error::Error;
pub use frame::{
    ClipKey, DrawBatch, Frame, FrameEncoder, FrameUniforms, PhaseRange, PostProcessPass,
    RenderPhase,
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
pub use shape::{ShapeBuilder, primitive_to_instance};
