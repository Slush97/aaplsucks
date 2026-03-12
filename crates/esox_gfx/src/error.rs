/// Errors produced by the graphics subsystem.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// No suitable GPU adapter found.
    #[error("no suitable GPU adapter found")]
    NoAdapter,

    /// Failed to request a GPU device.
    #[error("device request failed: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),

    /// Failed to configure the surface.
    #[error("surface configuration failed: {0}")]
    SurfaceConfig(String),

    /// Failed to acquire the next surface texture.
    #[error("surface texture acquisition failed: {0}")]
    SurfaceTexture(#[from] wgpu::SurfaceError),

    /// A shader failed to compile.
    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),

    /// The texture atlas is full.
    #[error("texture atlas is full")]
    AtlasFull,

    /// A referenced pipeline was not found in the registry.
    #[error("pipeline not found: {0}")]
    PipelineNotFound(String),

    /// Naga rejected a user-supplied WGSL shader before pipeline creation.
    #[error("shader validation failed: {0}")]
    ShaderValidation(String),

    /// A shader pipeline with the given ID is already registered.
    #[error("shader ID {0} is already registered")]
    ShaderIdAlreadyRegistered(u32),

    /// The scene graph has reached its maximum node capacity.
    #[error("scene graph full: {0} nodes")]
    SceneGraphFull(usize),
}
