//! glTF/GLB asset loading — parse glTF files into esox mesh, material, and animation types.

use std::path::Path;

use glam::{Mat4, Quat, Vec3};

use super::material::{BlendMode3D, MaterialDescriptor, MaterialType};
use super::mesh::MeshData;
use super::texture::TextureHandle;
use super::transform::Transform;
use super::vertex::Vertex3D;

// ── Error ──

/// Errors that can occur during glTF loading.
#[derive(Debug)]
pub enum GltfError {
    /// I/O or parsing error from the gltf crate.
    Import(gltf::Error),
    /// A mesh primitive has no POSITION attribute.
    MissingPositions,
}

impl std::fmt::Display for GltfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GltfError::Import(e) => write!(f, "glTF import error: {e}"),
            GltfError::MissingPositions => write!(f, "mesh primitive has no POSITION attribute"),
        }
    }
}

impl std::error::Error for GltfError {}

impl From<gltf::Error> for GltfError {
    fn from(e: gltf::Error) -> Self {
        GltfError::Import(e)
    }
}

// ── Scene types ──

/// A loaded glTF scene containing meshes, materials, images, nodes, skeletons, and animations.
pub struct GltfScene {
    /// One `GltfMesh` per primitive (not per glTF mesh node).
    pub meshes: Vec<GltfMesh>,
    /// Materials mapped from glTF PBR metallic-roughness.
    pub materials: Vec<MaterialDescriptor>,
    /// Decoded RGBA8 images ready for GPU upload.
    pub images: Vec<GltfImage>,
    /// Scene hierarchy nodes.
    pub nodes: Vec<GltfNode>,
    /// Root node indices.
    pub roots: Vec<usize>,
    /// Skeleton data (one per glTF skin).
    pub skins: Vec<GltfSkin>,
    /// Animation clips.
    pub animations: Vec<AnimationClip>,
}

/// A single mesh primitive with CPU-side geometry.
pub struct GltfMesh {
    /// Vertex + index data ready for GPU upload.
    pub data: MeshData,
    /// Index into `GltfScene::materials`, or `None` for default material.
    pub material_index: Option<usize>,
    /// Per-vertex skinning data (joints + weights), present for skinned meshes.
    pub skin_data: Option<SkinningData>,
}

/// Decoded image data ready for GPU upload.
pub struct GltfImage {
    /// RGBA8 pixel data.
    pub data: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Whether this image should be uploaded as sRGB (true for albedo/emissive).
    pub srgb: bool,
}

/// A node in the scene hierarchy.
pub struct GltfNode {
    /// Local transform relative to parent.
    pub transform: Transform,
    /// Indices into `GltfScene::meshes` (one per primitive of the glTF mesh).
    pub mesh_indices: Vec<usize>,
    /// Index into `GltfScene::skins` if this node is skinned.
    pub skin_index: Option<usize>,
    /// Child node indices.
    pub children: Vec<usize>,
}

// ── Skeleton types ──

/// Skeleton data parsed from a glTF skin.
#[derive(Clone)]
pub struct GltfSkin {
    /// Joint names (if present in glTF).
    pub joint_names: Vec<Option<String>>,
    /// Parent joint index (-1 = root joint).
    pub parent_indices: Vec<i32>,
    /// Inverse bind matrices (one per joint).
    pub inverse_bind_matrices: Vec<Mat4>,
    /// Bind-pose local transforms per joint (rest pose from glTF nodes).
    pub bind_pose_transforms: Vec<Transform>,
    /// Number of joints.
    pub joint_count: usize,
}

/// Per-vertex skinning data (joint indices + weights).
pub struct SkinningData {
    /// 4 joint indices per vertex.
    pub joints: Vec<[u32; 4]>,
    /// 4 blend weights per vertex (sum ≈ 1.0).
    pub weights: Vec<[f32; 4]>,
}

// ── Animation types ──

/// A complete animation clip.
#[derive(Clone)]
pub struct AnimationClip {
    /// Name from glTF (if present).
    pub name: Option<String>,
    /// Duration in seconds.
    pub duration: f32,
    /// Animation channels (one per animated property per joint).
    pub channels: Vec<AnimChannel>,
}

/// A single animation channel targeting one joint's property.
#[derive(Clone)]
pub struct AnimChannel {
    /// Index of the targeted joint in the skeleton.
    pub joint_index: usize,
    /// Which property is animated.
    pub property: AnimProperty,
    /// Interpolation mode.
    pub interpolation: Interpolation,
    /// Keyframe timestamps in seconds.
    pub times: Vec<f32>,
    /// Keyframe values. Vec3 stored as [x, y, z, 0], Quat as [x, y, z, w].
    pub values: Vec<[f32; 4]>,
}

/// Animated property type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimProperty {
    Translation,
    Rotation,
    Scale,
}

/// Keyframe interpolation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Step,
    Linear,
    CubicSpline,
}

// ── Loading ──

impl GltfScene {
    /// Load a glTF or GLB file from disk.
    pub fn load(path: &Path) -> Result<Self, GltfError> {
        let (document, buffers, gltf_images) = gltf::import(path)?;

        // ── Images ──
        let images = decode_images(&gltf_images);

        // ── Track which images are used as sRGB vs linear ──
        // We'll update srgb flags based on material usage below.
        let mut image_srgb = vec![true; images.len()]; // default sRGB

        // ── Materials ──
        let materials: Vec<MaterialDescriptor> = document
            .materials()
            .map(|mat| convert_material(&mat, &mut image_srgb))
            .collect();

        // Apply srgb flags to images.
        let images: Vec<GltfImage> = images
            .into_iter()
            .enumerate()
            .map(|(i, mut img)| {
                img.srgb = image_srgb.get(i).copied().unwrap_or(true);
                img
            })
            .collect();

        // ── Meshes ──
        let mut meshes = Vec::new();
        // Map from (glTF mesh index, primitive index) to our mesh index.
        let mut mesh_index_map: Vec<Vec<usize>> = Vec::new();

        for gltf_mesh in document.meshes() {
            let mut prim_indices = Vec::new();
            for primitive in gltf_mesh.primitives() {
                let our_idx = meshes.len();
                prim_indices.push(our_idx);
                meshes.push(convert_primitive(&primitive, &buffers)?);
            }
            mesh_index_map.push(prim_indices);
        }

        // ── Nodes ──
        let nodes: Vec<GltfNode> = document
            .nodes()
            .map(|node| {
                let transform = convert_transform(&node);
                let mesh_indices = node
                    .mesh()
                    .map(|m| {
                        mesh_index_map
                            .get(m.index())
                            .cloned()
                            .unwrap_or_default()
                    })
                    .unwrap_or_default();
                let skin_index = node.skin().map(|s| s.index());
                let children: Vec<usize> = node.children().map(|c| c.index()).collect();
                GltfNode {
                    transform,
                    mesh_indices,
                    skin_index,
                    children,
                }
            })
            .collect();

        // ── Root nodes ──
        let roots: Vec<usize> = document
            .default_scene()
            .or_else(|| document.scenes().next())
            .map(|scene| scene.nodes().map(|n| n.index()).collect())
            .unwrap_or_default();

        // ── Skins ──
        let skins: Vec<GltfSkin> = document
            .skins()
            .map(|skin| convert_skin(&skin, &buffers, &document))
            .collect();

        // ── Animations ──
        let animations: Vec<AnimationClip> = document
            .animations()
            .map(|anim| convert_animation(&anim, &buffers, &document))
            .collect();

        Ok(GltfScene {
            meshes,
            materials,
            images,
            nodes,
            roots,
            skins,
            animations,
        })
    }
}

// ── Image decoding ──

fn decode_images(gltf_images: &[gltf::image::Data]) -> Vec<GltfImage> {
    gltf_images
        .iter()
        .map(|img| {
            let (data, width, height) = convert_image_to_rgba8(img);
            GltfImage {
                data,
                width,
                height,
                srgb: true, // default, updated later
            }
        })
        .collect()
}

fn convert_image_to_rgba8(img: &gltf::image::Data) -> (Vec<u8>, u32, u32) {
    let w = img.width;
    let h = img.height;
    let pixels = &img.pixels;

    match img.format {
        gltf::image::Format::R8G8B8A8 => (pixels.clone(), w, h),
        gltf::image::Format::R8G8B8 => {
            // Pad RGB → RGBA
            let pixel_count = (w * h) as usize;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for chunk in pixels.chunks_exact(3) {
                rgba.extend_from_slice(chunk);
                rgba.push(255);
            }
            (rgba, w, h)
        }
        gltf::image::Format::R8 => {
            let pixel_count = (w * h) as usize;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for &v in pixels {
                rgba.extend_from_slice(&[v, v, v, 255]);
            }
            (rgba, w, h)
        }
        gltf::image::Format::R8G8 => {
            let pixel_count = (w * h) as usize;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for chunk in pixels.chunks_exact(2) {
                rgba.extend_from_slice(&[chunk[0], chunk[1], 0, 255]);
            }
            (rgba, w, h)
        }
        _ => {
            // For 16-bit formats, convert to 8-bit by taking the high byte.
            let format_bpp = match img.format {
                gltf::image::Format::R16 => 2,
                gltf::image::Format::R16G16 => 4,
                gltf::image::Format::R16G16B16 => 6,
                gltf::image::Format::R16G16B16A16 => 8,
                _ => 4,
            };
            let channels = format_bpp / 2;
            let pixel_count = (w * h) as usize;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for pixel in pixels.chunks_exact(format_bpp) {
                for c in 0..channels.min(4) {
                    // Take high byte of u16le
                    rgba.push(pixel[c * 2 + 1]);
                }
                for _ in channels..3 {
                    rgba.push(0);
                }
                if channels < 4 {
                    rgba.push(255);
                }
            }
            (rgba, w, h)
        }
    }
}

// ── Material conversion ──

fn convert_material(
    mat: &gltf::Material<'_>,
    image_srgb: &mut [bool],
) -> MaterialDescriptor {
    let pbr = mat.pbr_metallic_roughness();
    let base_color = pbr.base_color_factor();

    let albedo_tex_idx = pbr
        .base_color_texture()
        .map(|info| info.texture().source().index());

    let normal_tex_idx = mat
        .normal_texture()
        .map(|info| info.texture().source().index());

    let mr_tex_idx = pbr
        .metallic_roughness_texture()
        .map(|info| info.texture().source().index());

    let emissive_tex_idx = mat
        .emissive_texture()
        .map(|info| info.texture().source().index());

    let normal_scale = mat.normal_texture().map(|t| t.scale()).unwrap_or(1.0);

    // Mark data textures as linear (not sRGB).
    if let Some(idx) = normal_tex_idx {
        if idx < image_srgb.len() {
            image_srgb[idx] = false;
        }
    }
    if let Some(idx) = mr_tex_idx {
        if idx < image_srgb.len() {
            image_srgb[idx] = false;
        }
    }

    let emissive = mat.emissive_factor();

    let blend_mode = match mat.alpha_mode() {
        gltf::material::AlphaMode::Opaque | gltf::material::AlphaMode::Mask => {
            BlendMode3D::Opaque
        }
        gltf::material::AlphaMode::Blend => BlendMode3D::AlphaBlend,
    };

    // Use TextureHandle indices that correspond to image indices.
    // These are temporary placeholders; the renderer's upload_gltf_scene will remap them.
    MaterialDescriptor {
        material_type: MaterialType::PBR,
        albedo: base_color,
        emissive,
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        blend_mode,
        double_sided: mat.double_sided(),
        depth_write: blend_mode == BlendMode3D::Opaque,
        texture: albedo_tex_idx.map(|i| TextureHandle(i as u32)),
        normal_texture: normal_tex_idx.map(|i| TextureHandle(i as u32)),
        metallic_roughness_texture: mr_tex_idx.map(|i| TextureHandle(i as u32)),
        emissive_texture: emissive_tex_idx.map(|i| TextureHandle(i as u32)),
        normal_scale,
        toon_bands: 3.0,
        rim_power: 3.0,
        rim_intensity: 0.4,
    }
}

// ── Mesh conversion ──

fn convert_primitive(
    primitive: &gltf::Primitive<'_>,
    buffers: &[gltf::buffer::Data],
) -> Result<GltfMesh, GltfError> {
    let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));

    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or(GltfError::MissingPositions)?
        .collect();

    let vertex_count = positions.len();

    let normals: Vec<[f32; 3]> = reader
        .read_normals()
        .map(|iter| iter.collect())
        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; vertex_count]);

    let uvs: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .map(|iter| iter.into_f32().collect())
        .unwrap_or_else(|| vec![[0.0, 0.0]; vertex_count]);

    let colors: Vec<[f32; 4]> = reader
        .read_colors(0)
        .map(|iter| iter.into_rgba_f32().collect())
        .unwrap_or_else(|| vec![[1.0, 1.0, 1.0, 1.0]; vertex_count]);

    let tangents: Option<Vec<[f32; 4]>> = reader.read_tangents().map(|iter| iter.collect());

    let indices: Vec<u32> = reader
        .read_indices()
        .map(|iter| iter.into_u32().collect())
        .unwrap_or_else(|| (0..vertex_count as u32).collect());

    // Build vertices.
    let mut vertices: Vec<Vertex3D> = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        vertices.push(Vertex3D {
            position: positions[i],
            normal: normals[i],
            uv: uvs[i],
            color: colors[i],
            tangent: tangents.as_ref().map(|t| t[i]).unwrap_or([0.0, 0.0, 0.0, 1.0]),
        });
    }

    // Compute tangents if not provided.
    if tangents.is_none() && !indices.is_empty() {
        compute_tangents(&mut vertices, &indices);
    }

    // Read skinning data if present.
    let skin_data = read_skin_data(primitive, buffers, vertex_count);

    let material_index = primitive.material().index();

    Ok(GltfMesh {
        data: MeshData::new(vertices, indices),
        material_index,
        skin_data,
    })
}

fn read_skin_data(
    primitive: &gltf::Primitive<'_>,
    buffers: &[gltf::buffer::Data],
    vertex_count: usize,
) -> Option<SkinningData> {
    let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));
    let joints_iter = reader.read_joints(0)?;
    let weights_iter = reader.read_weights(0)?;

    let joints: Vec<[u32; 4]> = joints_iter
        .into_u16()
        .map(|j| [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32])
        .collect();

    let weights: Vec<[f32; 4]> = weights_iter.into_f32().collect();

    if joints.len() != vertex_count || weights.len() != vertex_count {
        return None;
    }

    Some(SkinningData { joints, weights })
}

// ── Tangent computation ──

fn compute_tangents(vertices: &mut [Vertex3D], indices: &[u32]) {
    let n = vertices.len();
    let mut tangents = vec![[0.0f32; 3]; n];

    for tri in indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        let p0 = Vec3::from(vertices[i0].position);
        let p1 = Vec3::from(vertices[i1].position);
        let p2 = Vec3::from(vertices[i2].position);

        let uv0 = vertices[i0].uv;
        let uv1 = vertices[i1].uv;
        let uv2 = vertices[i2].uv;

        let dp1 = p1 - p0;
        let dp2 = p2 - p0;
        let duv1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
        let duv2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];

        let det = duv1[0] * duv2[1] - duv1[1] * duv2[0];
        if det.abs() < 1e-8 {
            continue;
        }
        let inv_det = 1.0 / det;

        let t = (dp1 * duv2[1] - dp2 * duv1[1]) * inv_det;

        for &idx in &[i0, i1, i2] {
            tangents[idx][0] += t.x;
            tangents[idx][1] += t.y;
            tangents[idx][2] += t.z;
        }
    }

    for (i, vert) in vertices.iter_mut().enumerate() {
        let t = Vec3::from(tangents[i]);
        let len = t.length();
        if len > 1e-6 {
            let normalized = t / len;
            vert.tangent = [normalized.x, normalized.y, normalized.z, 1.0];
        }
        // else leave as [0,0,0,1] — shader will skip normal mapping
    }
}

// ── Transform conversion ──

fn convert_transform(node: &gltf::Node<'_>) -> Transform {
    let (translation, rotation, scale) = node.transform().decomposed();
    Transform {
        position: Vec3::from(translation),
        rotation: Quat::from_array(rotation),
        scale: Vec3::from(scale),
    }
}

// ── Skin conversion ──

fn convert_skin(
    skin: &gltf::Skin<'_>,
    buffers: &[gltf::buffer::Data],
    _document: &gltf::Document,
) -> GltfSkin {
    let joints: Vec<gltf::Node<'_>> = skin.joints().collect();
    let joint_count = joints.len();

    let joint_names: Vec<Option<String>> = joints
        .iter()
        .map(|j| j.name().map(String::from))
        .collect();

    // Build parent index map: for each joint, find its parent in the joint list.
    let parent_indices: Vec<i32> = joints
        .iter()
        .map(|joint| {
            // Look through all joints' children to find who is parent of this joint.
            for (ji, j) in joints.iter().enumerate() {
                for child in j.children() {
                    if child.index() == joint.index() {
                        return ji as i32;
                    }
                }
            }
            -1 // root joint
        })
        .collect();

    // Read inverse bind matrices.
    let inverse_bind_matrices = skin
        .reader(|buf| Some(&buffers[buf.index()]))
        .read_inverse_bind_matrices()
        .map(|iter| iter.map(|m| Mat4::from_cols_array_2d(&m)).collect())
        .unwrap_or_else(|| vec![Mat4::IDENTITY; joint_count]);

    // Capture bind-pose local transforms from the glTF node tree.
    let bind_pose_transforms: Vec<Transform> = joints.iter().map(|j| convert_transform(j)).collect();

    GltfSkin {
        joint_names,
        parent_indices,
        inverse_bind_matrices,
        bind_pose_transforms,
        joint_count,
    }
}

// ── Animation conversion ──

fn convert_animation(
    anim: &gltf::Animation<'_>,
    buffers: &[gltf::buffer::Data],
    document: &gltf::Document,
) -> AnimationClip {
    let name = anim.name().map(String::from);

    // We need a mapping from node index to joint index.
    // This works for the first skin; for multi-skin files it would need refinement.
    let joint_map: std::collections::HashMap<usize, usize> = document
        .skins()
        .next()
        .map(|skin| {
            skin.joints()
                .enumerate()
                .map(|(ji, node)| (node.index(), ji))
                .collect()
        })
        .unwrap_or_default();

    let mut duration: f32 = 0.0;
    let mut channels = Vec::new();

    for channel in anim.channels() {
        let target = channel.target();
        let node_idx = target.node().index();

        let joint_index = match joint_map.get(&node_idx) {
            Some(&ji) => ji,
            None => continue, // skip channels targeting non-joint nodes
        };

        let property = match target.property() {
            gltf::animation::Property::Translation => AnimProperty::Translation,
            gltf::animation::Property::Rotation => AnimProperty::Rotation,
            gltf::animation::Property::Scale => AnimProperty::Scale,
            _ => continue, // skip morph targets
        };

        let sampler = channel.sampler();
        let interpolation = match sampler.interpolation() {
            gltf::animation::Interpolation::Step => Interpolation::Step,
            gltf::animation::Interpolation::Linear => Interpolation::Linear,
            gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
        };

        let reader = channel.reader(|buf| Some(&buffers[buf.index()]));
        let times: Vec<f32> = reader.read_inputs().map(|iter| iter.collect()).unwrap_or_default();
        if let Some(&last) = times.last() {
            duration = duration.max(last);
        }

        let values: Vec<[f32; 4]> = match reader.read_outputs() {
            Some(gltf::animation::util::ReadOutputs::Translations(iter)) => {
                iter.map(|t| [t[0], t[1], t[2], 0.0]).collect()
            }
            Some(gltf::animation::util::ReadOutputs::Rotations(iter)) => {
                iter.into_f32().map(|r| r).collect()
            }
            Some(gltf::animation::util::ReadOutputs::Scales(iter)) => {
                iter.map(|s| [s[0], s[1], s[2], 0.0]).collect()
            }
            _ => continue,
        };

        channels.push(AnimChannel {
            joint_index,
            property,
            interpolation,
            times,
            values,
        });
    }

    AnimationClip {
        name,
        duration,
        channels,
    }
}

// ── GPU upload helper ──

/// Handles returned from uploading a glTF scene to the renderer.
pub struct GltfSceneHandles {
    /// Texture handles (one per glTF image, None if upload failed).
    pub textures: Vec<Option<TextureHandle>>,
    /// Material handles (one per glTF material).
    pub materials: Vec<super::material::MaterialHandle>,
    /// Mesh handles (one per primitive).
    pub meshes: Vec<super::mesh::MeshHandle>,
    /// Nodes (copied from scene, with index references intact).
    pub nodes: Vec<GltfNode>,
    /// Root node indices.
    pub roots: Vec<usize>,
    /// Skins (moved from scene).
    pub skins: Vec<GltfSkin>,
    /// Animations (moved from scene).
    pub animations: Vec<AnimationClip>,
    /// Per-mesh skinning data (same length as meshes, None for non-skinned).
    pub skin_data: Vec<Option<SkinningData>>,
    /// Per-mesh material index from glTF (same length as meshes, None = default material).
    pub mesh_material_indices: Vec<Option<usize>>,
    /// Per-mesh skinned mesh index into `Renderer3D::skinned_meshes` (None for non-skinned).
    pub skinned_mesh_indices: Vec<Option<usize>>,
}

impl super::renderer::Renderer3D {
    /// Upload all images, materials, and meshes from a loaded glTF scene.
    pub fn upload_gltf_scene(
        &mut self,
        gpu: &crate::pipeline::GpuContext,
        scene: GltfScene,
    ) -> GltfSceneHandles {
        // Upload images → textures.
        // Use map (not filter_map) so indices stay aligned with materials' image references.
        let textures: Vec<Option<TextureHandle>> = scene
            .images
            .iter()
            .map(|img| {
                if img.srgb {
                    self.upload_texture(gpu, img.width, img.height, &img.data)
                } else {
                    self.upload_texture_linear(gpu, img.width, img.height, &img.data)
                }
            })
            .collect();

        // Create materials, remapping image indices → TextureHandles.
        let materials: Vec<super::material::MaterialHandle> = scene
            .materials
            .iter()
            .map(|desc| {
                let mut remapped = desc.clone();
                remapped.texture = remap_texture_handle(desc.texture, &textures);
                remapped.normal_texture = remap_texture_handle(desc.normal_texture, &textures);
                remapped.metallic_roughness_texture =
                    remap_texture_handle(desc.metallic_roughness_texture, &textures);
                remapped.emissive_texture =
                    remap_texture_handle(desc.emissive_texture, &textures);
                self.create_material(gpu, &remapped)
            })
            .collect();

        // Build mesh → skin mapping from nodes.
        let mut mesh_skin_map = std::collections::HashMap::new();
        for node in &scene.nodes {
            if let Some(skin_idx) = node.skin_index {
                for &mesh_idx in &node.mesh_indices {
                    mesh_skin_map.insert(mesh_idx, skin_idx);
                }
            }
        }

        // Upload meshes, using skinned upload for meshes with skinning data.
        let mut mesh_handles = Vec::with_capacity(scene.meshes.len());
        let mut skin_data = Vec::with_capacity(scene.meshes.len());
        let mut skinned_mesh_indices = Vec::with_capacity(scene.meshes.len());
        let mut mesh_material_indices = Vec::with_capacity(scene.meshes.len());
        for (i, mesh) in scene.meshes.into_iter().enumerate() {
            match (&mesh.skin_data, mesh_skin_map.get(&i)) {
                (Some(sd), Some(&skin_idx)) => {
                    let joint_count = scene.skins[skin_idx].joint_count as u32;
                    let (handle, si) =
                        self.upload_skinned_mesh(gpu, &mesh.data, sd, joint_count);
                    mesh_handles.push(handle);
                    skinned_mesh_indices.push(Some(si));
                }
                _ => {
                    mesh_handles.push(self.upload_mesh(gpu, &mesh.data));
                    skinned_mesh_indices.push(None);
                }
            }
            skin_data.push(mesh.skin_data);
            mesh_material_indices.push(mesh.material_index);
        }

        GltfSceneHandles {
            textures,
            materials,
            meshes: mesh_handles,
            nodes: scene.nodes,
            roots: scene.roots,
            skins: scene.skins,
            animations: scene.animations,
            skin_data,
            mesh_material_indices,
            skinned_mesh_indices,
        }
    }
}

fn remap_texture_handle(
    original: Option<TextureHandle>,
    uploaded: &[Option<TextureHandle>],
) -> Option<TextureHandle> {
    original.and_then(|h| uploaded.get(h.0 as usize).copied().flatten())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tangent_computation_on_plane() {
        // A simple quad in XZ plane, UV = position XZ.
        let mut vertices = vec![
            Vertex3D::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0]),
            Vertex3D::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0]),
            Vertex3D::new([1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0]),
            Vertex3D::new([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0]),
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];

        compute_tangents(&mut vertices, &indices);

        // Tangent should point along +X (U direction).
        for v in &vertices {
            let t = Vec3::from_slice(&v.tangent[..3]);
            assert!(
                (t.length() - 1.0).abs() < 1e-4,
                "tangent should be unit length, got {}",
                t.length()
            );
            assert!(
                t.dot(Vec3::X).abs() > 0.99,
                "tangent should point along X, got {:?}",
                t
            );
        }
    }

    #[test]
    fn tangent_degenerate_uv_leaves_zero() {
        // All UVs identical → degenerate, tangent stays zero.
        let mut vertices = vec![
            Vertex3D::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5]),
            Vertex3D::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5]),
            Vertex3D::new([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.5, 0.5]),
        ];
        let indices = vec![0, 1, 2];

        compute_tangents(&mut vertices, &indices);

        for v in &vertices {
            assert_eq!(
                v.tangent,
                [0.0, 0.0, 0.0, 1.0],
                "degenerate UVs should leave tangent as zero"
            );
        }
    }

    #[test]
    fn convert_image_rgb_to_rgba() {
        let img = gltf::image::Data {
            pixels: vec![255, 0, 0, 0, 255, 0],
            width: 2,
            height: 1,
            format: gltf::image::Format::R8G8B8,
        };
        let (rgba, w, h) = convert_image_to_rgba8(&img);
        assert_eq!(w, 2);
        assert_eq!(h, 1);
        assert_eq!(rgba.len(), 8);
        assert_eq!(&rgba[0..4], &[255, 0, 0, 255]);
        assert_eq!(&rgba[4..8], &[0, 255, 0, 255]);
    }

    #[test]
    fn convert_image_rgba_passthrough() {
        let img = gltf::image::Data {
            pixels: vec![1, 2, 3, 4],
            width: 1,
            height: 1,
            format: gltf::image::Format::R8G8B8A8,
        };
        let (rgba, w, h) = convert_image_to_rgba8(&img);
        assert_eq!(w, 1);
        assert_eq!(h, 1);
        assert_eq!(rgba, vec![1, 2, 3, 4]);
    }

    #[test]
    fn remap_texture_handle_works() {
        let uploaded = vec![Some(TextureHandle(10)), Some(TextureHandle(11)), Some(TextureHandle(12))];
        assert_eq!(
            remap_texture_handle(Some(TextureHandle(0)), &uploaded),
            Some(TextureHandle(10))
        );
        assert_eq!(
            remap_texture_handle(Some(TextureHandle(2)), &uploaded),
            Some(TextureHandle(12))
        );
        assert_eq!(remap_texture_handle(None, &uploaded), None);
        assert_eq!(
            remap_texture_handle(Some(TextureHandle(99)), &uploaded),
            None
        );
        // Failed upload (None) at index 1.
        let with_gap = vec![Some(TextureHandle(10)), None, Some(TextureHandle(12))];
        assert_eq!(
            remap_texture_handle(Some(TextureHandle(1)), &with_gap),
            None
        );
    }
}
