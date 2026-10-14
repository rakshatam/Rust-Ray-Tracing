use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use image::imageops::FilterType;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowBuilder};

// Pinned to an immutable Khronos Group glTF-Sample-Assets commit SHA (723ffc6)
const SPONZA_COMMIT_SHA: &str = "723ffc6";
const SPONZA_BASE_URL: &str =
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/723ffc6/Models/Sponza/glTF";
const SPONZA_GLTF_URL: &str =
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/723ffc6/Models/Sponza/glTF/Sponza.gltf";
const SPONZA_BIN_URL: &str =
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/723ffc6/Models/Sponza/glTF/Sponza.bin";
const ASSET_DIR: &str = "assets/sponza/glTF";

// Hard upper bound per HTTP asset read (32 MB) to prevent unbounded memory allocation
const MAX_ASSET_BYTES: u64 = 32 * 1024 * 1024;
const TEX_DIM: u32 = 512;

// GPU traversal stack holds 32 entries; enforce max tree depth <= 31 so no subtree is ever dropped
const GPU_MAX_STACK_DEPTH: u32 = 32;
const MAX_BVH_DEPTH: u32 = GPU_MAX_STACK_DEPTH - 1;

static TEMP_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

// ============================================================================
// GPU Data Structures (16-byte aligned for WGSL / SPIR-V / DXIL compatibility)
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    pub cam_pos_fov: [f32; 4],
    pub cam_fwd_aspect: [f32; 4],
    pub cam_right_sun: [f32; 4],
    pub cam_up_rough: [f32; 4],
    pub frame_info: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuTriangle {
    pub v0: [f32; 4],           // xyz: v0, w: uv0.x
    pub e1: [f32; 4],           // xyz: e1, w: uv0.y
    pub e2: [f32; 4],           // xyz: e2, w: uv1.x
    pub n0: [f32; 4],           // xyz: n0, w: uv1.y
    pub n1: [f32; 4],           // xyz: n1, w: uv2.x
    pub n2: [f32; 4],           // xyz: n2, w: uv2.y
    pub albedo_rough: [f32; 4], // xyz: base_color_factor, w: roughness_factor
    pub emissive_mat: [f32; 4], // x: tex_layer_idx, y: alpha_cutoff, z: metallic_factor, w: pad
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuBvhNode {
    pub aabb_min: [f32; 3],
    pub left_or_first: u32,
    pub aabb_max: [f32; 3],
    pub tri_count: u32,
}

#[derive(Clone, Copy, Debug)]
struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    fn grow(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    fn union(&self, other: &Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    fn area(&self) -> f32 {
        let d = (self.max - self.min).max(Vec3::ZERO);
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }
}

pub struct SceneAssets {
    pub triangles: Vec<GpuTriangle>,
    pub texture_layers: Vec<Vec<u8>>, // Each layer is TEX_DIM * TEX_DIM * 4 bytes (RGBA8 sRGB)
}

// ============================================================================
// Phase 1: Startup Host Environment & CLI Configuration Banner
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackendPreference {
    Auto,
    Dx12,
    Vulkan,
}

fn print_startup_banner() -> BackendPreference {
    println!("========================================================================");
    println!(" [PHASE 1] HOST ENVIRONMENT & CLI CONFIGURATION BANNER");
    println!("========================================================================");

    let pkg_name = env!("CARGO_PKG_NAME");
    let pkg_ver = env!("CARGO_PKG_VERSION");
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!(" -> Binary Package   : {} v{}", pkg_name, pkg_ver);
    println!(
        " -> Host Platform    : {} ({}) | Logical CPU Cores: {}",
        os, arch, threads
    );
    println!(
        " -> Asset Pin SHA    : KhronosGroup/glTF-Sample-Assets @ {}",
        SPONZA_COMMIT_SHA
    );

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--vulkan") {
        println!(" -> CLI Override     : Forcing Vulkan Backend");
        BackendPreference::Vulkan
    } else if args.iter().any(|a| a == "--dx12") {
        println!(" -> CLI Override     : Forcing DirectX 12 Backend");
        BackendPreference::Dx12
    } else {
        println!(" -> Backend Mode     : Auto-Select Optimal Discrete Adapter (DX12/Vulkan)");
        BackendPreference::Auto
    }
}

// ============================================================================
// Phase 2: Atomic Sponza Asset Provisioner, Texture Downloader & glTF Importer
// ============================================================================

/// Downloads `url` with a hard size limit (`MAX_ASSET_BYTES`), verifies `Content-Length`
/// if provided by the server, writes to an isolated `.part` temporary file, flushes to disk,
/// and atomically renames to `dest` so interrupted downloads never corrupt the cache.
fn download_file_atomic(url: &str, dest: &Path, min_expected_bytes: usize, quiet: bool) -> Result<()> {
    if !quiet {
        println!(" -> Downloading      : {} -> {}", url, dest.display());
    }
    let response = ureq::get(url)
        .timeout(std::time::Duration::from_secs(45))
        .call()
        .with_context(|| format!("HTTP GET failed for {}", url))?;

    let expected_len: Option<usize> = response
        .header("Content-Length")
        .and_then(|s| s.parse::<usize>().ok());

    let mut bytes = Vec::with_capacity(expected_len.unwrap_or(64 * 1024).min(MAX_ASSET_BYTES as usize));
    let mut bounded_reader = response.into_reader().take(MAX_ASSET_BYTES + 1);
    bounded_reader.read_to_end(&mut bytes)?;

    if (bytes.len() as u64) > MAX_ASSET_BYTES {
        anyhow::bail!(
            "Download from {} exceeded safety cap of {} bytes",
            url,
            MAX_ASSET_BYTES
        );
    }

    if let Some(cl) = expected_len {
        if bytes.len() != cl {
            anyhow::bail!(
                "Truncated HTTP stream for {}: received {} of {} bytes",
                url,
                bytes.len(),
                cl
            );
        }
    }

    if bytes.len() < min_expected_bytes {
        anyhow::bail!(
            "Downloaded payload from {} too small ({} bytes < {} min)",
            url,
            bytes.len(),
            min_expected_bytes
        );
    }

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    let seq = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_path = dest.with_extension(format!("part.{}.{}", std::process::id(), seq));

    {
        let mut tmp_file = fs::File::create(&tmp_path)
            .with_context(|| format!("Failed to create temp file {}", tmp_path.display()))?;
        tmp_file.write_all(&bytes)?;
        tmp_file.sync_all()?;
    }

    if let Err(e) = fs::rename(&tmp_path, dest) {
        let _ = fs::remove_file(&tmp_path);
        return Err(e).with_context(|| {
            format!(
                "Failed to atomically rename {} to {}",
                tmp_path.display(),
                dest.display()
            )
        });
    }

    if !quiet {
        println!(
            " -> Installed        : {} ({:.2} MB)",
            dest.display(),
            bytes.len() as f64 / (1024.0 * 1024.0)
        );
    }
    Ok(())
}

fn ensure_sponza_geometry_files(force_redownload: bool) -> Option<PathBuf> {
    println!("========================================================================");
    println!(" [PHASE 2] SPONZA ATRIUM ASSET & TEXTURE ARRAY PROVISIONER");
    println!("========================================================================");

    let gltf_path = Path::new(ASSET_DIR).join("Sponza.gltf");
    let bin_path = Path::new(ASSET_DIR).join("Sponza.bin");

    if force_redownload {
        let _ = fs::remove_file(&gltf_path);
        let _ = fs::remove_file(&bin_path);
    }

    let gltf_valid = gltf_path
        .metadata()
        .map(|m| m.len() >= 100_000)
        .unwrap_or(false);
    if !gltf_valid {
        if let Err(e) = download_file_atomic(SPONZA_GLTF_URL, &gltf_path, 100_000, false) {
            println!(" [!] Network fetch warning (Sponza.gltf): {}", e);
            return None;
        }
    } else {
        println!(" -> Sponza Manifest  : FOUND at {}", gltf_path.display());
    }

    let bin_valid = bin_path
        .metadata()
        .map(|m| m.len() >= 9_000_000)
        .unwrap_or(false);
    if !bin_valid {
        if let Err(e) = download_file_atomic(SPONZA_BIN_URL, &bin_path, 9_000_000, false) {
            println!(" [!] Network fetch warning (Sponza.bin): {}", e);
            return None;
        }
    } else {
        println!(" -> Sponza Geometry  : FOUND at {}", bin_path.display());
    }

    Some(gltf_path)
}

fn make_solid_layer(rgba: [u8; 4]) -> Vec<u8> {
    let mut buf = Vec::with_capacity((TEX_DIM * TEX_DIM * 4) as usize);
    for _ in 0..(TEX_DIM * TEX_DIM) {
        buf.extend_from_slice(&rgba);
    }
    buf
}

fn load_sponza_gltf_with_textures(gltf_path: &Path) -> Result<SceneAssets> {
    println!(" -> Parsing glTF     : {}", gltf_path.display());
    let file = fs::File::open(gltf_path)?;
    let reader = std::io::BufReader::new(file);
    let gltf = gltf::Gltf::from_reader(reader)?;

    let bin_path = gltf_path.with_file_name("Sponza.bin");
    let bin_data = fs::read(&bin_path)
        .with_context(|| format!("Failed to read binary buffer {}", bin_path.display()))?;

    // Verify binary buffer size against the byte_length declared inside Sponza.gltf
    if let Some(first_buf) = gltf.buffers().next() {
        if bin_data.len() < first_buf.length() {
            anyhow::bail!(
                "Cached Sponza.bin is truncated ({} bytes < {} bytes declared in glTF)",
                bin_data.len(),
                first_buf.length()
            );
        }
    }

    // Collect all unique baseColorTexture image URIs referenced by Sponza materials
    let mut unique_uris: Vec<String> = Vec::new();
    let mut uri_to_layer: HashMap<String, usize> = HashMap::new();

    // Layer 0 is a solid warm-white fallback texture for untextured primitives
    let mut texture_layers: Vec<Vec<u8>> = vec![make_solid_layer([235, 230, 220, 255])];

    for mat in gltf.materials() {
        if let Some(tex_info) = mat.pbr_metallic_roughness().base_color_texture() {
            if let gltf::image::Source::Uri { uri, .. } = tex_info.texture().source().source() {
                if !uri_to_layer.contains_key(uri) {
                    let layer_idx = unique_uris.len() + 1;
                    uri_to_layer.insert(uri.to_string(), layer_idx);
                    unique_uris.push(uri.to_string());
                }
            }
        }
    }

    // Download any missing Sponza baseColor textures in parallel (8 threads) using atomic temp-file rename
    let missing_downloads: Vec<(usize, String, PathBuf)> = unique_uris
        .iter()
        .enumerate()
        .filter_map(|(i, uri)| {
            let local_path = Path::new(ASSET_DIR).join(uri);
            let exists_and_nonempty = local_path
                .metadata()
                .map(|m| m.len() >= 512)
                .unwrap_or(false);
            if !exists_and_nonempty {
                Some((i + 1, uri.clone(), local_path))
            } else {
                None
            }
        })
        .collect();

    let total_textures = unique_uris.len();
    if !missing_downloads.is_empty() {
        println!(
            " -> Provisioning     : Downloading {} Sponza baseColor textures in parallel...",
            missing_downloads.len()
        );
        let queue = Arc::new(Mutex::new(missing_downloads));
        let worker_count = 8.min(total_textures.max(1));

        std::thread::scope(|s| {
            for _ in 0..worker_count {
                let q = Arc::clone(&queue);
                s.spawn(move || loop {
                    let item = {
                        let mut lock = q.lock().unwrap();
                        lock.pop()
                    };
                    let Some((idx, uri, dest)) = item else {
                        break;
                    };
                    let url = format!("{}/{}", SPONZA_BASE_URL, uri);
                    match download_file_atomic(&url, &dest, 512, true) {
                        Ok(_) => println!(
                            "    [Texture {:02}/{}] Installed {}",
                            idx, total_textures, uri
                        ),
                        Err(e) => println!(
                            "    [!] Texture {:02}/{} download warning ({}): {}",
                            idx, total_textures, uri, e
                        ),
                    }
                });
            }
        });
    } else {
        println!(
            " -> Sponza Textures  : All {} baseColor textures verified in cache",
            total_textures
        );
    }

    // Decode & resize all unique textures in parallel across CPU cores.
    // If a cached texture file is corrupted, delete it and re-fetch once atomically.
    println!(
        " -> Decoding Textures: Building {}x{} RGBA8 sRGB GPU Texture2DArray ({} layers)...",
        TEX_DIM,
        TEX_DIM,
        total_textures + 1
    );
    let decode_start = Instant::now();
    let decoded_layers: Mutex<Vec<(usize, Vec<u8>)>> =
        Mutex::new(Vec::with_capacity(total_textures));
    let decode_queue: Mutex<Vec<(usize, String)>> =
        Mutex::new(unique_uris.into_iter().enumerate().collect());

    let cpu_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    std::thread::scope(|s| {
        for _ in 0..cpu_threads {
            let q = &decode_queue;
            let out = &decoded_layers;
            s.spawn(move || loop {
                let item = {
                    let mut lock = q.lock().unwrap();
                    lock.pop()
                };
                let Some((idx, uri)) = item else {
                    break;
                };
                let path = Path::new(ASSET_DIR).join(&uri);
                let decoded = image::open(&path).or_else(|_| {
                    // Self-heal corrupted cached texture file by purging and re-downloading
                    let _ = fs::remove_file(&path);
                    let url = format!("{}/{}", SPONZA_BASE_URL, uri);
                    let _ = download_file_atomic(&url, &path, 512, true);
                    image::open(&path)
                });

                let layer_bytes = match decoded {
                    Ok(img) => {
                        let resized = img.resize_exact(TEX_DIM, TEX_DIM, FilterType::Triangle);
                        resized.to_rgba8().into_raw()
                    }
                    Err(_) => make_solid_layer([210, 195, 175, 255]),
                };
                out.lock().unwrap().push((idx, layer_bytes));
            });
        }
    });

    let mut sorted_layers = decoded_layers.into_inner().unwrap();
    sorted_layers.sort_by_key(|(idx, _)| *idx);
    for (_, bytes) in sorted_layers {
        texture_layers.push(bytes);
    }
    println!(
        " -> Texture Array OK : {} layers decoded in {:.2?}",
        texture_layers.len(),
        decode_start.elapsed()
    );

    let mut triangles = Vec::with_capacity(265_000);
    for scene in gltf.scenes() {
        for node in scene.nodes() {
            traverse_gltf_node(
                &node,
                Mat4::IDENTITY,
                &bin_data,
                &uri_to_layer,
                &mut triangles,
            );
        }
    }

    if triangles.is_empty() {
        anyhow::bail!("glTF parsed 0 triangles");
    }

    println!(
        " -> Scene Imported   : {} textured triangles extracted from Sponza Atrium",
        triangles.len()
    );
    Ok(SceneAssets {
        triangles,
        texture_layers,
    })
}

fn traverse_gltf_node(
    node: &gltf::Node,
    parent_xform: Mat4,
    bin_data: &[u8],
    uri_to_layer: &HashMap<String, usize>,
    out_tris: &mut Vec<GpuTriangle>,
) {
    let local_xform = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world_xform = parent_xform * local_xform;
    let normal_xform = world_xform.inverse().transpose();

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }

            let reader = primitive.reader(|_buffer| Some(bin_data));
            let positions: Vec<[f32; 3]> = match reader.read_positions() {
                Some(iter) => iter.collect(),
                None => continue,
            };

            let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|iter| iter.collect());
            let tex_coords: Option<Vec<[f32; 2]>> = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().collect());
            let indices: Vec<u32> = match reader.read_indices() {
                Some(iter) => iter.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };

            let mat = primitive.material();
            let pbr = mat.pbr_metallic_roughness();
            let base_color = pbr.base_color_factor();
            let roughness = pbr.roughness_factor().clamp(0.08, 0.98);
            let metallic = pbr.metallic_factor();

            let tex_layer = pbr
                .base_color_texture()
                .and_then(|info| match info.texture().source().source() {
                    gltf::image::Source::Uri { uri, .. } => uri_to_layer.get(uri).copied(),
                    _ => None,
                })
                .unwrap_or(0) as f32;

            let alpha_cutoff = match mat.alpha_mode() {
                gltf::material::AlphaMode::Mask => mat.alpha_cutoff().unwrap_or(0.5),
                _ => 0.0,
            };

            for tri in indices.chunks_exact(3) {
                let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
                if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
                    continue;
                }

                let p0 = world_xform.transform_point3(Vec3::from(positions[i0]));
                let p1 = world_xform.transform_point3(Vec3::from(positions[i1]));
                let p2 = world_xform.transform_point3(Vec3::from(positions[i2]));

                let e1 = p1 - p0;
                let e2 = p2 - p0;
                let face_n = e1.cross(e2).normalize_or_zero();
                if face_n.length_squared() < 1e-6 {
                    continue;
                }

                let (n0, n1, n2) = if let Some(ref n_list) = normals {
                    let wn0 = normal_xform
                        .transform_vector3(Vec3::from(n_list[i0]))
                        .normalize_or_zero();
                    let wn1 = normal_xform
                        .transform_vector3(Vec3::from(n_list[i1]))
                        .normalize_or_zero();
                    let wn2 = normal_xform
                        .transform_vector3(Vec3::from(n_list[i2]))
                        .normalize_or_zero();
                    (
                        if wn0.length_squared() > 0.1 { wn0 } else { face_n },
                        if wn1.length_squared() > 0.1 { wn1 } else { face_n },
                        if wn2.length_squared() > 0.1 { wn2 } else { face_n },
                    )
                } else {
                    (face_n, face_n, face_n)
                };

                let (uv0, uv1, uv2) = if let Some(ref uvs) = tex_coords {
                    (uvs[i0], uvs[i1], uvs[i2])
                } else {
                    ([p0.x, p0.z], [p1.x, p1.z], [p2.x, p2.z])
                };

                out_tris.push(GpuTriangle {
                    v0: [p0.x, p0.y, p0.z, uv0[0]],
                    e1: [e1.x, e1.y, e1.z, uv0[1]],
                    e2: [e2.x, e2.y, e2.z, uv1[0]],
                    n0: [n0.x, n0.y, n0.z, uv1[1]],
                    n1: [n1.x, n1.y, n1.z, uv2[0]],
                    n2: [n2.x, n2.y, n2.z, uv2[1]],
                    albedo_rough: [base_color[0], base_color[1], base_color[2], roughness],
                    emissive_mat: [tex_layer, alpha_cutoff, metallic, 0.0],
                });
            }
        }
    }

    for child in node.children() {
        traverse_gltf_node(&child, world_xform, bin_data, uri_to_layer, out_tris);
    }
}

fn generate_procedural_sponza_atrium() -> SceneAssets {
    println!(" -> Synthesizing Procedural Sponza Atrium Fallback...");
    let mut tris = Vec::new();
    let texture_layers = vec![make_solid_layer([255, 255, 255, 255])];

    let mut add_box = |bmin: Vec3, bmax: Vec3, albedo: [f32; 3], rough: f32| {
        let corners = [
            Vec3::new(bmin.x, bmin.y, bmin.z),
            Vec3::new(bmax.x, bmin.y, bmin.z),
            Vec3::new(bmax.x, bmax.y, bmin.z),
            Vec3::new(bmin.x, bmax.y, bmin.z),
            Vec3::new(bmin.x, bmin.y, bmax.z),
            Vec3::new(bmax.x, bmin.y, bmax.z),
            Vec3::new(bmax.x, bmax.y, bmax.z),
            Vec3::new(bmin.x, bmax.y, bmax.z),
        ];
        let faces: [([usize; 4], Vec3); 6] = [
            ([0, 1, 2, 3], Vec3::NEG_Z),
            ([5, 4, 7, 6], Vec3::Z),
            ([4, 0, 3, 7], Vec3::NEG_X),
            ([1, 5, 6, 2], Vec3::X),
            ([3, 2, 6, 7], Vec3::Y),
            ([4, 5, 1, 0], Vec3::NEG_Y),
        ];
        for (idx, n) in faces {
            let (p0, p1, p2, p3) = (
                corners[idx[0]],
                corners[idx[1]],
                corners[idx[2]],
                corners[idx[3]],
            );
            for (a, b, c) in [(p0, p1, p2), (p0, p2, p3)] {
                let e1 = b - a;
                let e2 = c - a;
                tris.push(GpuTriangle {
                    v0: [a.x, a.y, a.z, 0.0],
                    e1: [e1.x, e1.y, e1.z, 0.0],
                    e2: [e2.x, e2.y, e2.z, 1.0],
                    n0: [n.x, n.y, n.z, 0.0],
                    n1: [n.x, n.y, n.z, 1.0],
                    n2: [n.x, n.y, n.z, 1.0],
                    albedo_rough: [albedo[0], albedo[1], albedo[2], rough],
                    emissive_mat: [0.0, 0.0, 0.0, 0.0],
                });
            }
        }
    };

    add_box(Vec3::new(-18.0, -0.5, -9.0), Vec3::new(18.0, 0.0, 9.0), [0.65, 0.60, 0.54], 0.25);
    add_box(Vec3::new(-18.0, 0.0, -9.0), Vec3::new(18.0, 13.0, -8.4), [0.68, 0.36, 0.25], 0.75);
    add_box(Vec3::new(-18.0, 0.0, 8.4), Vec3::new(18.0, 13.0, 9.0), [0.68, 0.36, 0.25], 0.75);

    SceneAssets {
        triangles: tris,
        texture_layers,
    }
}

// ============================================================================
// Phase 3: 12-Bin Surface Area Heuristic (SAH) BVH Compiler with Depth Guard
// ============================================================================

struct BvhPrimitive {
    orig_idx: usize,
    aabb: Aabb,
    centroid: Vec3,
}

pub struct BvhBuilder;

impl BvhBuilder {
    pub fn build(triangles: &[GpuTriangle]) -> (Vec<GpuTriangle>, Vec<GpuBvhNode>) {
        println!("========================================================================");
        println!(" [PHASE 3] BUILDING SURFACE AREA HEURISTIC (SAH) GPU BVH");
        println!("========================================================================");
        let start = Instant::now();

        let mut prims: Vec<BvhPrimitive> = triangles
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let v0 = Vec3::new(t.v0[0], t.v0[1], t.v0[2]);
                let v1 = v0 + Vec3::new(t.e1[0], t.e1[1], t.e1[2]);
                let v2 = v0 + Vec3::new(t.e2[0], t.e2[1], t.e2[2]);
                let mut aabb = Aabb::empty();
                aabb.grow(v0);
                aabb.grow(v1);
                aabb.grow(v2);
                aabb.min -= Vec3::splat(1e-4);
                aabb.max += Vec3::splat(1e-4);
                let centroid = (v0 + v1 + v2) * (1.0 / 3.0);
                BvhPrimitive {
                    orig_idx: i,
                    aabb,
                    centroid,
                }
            })
            .collect();

        let mut nodes = Vec::with_capacity(triangles.len() * 2);
        let mut max_depth: u32 = 0;
        Self::build_recursive(&mut prims, 0, 0, &mut max_depth, &mut nodes);

        assert!(
            max_depth <= GPU_MAX_STACK_DEPTH,
            "BVH max depth ({}) exceeded GPU traversal stack capacity ({})",
            max_depth,
            GPU_MAX_STACK_DEPTH
        );

        let reordered_tris: Vec<GpuTriangle> =
            prims.iter().map(|p| triangles[p.orig_idx]).collect();

        println!(
            " -> SAH BVH Compiled : {} nodes (Max Depth: {}/{}) for {} triangles in {:.2?}",
            nodes.len(),
            max_depth,
            GPU_MAX_STACK_DEPTH,
            reordered_tris.len(),
            start.elapsed()
        );
        (reordered_tris, nodes)
    }

    fn build_recursive(
        prims: &mut [BvhPrimitive],
        first_global: usize,
        depth: u32,
        max_depth: &mut u32,
        nodes: &mut Vec<GpuBvhNode>,
    ) -> u32 {
        *max_depth = (*max_depth).max(depth);

        let node_idx = nodes.len() as u32;
        nodes.push(GpuBvhNode {
            aabb_min: [0.0; 3],
            left_or_first: 0,
            aabb_max: [0.0; 3],
            tri_count: 0,
        });

        let mut bounds = Aabb::empty();
        let mut centroid_bounds = Aabb::empty();
        for p in prims.iter() {
            bounds = bounds.union(&p.aabb);
            centroid_bounds.grow(p.centroid);
        }

        let count = prims.len();
        // Force a leaf node if count <= 4 OR if depth reaches MAX_BVH_DEPTH (guarantees stack never overflows)
        if count <= 4 || depth >= MAX_BVH_DEPTH {
            nodes[node_idx as usize] = GpuBvhNode {
                aabb_min: bounds.min.to_array(),
                left_or_first: first_global as u32,
                aabb_max: bounds.max.to_array(),
                tri_count: count as u32,
            };
            return node_idx;
        }

        const BINS: usize = 12;
        let mut best_cost = count as f32;
        let mut best_axis = 0;
        let mut best_split_pos = 0.0_f32;
        let parent_area = bounds.area().max(1e-6);

        for axis in 0..3 {
            let cmin = centroid_bounds.min[axis];
            let cmax = centroid_bounds.max[axis];
            if (cmax - cmin) < 1e-5 {
                continue;
            }

            let mut bin_boxes = [Aabb::empty(); BINS];
            let mut bin_counts = [0usize; BINS];
            let scale = BINS as f32 / (cmax - cmin);

            for p in prims.iter() {
                let b = (((p.centroid[axis] - cmin) * scale) as usize).min(BINS - 1);
                bin_counts[b] += 1;
                bin_boxes[b] = bin_boxes[b].union(&p.aabb);
            }

            let mut left_area = [0.0_f32; BINS - 1];
            let mut left_count = [0usize; BINS - 1];
            let mut right_area = [0.0_f32; BINS - 1];
            let mut right_count = [0usize; BINS - 1];

            let mut cur_left_box = Aabb::empty();
            let mut cur_left_cnt = 0;
            for i in 0..(BINS - 1) {
                cur_left_cnt += bin_counts[i];
                cur_left_box = cur_left_box.union(&bin_boxes[i]);
                left_count[i] = cur_left_cnt;
                left_area[i] = cur_left_box.area();
            }

            let mut cur_right_box = Aabb::empty();
            let mut cur_right_cnt = 0;
            for i in (1..BINS).rev() {
                cur_right_cnt += bin_counts[i];
                cur_right_box = cur_right_box.union(&bin_boxes[i]);
                right_count[i - 1] = cur_right_cnt;
                right_area[i - 1] = cur_right_box.area();
            }

            let step = (cmax - cmin) / BINS as f32;
            for i in 0..(BINS - 1) {
                if left_count[i] == 0 || right_count[i] == 0 {
                    continue;
                }
                let cost = 0.5
                    + (left_count[i] as f32 * left_area[i]
                        + right_count[i] as f32 * right_area[i])
                        / parent_area;
                if cost < best_cost {
                    best_cost = cost;
                    best_axis = axis;
                    best_split_pos = cmin + step * (i as f32 + 1.0);
                }
            }
        }

        let mid = if best_cost < count as f32 {
            let mut i = 0;
            let mut j = count;
            while i < j {
                if prims[i].centroid[best_axis] < best_split_pos {
                    i += 1;
                } else {
                    j -= 1;
                    prims.swap(i, j);
                }
            }
            if i == 0 || i == count {
                count / 2
            } else {
                i
            }
        } else {
            let extent = centroid_bounds.max - centroid_bounds.min;
            let axis = if extent.x >= extent.y && extent.x >= extent.z {
                0
            } else if extent.y >= extent.z {
                1
            } else {
                2
            };
            prims.sort_unstable_by(|a, b| {
                a.centroid[axis]
                    .partial_cmp(&b.centroid[axis])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            count / 2
        };

        let (left_slice, right_slice) = prims.split_at_mut(mid);
        let _left_child =
            Self::build_recursive(left_slice, first_global, depth + 1, max_depth, nodes);
        let right_child =
            Self::build_recursive(right_slice, first_global + mid, depth + 1, max_depth, nodes);

        nodes[node_idx as usize] = GpuBvhNode {
            aabb_min: bounds.min.to_array(),
            left_or_first: right_child,
            aabb_max: bounds.max.to_array(),
            tri_count: 0,
        };

        node_idx
    }
}

// ============================================================================
// Phase 4: Free-Fly 6-DOF Viewport Camera (WASD + Mouse Look)
// ============================================================================

struct CameraController {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    fov_y_rad: f32,
    move_speed: f32,
    mouse_sensitivity: f32,
    pressed_keys: HashSet<KeyCode>,
    cursor_locked: bool,
    sun_azimuth: f32,
}

impl CameraController {
    fn new() -> Self {
        Self {
            position: Vec3::new(-7.5, 2.4, 0.0),
            yaw: 0.0,
            pitch: 0.05,
            fov_y_rad: 60.0_f32.to_radians(),
            move_speed: 6.0,
            mouse_sensitivity: 0.0022,
            pressed_keys: HashSet::new(),
            cursor_locked: false,
            sun_azimuth: 0.65,
        }
    }

    fn reset(&mut self) {
        self.position = Vec3::new(-7.5, 2.4, 0.0);
        self.yaw = 0.0;
        self.pitch = 0.05;
    }

    fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let forward = Vec3::new(cy * cp, sp, sy * cp).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();
        (forward, right, up)
    }

    fn handle_mouse_delta(&mut self, dx: f64, dy: f64) -> bool {
        if !self.cursor_locked {
            return false;
        }
        if dx.abs() < 1e-5 && dy.abs() < 1e-5 {
            return false;
        }
        self.yaw += (dx as f32) * self.mouse_sensitivity;
        self.pitch = (self.pitch - (dy as f32) * self.mouse_sensitivity).clamp(-1.54, 1.54);
        true
    }

    fn update(&mut self, dt: f32) -> bool {
        let (forward, right, _) = self.basis();
        let flat_fwd = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
        let mut vel = Vec3::ZERO;

        if self.pressed_keys.contains(&KeyCode::KeyW) {
            vel += flat_fwd;
        }
        if self.pressed_keys.contains(&KeyCode::KeyS) {
            vel -= flat_fwd;
        }
        if self.pressed_keys.contains(&KeyCode::KeyD) {
            vel += right;
        }
        if self.pressed_keys.contains(&KeyCode::KeyA) {
            vel -= right;
        }
        if self.pressed_keys.contains(&KeyCode::Space) {
            vel += Vec3::Y;
        }
        if self.pressed_keys.contains(&KeyCode::ShiftLeft) {
            vel -= Vec3::Y;
        }

        let mut changed = false;
        if self.pressed_keys.contains(&KeyCode::KeyQ) {
            self.sun_azimuth -= 0.8 * dt;
            changed = true;
        }
        if self.pressed_keys.contains(&KeyCode::KeyE) {
            self.sun_azimuth += 0.8 * dt;
            changed = true;
        }

        if vel.length_squared() > 1e-5 {
            let boost = if self.pressed_keys.contains(&KeyCode::ControlLeft) {
                2.75
            } else {
                1.0
            };
            self.position += vel.normalize() * self.move_speed * boost * dt;
            changed = true;
        }

        changed
    }

    fn to_uniform(&self, width: u32, height: u32, frame_idx: u32, bvh_nodes: u32) -> CameraUniform {
        let (fwd, right, up) = self.basis();
        let aspect = (width.max(1) as f32) / (height.max(1) as f32);
        let tan_half_fov = (self.fov_y_rad * 0.5).tan();

        CameraUniform {
            cam_pos_fov: [self.position.x, self.position.y, self.position.z, tan_half_fov],
            cam_fwd_aspect: [fwd.x, fwd.y, fwd.z, aspect],
            cam_right_sun: [right.x, right.y, right.z, self.sun_azimuth],
            cam_up_rough: [up.x, up.y, up.z, 1.0],
            frame_info: [width, height, frame_idx, bvh_nodes],
        }
    }
}

// ============================================================================
// Phase 5: DX12 / Vulkan Hardware Scanner & Ray Tracing GPU Engine
// ============================================================================

struct RayTracerEngine {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,

    camera_buffer: wgpu::Buffer,
    tri_buffer: wgpu::Buffer,
    bvh_buffer: wgpu::Buffer,
    accum_buffer: wgpu::Buffer,
    output_texture_view: wgpu::TextureView,
    mat_texture_array_view: wgpu::TextureView,

    compute_bgl: wgpu::BindGroupLayout,
    compute_bg: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,

    render_bgl: wgpu::BindGroupLayout,
    render_bg: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,

    frame_index: u32,
    bvh_node_count: u32,
    active_backend_name: String,
    active_gpu_name: String,
}

impl RayTracerEngine {
    async fn new(
        window: Arc<Window>,
        pref: BackendPreference,
        triangles: &[GpuTriangle],
        bvh_nodes: &[GpuBvhNode],
        texture_layers: &[Vec<u8>],
    ) -> Result<Self> {
        println!("========================================================================");
        println!(" [PHASE 4] GPU ADAPTER SCAN & DX12 / VULKAN BACKEND SELECTION");
        println!("========================================================================");

        let size = window.inner_size();
        let width = size.width.max(64);
        let height = size.height.max(64);

        let backends = wgpu::Backends::DX12 | wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::empty(),
            dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        let surface = instance.create_surface(window.clone())?;
        let adapters = instance.enumerate_adapters(backends);

        if adapters.is_empty() {
            anyhow::bail!("No compatible DirectX 12 or Vulkan GPU adapter detected!");
        }

        println!(" -> Detected {} DX12/Vulkan Hardware Adapter(s):", adapters.len());
        let mut best_adapter: Option<(wgpu::Adapter, i32)> = None;

        for (idx, adapter) in adapters.into_iter().enumerate() {
            let info = adapter.get_info();
            let limits = adapter.limits();
            let surface_ok = adapter.is_surface_supported(&surface);

            let mut score = 0i32;
            if !surface_ok {
                score -= 10_000;
            }
            score += match info.device_type {
                wgpu::DeviceType::DiscreteGpu => 1000,
                wgpu::DeviceType::IntegratedGpu => 400,
                wgpu::DeviceType::VirtualGpu => 200,
                wgpu::DeviceType::Cpu => 50,
                wgpu::DeviceType::Other => 100,
            };

            match (pref, info.backend) {
                (BackendPreference::Dx12, wgpu::Backend::Dx12) => score += 2000,
                (BackendPreference::Vulkan, wgpu::Backend::Vulkan) => score += 2000,
                (BackendPreference::Auto, wgpu::Backend::Dx12) if cfg!(target_os = "windows") => {
                    score += 300
                }
                (BackendPreference::Auto, wgpu::Backend::Vulkan) => score += 250,
                _ => {}
            }

            println!(
                "    [{}] {} | Backend: {:?} | Type: {:?} | Vendor: 0x{:04X} | MaxStorageBuf: {} MB | Score: {}",
                idx,
                info.name,
                info.backend,
                info.device_type,
                info.vendor,
                limits.max_storage_buffer_binding_size / (1024 * 1024),
                score
            );

            if best_adapter.as_ref().map_or(true, |(_, s)| score > *s) {
                best_adapter = Some((adapter, score));
            }
        }

        let (adapter, _) = best_adapter.context("Failed to select a DX12/Vulkan GPU adapter")?;
        let info = adapter.get_info();
        let active_backend_name = format!("{:?}", info.backend);
        let active_gpu_name = info.name.clone();

        println!(
            " -> LOCKED ADAPTER   : {} via {} ({:?})",
            active_gpu_name, active_backend_name, info.device_type
        );

        let mut required_limits = wgpu::Limits::default();
        let adapter_limits = adapter.limits();
        required_limits.max_storage_buffer_binding_size =
            adapter_limits.max_storage_buffer_binding_size;
        required_limits.max_buffer_size = adapter_limits.max_buffer_size;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Sponza RayTracer Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else if caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::Fifo
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Upload Sponza baseColor textures into a GPU Texture2DArray (Rgba8UnormSrgb)
        let num_layers = (texture_layers.len() as u32).max(1);
        let mat_tex_array = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Sponza Material Texture2DArray"),
            size: wgpu::Extent3d {
                width: TEX_DIM,
                height: TEX_DIM,
                depth_or_array_layers: num_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        for (layer_idx, raw_rgba) in texture_layers.iter().enumerate() {
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &mat_tex_array,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer_idx as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                raw_rgba,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * TEX_DIM),
                    rows_per_image: Some(TEX_DIM),
                },
                wgpu::Extent3d {
                    width: TEX_DIM,
                    height: TEX_DIM,
                    depth_or_array_layers: 1,
                },
            );
        }

        let mat_texture_array_view = mat_tex_array.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Sponza Material Texture2DArray View"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sponza RayTracing Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let dummy_cam = CameraUniform::zeroed();
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::bytes_of(&dummy_cam),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let tri_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scene Triangle SSBO"),
            contents: bytemuck::cast_slice(triangles),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bvh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scene SAH BVH SSBO"),
            contents: bytemuck::cast_slice(bvh_nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let (accum_buffer, output_texture_view) =
            Self::create_screen_resources(&device, width, height);

        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
            ],
        });

        let compute_bg = Self::create_compute_bg(
            &device,
            &compute_bgl,
            &camera_buffer,
            &tri_buffer,
            &bvh_buffer,
            &accum_buffer,
            &output_texture_view,
            &mat_texture_array_view,
        );

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bgl],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BVH RayTracing Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "cs_main",
            compilation_options: Default::default(),
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Display Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let render_bg =
            Self::create_render_bg(&device, &render_bgl, &output_texture_view, &sampler);

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&render_bgl],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size: PhysicalSize::new(width, height),
            camera_buffer,
            tri_buffer,
            bvh_buffer,
            accum_buffer,
            output_texture_view,
            mat_texture_array_view,
            compute_bgl,
            compute_bg,
            compute_pipeline,
            render_bgl,
            render_bg,
            render_pipeline,
            sampler,
            frame_index: 0,
            bvh_node_count: bvh_nodes.len() as u32,
            active_backend_name,
            active_gpu_name,
        })
    }

    fn create_screen_resources(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Buffer, wgpu::TextureView) {
        let accum_size = (width as u64) * (height as u64) * 16;
        let accum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HDR Progressive Accumulation Buffer"),
            size: accum_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RayTracer Output Storage Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (accum_buffer, view)
    }

    #[allow(clippy::too_many_arguments)]
    fn create_compute_bg(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        camera_buffer: &wgpu::Buffer,
        tri_buffer: &wgpu::Buffer,
        bvh_buffer: &wgpu::Buffer,
        accum_buffer: &wgpu::Buffer,
        output_view: &wgpu::TextureView,
        mat_array_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tri_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bvh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: accum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(mat_array_view),
                },
            ],
        })
    }

    fn create_render_bg(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        output_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);

        let (accum_buffer, output_texture_view) =
            Self::create_screen_resources(&self.device, new_size.width, new_size.height);
        self.accum_buffer = accum_buffer;
        self.output_texture_view = output_texture_view;

        self.compute_bg = Self::create_compute_bg(
            &self.device,
            &self.compute_bgl,
            &self.camera_buffer,
            &self.tri_buffer,
            &self.bvh_buffer,
            &self.accum_buffer,
            &self.output_texture_view,
            &self.mat_texture_array_view,
        );
        self.render_bg = Self::create_render_bg(
            &self.device,
            &self.render_bgl,
            &self.output_texture_view,
            &self.sampler,
        );
        self.frame_index = 0;
    }

    fn render(&mut self, cam: &CameraController) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let surface_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let cam_uniform = cam.to_uniform(
            self.size.width,
            self.size.height,
            self.frame_index,
            self.bvh_node_count,
        );
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&cam_uniform));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RayTracer Frame Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BVH RayTracing Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bg, &[]);
            let wg_x = (self.size.width + 7) / 8;
            let wg_y = (self.size.height + 7) / 8;
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fullscreen ToneMap & Bilateral Blit Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.render_bg, &[]);
            rpass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        self.frame_index = self.frame_index.saturating_add(1);
        Ok(())
    }
}

fn set_cursor_grab(window: &Window, locked: bool) {
    if locked {
        if window.set_cursor_grab(CursorGrabMode::Locked).is_err() {
            let _ = window.set_cursor_grab(CursorGrabMode::Confined);
        }
        window.set_cursor_visible(false);
    } else {
        let _ = window.set_cursor_grab(CursorGrabMode::None);
        window.set_cursor_visible(true);
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let backend_pref = print_startup_banner();

    // Load Sponza glTF + textures; if cached files fail integrity checks, purge & re-fetch once
    let scene_assets = match ensure_sponza_geometry_files(false)
        .and_then(|p| load_sponza_gltf_with_textures(&p).ok())
        .or_else(|| {
            println!(" [!] Cached Sponza asset validation failed; attempting clean re-download...");
            ensure_sponza_geometry_files(true)
                .and_then(|p| load_sponza_gltf_with_textures(&p).ok())
        }) {
        Some(assets) => assets,
        None => generate_procedural_sponza_atrium(),
    };

    let (ordered_tris, bvh_nodes) = BvhBuilder::build(&scene_assets.triangles);

    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Sponza Atrium Real-Time GPU Ray Tracer [Initializing...]")
            .with_inner_size(PhysicalSize::new(1280, 720))
            .build(&event_loop)?,
    );

    let mut engine = pollster::block_on(RayTracerEngine::new(
        window.clone(),
        backend_pref,
        &ordered_tris,
        &bvh_nodes,
        &scene_assets.texture_layers,
    ))?;

    println!("========================================================================");
    println!(" [PHASE 5] REAL-TIME VIEWPORT CONTROLS ACTIVE");
    println!(" -> Left-Click Viewport : Lock mouse for 360° Free-Look");
    println!(" -> Escape              : Release mouse cursor");
    println!(" -> W / A / S / D       : Move through Sponza Atrium");
    println!(" -> Space / L-Shift     : Ascend / Descend gallery levels");
    println!(" -> L-Ctrl              : Sprint (2.75x speed)");
    println!(" -> Q / E               : Rotate Sun Direction in real time");
    println!(" -> R                   : Reset Camera to Main Atrium Corridor");
    println!("========================================================================");

    let mut camera = CameraController::new();
    let mut last_tick = Instant::now();
    let mut fps_timer = Instant::now();
    let mut frames_counted = 0u32;

    let window_loop = window.clone();
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta: (dx, dy) },
                ..
            } => {
                if camera.handle_mouse_delta(dx, dy) {
                    engine.frame_index = 0;
                }
            }
            Event::WindowEvent { event, window_id } if window_id == window_loop.id() => {
                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(physical_size) => {
                        engine.resize(physical_size);
                    }
                    WindowEvent::MouseInput {
                        state: ElementState::Pressed,
                        button: MouseButton::Left,
                        ..
                    } => {
                        if !camera.cursor_locked {
                            camera.cursor_locked = true;
                            set_cursor_grab(&window_loop, true);
                        }
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if let PhysicalKey::Code(code) = event.physical_key {
                            match event.state {
                                ElementState::Pressed => {
                                    if code == KeyCode::Escape {
                                        if camera.cursor_locked {
                                            camera.cursor_locked = false;
                                            set_cursor_grab(&window_loop, false);
                                        } else {
                                            elwt.exit();
                                        }
                                    } else if code == KeyCode::KeyR {
                                        camera.reset();
                                        engine.frame_index = 0;
                                    } else {
                                        camera.pressed_keys.insert(code);
                                    }
                                }
                                ElementState::Released => {
                                    camera.pressed_keys.remove(&code);
                                }
                            }
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        let now = Instant::now();
                        let dt = (now - last_tick).as_secs_f32().min(0.1);
                        last_tick = now;

                        if camera.update(dt) {
                            engine.frame_index = 0;
                        }

                        match engine.render(&camera) {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                engine.resize(engine.size);
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                            Err(wgpu::SurfaceError::Timeout) => {}
                        }

                        frames_counted += 1;
                        let elapsed = fps_timer.elapsed().as_secs_f32();
                        if elapsed >= 0.4 {
                            let fps = frames_counted as f32 / elapsed;
                            window_loop.set_title(&format!(
                                "Sponza RT | {} [{}] | {:.1} FPS | SPP: {} | Tris: {} | Pos: ({:.1}, {:.1}, {:.1})",
                                engine.active_gpu_name,
                                engine.active_backend_name,
                                fps,
                                engine.frame_index,
                                ordered_tris.len(),
                                camera.position.x,
                                camera.position.y,
                                camera.position.z,
                            ));
                            frames_counted = 0;
                            fps_timer = Instant::now();
                        }
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                window_loop.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}