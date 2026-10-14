# Sponza-RT: Real-Time GPU Compute Ray Tracer in Rust

![Sponza Real-Time GPU Ray Tracer Viewport]<img width="1278" height="753" alt="SponzaRT" src="https://github.com/user-attachments/assets/1ccf2e1e-9a74-462a-88dc-00facc275600" />


> **Real-time hardware-agnostic GPU compute ray tracer for the 262,266-triangle Khronos Sponza Atrium in pure Rust (`wgpu 22.1` + WGSL), running at ~43.3 FPS on an NVIDIA GeForce RTX 3050 Laptop GPU (DirectX 12 / Vulkan). Features a depth-bounded 12-bin Surface Area Heuristic (SAH) BVH, atomic parallel texture provisioning, ray-traced alpha-cutout foliage, ray-traced sun shadows, short-range RTAO with approximate single-bounce albedo bleed, and an edge-stopping cross-bilateral denoiser.**

---

## Table of Contents
1. [Project Overview & Key Features](#1-project-overview--key-features)
2. [System Architecture & 5-Phase Execution Pipeline](#2-system-architecture--5-phase-execution-pipeline)
3. [Deep-Dive Systems & Graphics Engineering Breakdown](#3-deep-dive-systems--graphics-engineering-breakdown)
   - [3.1 GPU Memory Alignment & Cache-Line Packing (`#[repr(C)]`)](#31-gpu-memory-alignment--cache-line-packing-reprc)
   - [3.2 Depth-Bounded 12-Bin Surface Area Heuristic (SAH) BVH Compiler](#32-depth-bounded-12-bin-surface-area-heuristic-sah-bvh-compiler)
   - [3.3 Pointerless DFS BVH Layout & Ordered Short-Stack Traversal](#33-pointerless-dfs-bvh-layout--ordered-short-stack-traversal)
   - [3.4 Möller–Trumbore Ray-Triangle Intersection](#34-möllertrumbore-ray-triangle-intersection)
   - [3.5 Base-Color `Texture2DArray` & Ray-Traced Alpha-Cutout Masks](#35-base-color-texture2darray--ray-traced-alpha-cutout-masks)
   - [3.6 Hybrid Lighting: Ray-Traced Shadows, Short-Range RTAO & Approximate Bounce Light](#36-hybrid-lighting-ray-traced-shadows-short-range-rtao--approximate-bounce-light)
   - [3.7 Edge-Stopping Cross-Bilateral Spatial Denoiser](#37-edge-stopping-cross-bilateral-spatial-denoiser)
4. [Engineering Post-Mortem: Visual & Systems Debugging Progression](#4-engineering-post-mortem-visual--systems-debugging-progression)
5. [Build, CLI Flags & Interactive Viewport Controls](#5-build-cli-flags--interactive-viewport-controls)
6. [Hardware Telemetry & Benchmark Reference](#6-hardware-telemetry--benchmark-reference)

---

## 1. Project Overview & Key Features

`sponza-rt` is a self-contained real-time GPU ray tracer written in **Rust** and **WGSL**. Rather than relying on vendor-specific hardware ray-tracing extensions (`DXR` or `VK_KHR_ray_tracing_pipeline`), it implements the acceleration structure compiler and BVH ray-traversal kernel in portable compute shaders (`@compute @workgroup_size(8, 8, 1)`), executing natively on **DirectX 12** and **Vulkan**.

### Core Capabilities
* **Host Environment Banner & Hardware Adapter Scanner (`print_startup_banner`)**: Reports compile-time package metadata, logical CPU parallelism, and pinned asset revision, then enumerates and scores all physical DirectX 12 and Vulkan GPU adapters by device type (`DiscreteGpu`, `IntegratedGpu`, `Cpu`), VRAM storage binding limits, and OS affinity.
* **Atomic Sponza Asset & Parallel Texture Auto-Installer (`download_file_atomic`)**: Streams `Sponza.gltf`, the `9.09 MB` binary geometry buffer (`Sponza.bin`), and all **24 base-color textures (`.jpg` / `.png`)** from a pinned Khronos `glTF-Sample-Assets` commit (`723ffc6`) across 8 worker threads. All HTTP reads are size-capped (`take(MAX_ASSET_BYTES + 1)`), verified against `Content-Length` and glTF buffer declarations, written to isolated `.part` temporary files with `sync_all()`, and atomically renamed so interrupted downloads never corrupt the cache.
* **Depth-Bounded 12-Bin SAH BVH Compiler (`BvhBuilder`)**: Compiles **262,266 triangles** into a **163,553-node** pointerless Bounding Volume Hierarchy in **`212.30 ms`** on the CPU. Enforces `MAX_BVH_DEPTH = 31` and asserts `max_depth <= 32` so the 32-entry GPU traversal stack is mathematically guaranteed never to overflow or drop a subtree.
* **Hardware sRGB `Texture2DArray` & Alpha-Cutout BVH Traversal (`passes_alpha_cutout`)**: Decodes and resizes all Sponza base-color textures in parallel across CPU cores into a 25-layer `512x512` `Rgba8UnormSrgb` GPU texture array. Evaluates `alphaMode: MASK` inside both closest-hit (`trace_bvh`) and shadow-occlusion (`trace_shadow`) BVH loops so planter leaves, flowers, thorns, and hanging chains cast exact cutout shadows.
* **Hybrid Ray-Traced Lighting & Bilateral Denoising**: Combines ray-traced direct sun shadows, Blinn-Phong specular modulated by glTF scalar roughness/metallic factors, short-range Ray-Traced Ambient Occlusion (RTAO) stratified with Interleaved Gradient Noise (IGN), approximate single-bounce albedo bleed, neighborhood-clamped temporal accumulation, and a $3 \times 3$ depth/luminance edge-stopping cross-bilateral filter.

---

## 2. System Architecture & 5-Phase Execution Pipeline

```text
+-----------------------------------------------------------------------------+
| [PHASE 1] Host Environment & CLI Configuration Banner                       |
|   -> Reports package version, OS/Arch, logical CPU cores, CLI flags         |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| [PHASE 2] Atomic Sponza glTF & Parallel Base-Color Texture Provisioner      |
|   -> Pinned to KhronosGroup/glTF-Sample-Assets @ 723ffc6                    |
|   -> Bounded HTTP reads + Content-Length check + .part atomic file rename   |
|   -> Parallel CPU image decoding into 512x512 RGBA8 sRGB Texture2DArray     |
|   -> Flattens glTF scene-graph hierarchy with 4x4 world & normal matrices   |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| [PHASE 3] Depth-Bounded 12-Bin Surface Area Heuristic (SAH) BVH Compiler    |
|   -> Evaluates 11 split planes across X, Y, Z axes per internal node        |
|   -> Compiles 262,266 triangles into 163,553 DFS nodes in ~212.30 ms        |
|   -> Enforces and asserts max_depth <= 32 to match GPU traversal stack      |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| [PHASE 4] DX12 / Vulkan Adapter Scoring & Compute/Render Pipeline Init      |
|   -> Scores all physical GPUs; locks onto Discrete GPU (DX12 or Vulkan)     |
|   -> Allocates SSBOs (Triangles: 32.0 MB, BVH: 5.23 MB, Accum: 14.7 MB)     |
|   -> Uploads 25-layer Rgba8UnormSrgb Texture2DArray (25.0 MB) to VRAM       |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| [PHASE 5] Real-Time 6-DOF Render Loop (Compute Pass + Bilateral Blit Pass)  |
|   -> Pass 1 (@compute cs_main): Primary ray + shadow ray + RTAO/bleed ray   |
|   -> Pass 2 (@fragment fs_main): Cross-bilateral spatial denoiser & blit    |
+-----------------------------------------------------------------------------+
```

---

## 3. Deep-Dive Systems & Graphics Engineering Breakdown

### 3.1 GPU Memory Alignment & Cache-Line Packing (`#[repr(C)]`)

Modern GPU backends (DirectX 12 DXIL, Vulkan SPIR-V, and WGSL) enforce strict `std140` / `std430` memory alignment rules. A `vec3<f32>` in WGSL has a **16-byte alignment** despite occupying 12 bytes, which causes stride mismatches if unpadded.

To guarantee zero padding ambiguity and optimal L1/L2 GPU cache-line utilization:

1. **`GpuTriangle` (128 bytes — two 64-byte GPU cache lines):**
   Instead of leaving the 4th float (`.w`) of each `vec4<f32>` unused, the 6 vertex UV coordinates (`uv0.xy`, `uv1.xy`, `uv2.xy`) are packed directly into the `.w` channels of `v0`, `e1`, `e2`, `n0`, `n1`, and `n2`:
   ```rust
   #[repr(C)]
   #[derive(Copy, Clone, Debug, Pod, Zeroable)]
   pub struct GpuTriangle {
       pub v0: [f32; 4],           // xyz: vertex 0,          w: uv0.x
       pub e1: [f32; 4],           // xyz: edge 1 (v1 - v0),  w: uv0.y
       pub e2: [f32; 4],           // xyz: edge 2 (v2 - v0),  w: uv1.x
       pub n0: [f32; 4],           // xyz: vertex normal 0,   w: uv1.y
       pub n1: [f32; 4],           // xyz: vertex normal 1,   w: uv2.x
       pub n2: [f32; 4],           // xyz: vertex normal 2,   w: uv2.y
       pub albedo_rough: [f32; 4], // xyz: base_color_factor, w: roughness_factor
       pub emissive_mat: [f32; 4], // x: tex_layer_idx, y: alpha_cutoff, z: metallic_factor, w: pad
   }
   ```
   Precomputing triangle edges $\mathbf{e}_1 = \mathbf{v}_1 - \mathbf{v}_0$ and $\mathbf{e}_2 = \mathbf{v}_2 - \mathbf{v}_0$ on the CPU saves two vector subtractions on every ray-triangle intersection test.

2. **`GpuBvhNode` (32 bytes — two nodes per 64-byte GPU cache line):**
   ```rust
   #[repr(C)]
   #[derive(Copy, Clone, Debug, Pod, Zeroable)]
   pub struct GpuBvhNode {
       pub aabb_min: [f32; 3],
       pub left_or_first: u32, // Internal: right child idx; Leaf: first triangle idx
       pub aabb_max: [f32; 3],
       pub tri_count: u32,     // 0 = Internal node; >0 = Leaf triangle count
   }
   ```

---

### 3.2 Depth-Bounded 12-Bin Surface Area Heuristic (SAH) BVH Compiler

We minimize expected ray traversal cost using the **Surface Area Heuristic (SAH)**. By geometric probability, the conditional probability of a random ray intersecting a convex child volume $B_i$ given that it intersects its parent volume $B_P$ is the ratio of their surface areas:

$$P(\text{hit } B_i \mid \text{hit } B_P) = \frac{SA(B_i)}{SA(B_P)}, \qquad SA(B) = 2(d_x d_y + d_y d_z + d_z d_x)$$

For any candidate split partitioning parent node $B_P$ into Left ($L$) and Right ($R$) child subsets containing $N_L$ and $N_R$ triangles, the SAH cost model (with traversal cost $C_{\text{trav}} = 0.5$ and triangle intersection cost $C_{\text{isect}} = 1.0$) is:

$$C_{\text{SAH}} = 0.5 + \frac{SA(B_L)}{SA(B_P)} N_L + \frac{SA(B_R)}{SA(B_P)} N_R$$

`BvhBuilder::build_recursive` evaluates **12-Bin Binned SAH** ($O(N)$ per tree level):
1. Along each axis $a \in \{x, y, z\}$, the centroid interval $[c_{\min}, c_{\max}]$ is divided into $B = 12$ uniform bins.
2. Each triangle is mapped to bin $b = \min\left(\lfloor B \cdot \frac{c_a - c_{\min}}{c_{\max} - c_{\min}} \rfloor, B - 1\right)$, accumulating bin AABBs and counts in a single pass.
3. Prefix and suffix sweeps evaluate all $B - 1 = 11$ split planes in $O(B)$ time, and primitives are partitioned in-place via a two-pointer swap whose predicate (`centroid < best_split_pos`) matches the bin boundaries.
4. **Hard Stack-Depth Invariant**: Because the WGSL traversal kernel uses a fixed 32-entry register/local stack (`var stack: array<u32, 32>`), `build_recursive` tracks `depth` and forces leaf creation if `depth >= MAX_BVH_DEPTH` (`31`), followed by a host-side `assert!(max_depth <= 32)`.

---

### 3.3 Pointerless DFS BVH Layout & Ordered Short-Stack Traversal

Because `BvhBuilder::build_recursive` serializes nodes in **Pre-Order Depth-First Search (DFS)** order:
* For any internal node at `node_idx` (`tri_count == 0`), its **left child is guaranteed to reside at `node_idx + 1`**.
* Only the **right child index** needs to be stored in `left_or_first`.

During GPU ray traversal (`trace_bvh`), each thread tests both children simultaneously:
1. Compute ray-slab entry distances `dist_l` and `dist_r` for both children against the current closest hit `result.t`.
2. If both children are intersected, **traverse the nearer child immediately** (`node_idx = near_idx`) and push the farther child onto the stack.
3. Visiting the nearer child first shrinks `result.t` early, allowing subsequent `intersect_aabb` slab tests to cull distant subtrees completely.

---

### 3.4 Möller–Trumbore Ray-Triangle Intersection

For ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ and triangle edges $\mathbf{e}_1 = \mathbf{v}_1 - \mathbf{v}_0$, $\mathbf{e}_2 = \mathbf{v}_2 - \mathbf{v}_0$, solving for distance $t$ and barycentric coordinates $(u, v)$ via Cramer's Rule yields:

$$\begin{bmatrix} t \\ u \\ v \end{bmatrix} = \frac{1}{\mathbf{e}_1 \cdot (\mathbf{d} \times \mathbf{e}_2)} \begin{bmatrix} \mathbf{e}_2 \cdot ((\mathbf{o} - \mathbf{v}_0) \times \mathbf{e}_1) \\ (\mathbf{o} - \mathbf{v}_0) \cdot (\mathbf{d} \times \mathbf{e}_2) \\ \mathbf{d} \cdot ((\mathbf{o} - \mathbf{v}_0) \times \mathbf{e}_1) \end{bmatrix}$$

Once a hit $(t, u, v)$ is confirmed, vertex normals and UV coordinates are interpolated via barycentric weights $w = 1 - u - v$:

$$\mathbf{n} = \text{normalize}(w\mathbf{n}_0 + u\mathbf{n}_1 + v\mathbf{n}_2), \qquad \mathbf{uv} = w\mathbf{uv}_0 + u\mathbf{uv}_1 + v\mathbf{uv}_2$$

---

### 3.5 Base-Color `Texture2DArray` & Ray-Traced Alpha-Cutout Masks

In compute shaders with divergent secondary rays, screen-space $2 \times 2$ quad derivatives do not exist, and `textureSample` cannot be called inside non-uniform BVH loops.

To solve this cleanly across DirectX 12 and Vulkan:
1. All 24 Sponza base-color textures are decoded in parallel and uploaded as a 25-layer `512x512` `Rgba8UnormSrgb` `texture_2d_array<f32>` (Layer 0 is a solid fallback layer; Layers 1..24 hold the Sponza base-color textures). Using an `Srgb` format instructs the GPU's texture hardware to perform **sRGB-to-linear conversion** automatically on every read.
2. `sample_material_tex` implements an explicit 4-tap bilinear filter using `textureLoad` with bitwise wrapping (`& 511`), which is valid inside any divergent control flow.
3. For cutout materials (`alphaMode: MASK` — planter flowers, ivy leaves, thorns, and hanging chains), `passes_alpha_cutout` evaluates the texture alpha at the exact $(u, v)$ hit point inside `trace_bvh` and `trace_shadow`. If $\alpha < \text{cutoff}$, the hit is rejected and the ray continues through the transparent region of the quad.

---

### 3.6 Hybrid Lighting: Ray-Traced Shadows, Short-Range RTAO & Approximate Bounce Light

Rather than tracing high-variance multi-bounce paths (which cause severe Monte Carlo fireflies at 1 SPP inside Sponza's enclosed courtyard), `cs_main` uses a fast hybrid ray-tracing model casting **up to 3 rays per pixel**:

1. **Primary Visibility Ray (`trace_bvh`)**: Finds the closest surface intersection and evaluates base-color texture albedo, scalar `roughnessFactor`, and scalar `metallicFactor`.
2. **Direct Sun Shadow Ray (`trace_shadow`) + Blinn-Phong Specular**: Casts a shadow ray toward the directional sun (`sun_dir`), with soft-shadow disk jittering enabled once `frame_idx > 2` using a 24-bit uniform float generator (`f32(*seed >> 8u) * (1.0 / 16777216.0)` in $[0, 1)$). Direct lighting combines Lambertian diffuse with a Blinn-Phong specular lobe modulated by the material's scalar roughness and metallic factors.
3. **Short-Range RTAO + Approximate Single-Bounce Albedo Bleed**: A cosine-weighted hemisphere ray (`max_dist = 8.5m`) stratified with **Interleaved Gradient Noise (IGN)** queries local ambient occlusion (`ao_factor = clamp(ao_hit.t / 4.2, 0.20, 1.0)`). When an occluder is hit within `8.5m`, its textured base-color albedo is sampled (`b_albedo * 0.42`) and combined with analytical hemisphere sky/floor/corridor fill terms.
4. **TAA-Style Motion Clamping**: On `frame_idx == 0` (during camera movement), previous frame radiance is clamped to $[0.55 L_{\text{cur}}, 1.45 L_{\text{cur}} + 0.08]$ and blended at 22% weight to suppress single-frame shimmering without ghosting.

---

### 3.7 Edge-Stopping Cross-Bilateral Spatial Denoiser

When `frame_idx < 20` (during or immediately after camera movement), `cs_main` writes normalized primary hit depth $z \in [0.02, 0.99]$ into the alpha channel of `output_tex`. The fullscreen fragment pass (`fs_main`) runs a $3 \times 3$ **Edge-Stopping Cross-Bilateral Filter**:

$$C_{\text{out}}(\mathbf{p}) = \frac{\sum_{\mathbf{q} \in \mathcal{N}(\mathbf{p})} w_s(\mathbf{p}, \mathbf{q}) \, w_z(\mathbf{p}, \mathbf{q}) \, w_l(\mathbf{p}, \mathbf{q}) \, C(\mathbf{q})}{\sum_{\mathbf{q} \in \mathcal{N}(\mathbf{p})} w_s(\mathbf{p}, \mathbf{q}) \, w_z(\mathbf{p}, \mathbf{q}) \, w_l(\mathbf{p}, \mathbf{q})}$$

Once progressive accumulation reaches `SPP >= 20`, `cs_main` writes `alpha = 0.0`, bypassing the spatial filter completely so the converged render retains 100% native pixel sharpness.

---

## 4. Engineering Post-Mortem: Visual & Systems Debugging Progression

| Iteration | Issue / Symptom Encountered | Root Cause Analysis | Engineering Fix Implemented |
| :--- | :--- | :--- | :--- |
| **Build 1** | `error[E0308]` on `entry_point: Some("cs_main")` & WGSL `meta` keyword panic | `wgpu 22.1.0` expects `&str` instead of `Option<&str>`; `meta` is a future-reserved keyword in W3C WGSL (`naga`). | Unwrapped `entry_point` string literals and renamed `meta` $\rightarrow$ `frame_info` in `CameraUniform`. |
| **Build 2** | Monochrome gray scene (`38.6 FPS`) with white salt-and-pepper fireflies during WASD/mouse movement | Khronos `Sponza.gltf` sets `baseColorFactor = [1,1,1,1]` on all materials and stores colors in external textures. Unweighted 1-SPP bounces into sunlit floor patches spiked 40x brighter than shadowed walls. | Added IGN-stratified short-range RTAO, bounded single-bounce albedo bleed, TAA-style motion history clamping, and a depth-aware cross-bilateral denoiser. |
| **Build 3** | Heuristic material classifier colored curtains beige and cauldrons/lion head green (`55.3 FPS`) | Khronos `Sponza.gltf` names texture files with 64-bit numeric hashes (e.g., `5784830445105232510.jpg`) and merges opposing curtain bays into single meshes centered at `(0, 0, 0)`. | Replaced heuristics with a parallel 8-thread HTTP texture provisioner, parallel `image` decoding, a 25-layer `Rgba8UnormSrgb` GPU `Texture2DArray`, and ray-traced alpha-cutout masks (`43.3 FPS`). |
| **Build 4** | Audit hardening: BVH stack depth invariant, atomic download cache, and 24-bit PRNG precision | Unbounded tree depth could overflow a 32-entry GPU stack; interrupted downloads could leave truncated files cached; `f32(seed) / 4294967295.0` could round to `1.0`. | Enforced `MAX_BVH_DEPTH = 31` + `assert!(max_depth <= 32)`, pinned commit SHA `723ffc6`, added `.take()` + `Content-Length` + `.part` atomic rename, and switched `rand_f32` to `f32(*seed >> 8u) * (1.0 / 16777216.0)`. |

---

## 5. Build, CLI Flags & Interactive Viewport Controls

### Build & Run
```powershell
# Auto-detect best GPU & backend (Selects Discrete GPU + DX12 on Windows / Vulkan on Linux)
cargo run --release

# Force Vulkan backend explicitly
cargo run --release -- --vulkan

# Force DirectX 12 backend explicitly (Windows)
cargo run --release -- --dx12
```

### Interactive 6-DOF Viewport Controls
| Input | Action |
| :--- | :--- |
| **Left Mouse Click** | Lock cursor into viewport for 360° free-look |
| **Mouse Motion** | Yaw / Pitch camera rotation |
| **Escape** | Unlock mouse cursor (press again while unlocked to exit) |
| **W / A / S / D** | Move Forward / Strafe Left / Backward / Strafe Right |
| **Space / Left Shift** | Ascend / Descend vertically between courtyard and upper gallery |
| **Left Ctrl (Hold)** | Sprint ($2.75\times$ movement speed) |
| **Q / E** | Rotate Sun Azimuth Angle in real time (dynamic ray-traced shadows) |
| **R** | Reset camera to the main Sponza Atrium colonnade |

---

## 6. Hardware Telemetry & Benchmark Reference

Captured on Windows `x86_64` (8 logical cores) at `1280x720`:

| Telemetry Metric | Measured Value |
| :--- | :--- |
| **Locked GPU Adapter** | `NVIDIA GeForce RTX 3050 Laptop GPU` (`DiscreteGpu`, Vendor `0x10DE`) |
| **Graphics Backend** | `DirectX 12` (Score: `1300`) / `Vulkan` (Score: `1250`) |
| **Scene Geometry** | `262,266` Textured Triangles (`Sponza.gltf` + `9.09 MB` `Sponza.bin` @ `723ffc6`) |
| **SAH BVH Acceleration Tree** | `163,553` Nodes (`5.23 MB` SSBO, built in `212.30 ms` via 12-bin SAH, depth $\le 32$) |
| **GPU Texture2DArray** | `25` Layers $\times$ `512x512` `Rgba8UnormSrgb` (`25.0 MB` VRAM) |
| **Real-Time Ray-Traced Framerate** | **`43.3 FPS`** (Base-Color Texture Array + Alpha-Cutout BVH + Shadows + RTAO) |
