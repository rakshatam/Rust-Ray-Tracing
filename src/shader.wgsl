// ============================================================================
// Sponza Real-Time GPU Compute Ray Tracer & Bilateral Denoiser (WGSL)
// - Ordered DFS BVH traversal with ray-traced alpha-cutout masks
// - Ray-traced sun shadows, short-range RTAO & approximate single-bounce bleed
// - Base-color Texture2DArray (sRGB) + Blinn-Phong specular
// - Interleaved Gradient Noise (IGN) & 3x3 cross-bilateral spatial filter
// ============================================================================

struct CameraUniform {
    cam_pos_fov: vec4<f32>,      // xyz: eye position, w: tan(fov_y * 0.5)
    cam_fwd_aspect: vec4<f32>,   // xyz: forward dir,  w: aspect (width / height)
    cam_right_sun: vec4<f32>,    // xyz: right dir,    w: sun azimuth angle
    cam_up_rough: vec4<f32>,     // xyz: up dir,       w: global sky intensity
    frame_info: vec4<u32>,       // x: width, y: height, z: frame_idx, w: bvh_node_count
};

struct GpuTriangle {
    v0: vec4<f32>,               // xyz: vertex 0, w: uv0.x
    e1: vec4<f32>,               // xyz: v1 - v0,  w: uv0.y
    e2: vec4<f32>,               // xyz: v2 - v0,  w: uv1.x
    n0: vec4<f32>,               // xyz: normal 0, w: uv1.y
    n1: vec4<f32>,               // xyz: normal 1, w: uv2.x
    n2: vec4<f32>,               // xyz: normal 2, w: uv2.y
    albedo_rough: vec4<f32>,     // xyz: base_color_factor, w: roughness_factor
    emissive_mat: vec4<f32>,     // x: tex_layer_idx, y: alpha_cutoff, z: metallic_factor, w: pad
};

struct GpuBvhNode {
    aabb_min: vec3<f32>,
    left_or_first: u32,
    aabb_max: vec3<f32>,
    tri_count: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<storage, read> triangles: array<GpuTriangle>;
@group(0) @binding(2) var<storage, read> bvh_nodes: array<GpuBvhNode>;
@group(0) @binding(3) var<storage, read_write> accum_buffer: array<vec4<f32>>;
@group(0) @binding(4) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var mat_textures: texture_2d_array<f32>;

const PI: f32 = 3.141592653589793;
const T_MIN: f32 = 0.001;
const T_MAX: f32 = 10000.0;
const MAX_STACK_DEPTH: i32 = 32;
const TEX_SIZE: f32 = 512.0;
const TEX_MAX_I: i32 = 511;

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
    inv_dir: vec3<f32>,
};

struct HitInfo {
    t: f32,
    u: f32,
    v: f32,
    tri_idx: u32,
    hit: bool,
};

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Uniform float in [0.0, 1.0) using exact 24-bit IEEE-754 mantissa precision
fn rand_f32(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed >> 8u) * (1.0 / 16777216.0);
}

// Interleaved Gradient Noise (Jimenez 2014) for screen-space hemisphere stratification
fn ign_noise(px: u32, py: u32, frame: u32) -> f32 {
    let fpx = f32(px) + 5.588238 * f32(frame & 63u);
    let fpy = f32(py) + 5.588238 * f32(frame & 63u);
    return fract(52.9829189 * fract(0.06711056 * fpx + 0.00583715 * fpy));
}

fn interpolate_uv(tri: GpuTriangle, u: f32, v: f32) -> vec2<f32> {
    let w = 1.0 - u - v;
    let uv0 = vec2<f32>(tri.v0.w, tri.e1.w);
    let uv1 = vec2<f32>(tri.e2.w, tri.n0.w);
    let uv2 = vec2<f32>(tri.n1.w, tri.n2.w);
    return uv0 * w + uv1 * u + uv2 * v;
}

// 4-tap bilinear fetch from the GPU Texture2DArray (safe in divergent compute control flow)
fn sample_material_tex(uv: vec2<f32>, layer: i32) -> vec4<f32> {
    let u_wrap = uv.x - floor(uv.x);
    let v_wrap = uv.y - floor(uv.y);

    let fx = u_wrap * TEX_SIZE - 0.5;
    let fy = v_wrap * TEX_SIZE - 0.5;

    let x0 = i32(floor(fx)) & TEX_MAX_I;
    let y0 = i32(floor(fy)) & TEX_MAX_I;
    let x1 = (x0 + 1) & TEX_MAX_I;
    let y1 = (y0 + 1) & TEX_MAX_I;

    let wx = fx - floor(fx);
    let wy = fy - floor(fy);

    let c00 = textureLoad(mat_textures, vec2<i32>(x0, y0), layer, 0);
    let c10 = textureLoad(mat_textures, vec2<i32>(x1, y0), layer, 0);
    let c01 = textureLoad(mat_textures, vec2<i32>(x0, y1), layer, 0);
    let c11 = textureLoad(mat_textures, vec2<i32>(x1, y1), layer, 0);

    let top = mix(c00, c10, wx);
    let bot = mix(c01, c11, wx);
    return mix(top, bot, wy);
}

// Alpha-cutout test for masked foliage, flowers, thorns, and chains inside BVH traversal
fn passes_alpha_cutout(tri: GpuTriangle, u: f32, v: f32) -> bool {
    let cutoff = tri.emissive_mat.y;
    if (cutoff <= 0.0) {
        return true;
    }
    let uv = interpolate_uv(tri, u, v);
    let u_wrap = uv.x - floor(uv.x);
    let v_wrap = uv.y - floor(uv.y);
    let tx = i32(u_wrap * TEX_SIZE) & TEX_MAX_I;
    let ty = i32(v_wrap * TEX_SIZE) & TEX_MAX_I;
    let layer = i32(tri.emissive_mat.x + 0.5);
    let alpha = textureLoad(mat_textures, vec2<i32>(tx, ty), layer, 0).a;
    return alpha >= cutoff;
}

fn sample_cosine_hemisphere(n: vec3<f32>, r1_in: f32, r2_in: f32) -> vec3<f32> {
    let r1 = 2.0 * PI * r1_in;
    let r2s = sqrt(r2_in);
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    return normalize(
        tangent * (cos(r1) * r2s) +
        bitangent * (sin(r1) * r2s) +
        n * sqrt(max(0.0, 1.0 - r2_in))
    );
}

fn intersect_aabb(ray: Ray, bmin: vec3<f32>, bmax: vec3<f32>, max_t: f32) -> f32 {
    let t0 = (bmin - ray.origin) * ray.inv_dir;
    let t1 = (bmax - ray.origin) * ray.inv_dir;
    let tsmaller = min(t0, t1);
    let tbigger = max(t0, t1);
    let tmin = max(T_MIN, max(tsmaller.x, max(tsmaller.y, tsmaller.z)));
    let tmax = min(max_t, min(tbigger.x, min(tbigger.y, tbigger.z)));
    return select(T_MAX, tmin, tmin <= tmax);
}

fn intersect_triangle(ray: Ray, tri: GpuTriangle, max_t: f32) -> vec3<f32> {
    let e1 = tri.e1.xyz;
    let e2 = tri.e2.xyz;
    let h = cross(ray.dir, e2);
    let a = dot(e1, h);
    if (abs(a) < 1e-7) {
        return vec3<f32>(T_MAX, 0.0, 0.0);
    }
    let f = 1.0 / a;
    let s = ray.origin - tri.v0.xyz;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return vec3<f32>(T_MAX, 0.0, 0.0);
    }
    let q = cross(s, e1);
    let v = f * dot(ray.dir, q);
    if (v < 0.0 || (u + v) > 1.0) {
        return vec3<f32>(T_MAX, 0.0, 0.0);
    }
    let t = f * dot(e2, q);
    if (t > T_MIN && t < max_t) {
        return vec3<f32>(t, u, v);
    }
    return vec3<f32>(T_MAX, 0.0, 0.0);
}

fn trace_bvh(ray: Ray, max_dist: f32) -> HitInfo {
    var result: HitInfo;
    result.t = max_dist;
    result.u = 0.0;
    result.v = 0.0;
    result.tri_idx = 0u;
    result.hit = false;

    if (camera.frame_info.w == 0u) {
        return result;
    }

    var stack: array<u32, 32>;
    var stack_ptr: i32 = 0;
    var node_idx: u32 = 0u;

    let root = bvh_nodes[0u];
    if (intersect_aabb(ray, root.aabb_min, root.aabb_max, result.t) >= result.t) {
        return result;
    }

    loop {
        let node = bvh_nodes[node_idx];

        if (node.tri_count > 0u) {
            let first = node.left_or_first;
            let count = node.tri_count;
            for (var i: u32 = 0u; i < count; i = i + 1u) {
                let tri_id = first + i;
                let tri = triangles[tri_id];
                let tuv = intersect_triangle(ray, tri, result.t);
                if (tuv.x < result.t) {
                    if (passes_alpha_cutout(tri, tuv.y, tuv.z)) {
                        result.t = tuv.x;
                        result.u = tuv.y;
                        result.v = tuv.z;
                        result.tri_idx = tri_id;
                        result.hit = true;
                    }
                }
            }
            if (stack_ptr <= 0) {
                break;
            }
            stack_ptr = stack_ptr - 1;
            node_idx = stack[stack_ptr];
        } else {
            let left_idx = node_idx + 1u;
            let right_idx = node.left_or_first;

            let left_node = bvh_nodes[left_idx];
            let right_node = bvh_nodes[right_idx];

            let dist_l = intersect_aabb(ray, left_node.aabb_min, left_node.aabb_max, result.t);
            let dist_r = intersect_aabb(ray, right_node.aabb_min, right_node.aabb_max, result.t);

            let hit_l = dist_l < result.t;
            let hit_r = dist_r < result.t;

            if (hit_l && hit_r) {
                let near_idx = select(right_idx, left_idx, dist_l <= dist_r);
                let far_idx  = select(left_idx, right_idx, dist_l <= dist_r);
                if (stack_ptr < MAX_STACK_DEPTH) {
                    stack[stack_ptr] = far_idx;
                    stack_ptr = stack_ptr + 1;
                }
                node_idx = near_idx;
            } else if (hit_l) {
                node_idx = left_idx;
            } else if (hit_r) {
                node_idx = right_idx;
            } else {
                if (stack_ptr <= 0) {
                    break;
                }
                stack_ptr = stack_ptr - 1;
                node_idx = stack[stack_ptr];
            }
        }
    }

    return result;
}

fn trace_shadow(ray: Ray, max_dist: f32) -> bool {
    if (camera.frame_info.w == 0u) {
        return false;
    }

    var stack: array<u32, 32>;
    var stack_ptr: i32 = 0;
    var node_idx: u32 = 0u;

    let root = bvh_nodes[0u];
    if (intersect_aabb(ray, root.aabb_min, root.aabb_max, max_dist) >= max_dist) {
        return false;
    }

    loop {
        let node = bvh_nodes[node_idx];
        if (node.tri_count > 0u) {
            let first = node.left_or_first;
            let count = node.tri_count;
            for (var i: u32 = 0u; i < count; i = i + 1u) {
                let tri = triangles[first + i];
                let tuv = intersect_triangle(ray, tri, max_dist);
                if (tuv.x < max_dist) {
                    if (passes_alpha_cutout(tri, tuv.y, tuv.z)) {
                        return true;
                    }
                }
            }
            if (stack_ptr <= 0) {
                break;
            }
            stack_ptr = stack_ptr - 1;
            node_idx = stack[stack_ptr];
        } else {
            let left_idx = node_idx + 1u;
            let right_idx = node.left_or_first;
            let dist_l = intersect_aabb(ray, bvh_nodes[left_idx].aabb_min, bvh_nodes[left_idx].aabb_max, max_dist);
            let dist_r = intersect_aabb(ray, bvh_nodes[right_idx].aabb_min, bvh_nodes[right_idx].aabb_max, max_dist);
            let hit_l = dist_l < max_dist;
            let hit_r = dist_r < max_dist;

            if (hit_l && hit_r) {
                if (stack_ptr < MAX_STACK_DEPTH) {
                    stack[stack_ptr] = right_idx;
                    stack_ptr = stack_ptr + 1;
                }
                node_idx = left_idx;
            } else if (hit_l) {
                node_idx = left_idx;
            } else if (hit_r) {
                node_idx = right_idx;
            } else {
                if (stack_ptr <= 0) {
                    break;
                }
                stack_ptr = stack_ptr - 1;
                node_idx = stack[stack_ptr];
            }
        }
    }
    return false;
}

fn evaluate_surface_albedo(tri: GpuTriangle, u: f32, v: f32) -> vec3<f32> {
    let uv = interpolate_uv(tri, u, v);
    let layer = i32(tri.emissive_mat.x + 0.5);
    let tex_col = sample_material_tex(uv, layer).rgb;
    return tex_col * tri.albedo_rough.xyz;
}

fn evaluate_sky(dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let zenith = vec3<f32>(0.10, 0.30, 0.72);
    let horizon = vec3<f32>(0.65, 0.78, 0.92);
    let ground = vec3<f32>(0.18, 0.15, 0.12);
    let t = clamp(dir.y, 0.0, 1.0);
    var sky = mix(horizon, zenith, pow(t, 0.55));
    if (dir.y < 0.0) {
        sky = mix(horizon, ground, clamp(-dir.y * 4.0, 0.0, 1.0));
    }
    let sun_amt = max(dot(dir, sun_dir), 0.0);
    sky += vec3<f32>(1.0, 0.88, 0.65) * pow(sun_amt, 256.0) * 6.0;
    sky += vec3<f32>(1.0, 0.78, 0.52) * pow(sun_amt, 10.0) * 0.30;
    return sky * camera.cam_up_rough.w;
}

fn safe_inv_dir(d: vec3<f32>) -> vec3<f32> {
    let eps = 1e-8;
    let sx = select(-eps, eps, d.x >= 0.0);
    let sy = select(-eps, eps, d.y >= 0.0);
    let sz = select(-eps, eps, d.z >= 0.0);
    return vec3<f32>(
        1.0 / select(d.x, sx, abs(d.x) < eps),
        1.0 / select(d.y, sy, abs(d.y) < eps),
        1.0 / select(d.z, sz, abs(d.z) < eps)
    );
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = camera.frame_info.x;
    let height = camera.frame_info.y;
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let pixel_idx = gid.y * width + gid.x;
    let frame_idx = camera.frame_info.z;
    var seed = pcg_hash(pixel_idx ^ pcg_hash(frame_idx * 1664525u + 1013904223u));

    let jitter_x = select(0.5, rand_f32(&seed), frame_idx > 1u);
    let jitter_y = select(0.5, rand_f32(&seed), frame_idx > 1u);

    let uv_x = ((f32(gid.x) + jitter_x) / f32(width)) * 2.0 - 1.0;
    let uv_y = 1.0 - ((f32(gid.y) + jitter_y) / f32(height)) * 2.0;

    let fov_scale = camera.cam_pos_fov.w;
    let aspect = camera.cam_fwd_aspect.w;

    let ray_dir = normalize(
        camera.cam_fwd_aspect.xyz +
        camera.cam_right_sun.xyz * (uv_x * aspect * fov_scale) +
        camera.cam_up_rough.xyz * (uv_y * fov_scale)
    );

    var ray: Ray;
    ray.origin = camera.cam_pos_fov.xyz;
    ray.dir = ray_dir;
    ray.inv_dir = safe_inv_dir(ray.dir);

    let sun_az = camera.cam_right_sun.w;
    let sun_dir = normalize(vec3<f32>(cos(sun_az) * 0.28, 0.94, sin(sun_az) * 0.22));
    let sun_radiance = vec3<f32>(6.5, 5.6, 4.4);

    let hit = trace_bvh(ray, T_MAX);
    var radiance = vec3<f32>(0.0);
    var primary_depth: f32 = 1.0;

    if (!hit.hit) {
        radiance = evaluate_sky(ray.dir, sun_dir);
        primary_depth = 0.99;
    } else {
        primary_depth = clamp(hit.t / 35.0, 0.02, 0.98);
        let tri = triangles[hit.tri_idx];
        let w = 1.0 - hit.u - hit.v;
        var n = normalize(tri.n0.xyz * w + tri.n1.xyz * hit.u + tri.n2.xyz * hit.v);
        if (dot(n, ray.dir) > 0.0) {
            n = -n;
        }

        let world_pos = ray.origin + ray.dir * hit.t;
        let hit_pos = world_pos + n * 0.003;
        let albedo = evaluate_surface_albedo(tri, hit.u, hit.v);
        let roughness = clamp(tri.albedo_rough.w, 0.08, 0.98);
        let metallic = clamp(tri.emissive_mat.z, 0.0, 1.0);

        // 1. Direct Sun Lighting with Ray-Traced Shadows & Blinn-Phong Specular
        var test_sun = sun_dir;
        if (frame_idx > 2u) {
            let disk_r = 0.015 * sqrt(rand_f32(&seed));
            let disk_theta = 2.0 * PI * rand_f32(&seed);
            test_sun = normalize(sun_dir + vec3<f32>(cos(disk_theta) * disk_r, 0.0, sin(disk_theta) * disk_r));
        }

        let n_dot_l = max(dot(n, test_sun), 0.0);
        if (n_dot_l > 0.0) {
            var shadow_ray: Ray;
            shadow_ray.origin = hit_pos;
            shadow_ray.dir = test_sun;
            shadow_ray.inv_dir = safe_inv_dir(shadow_ray.dir);

            if (!trace_shadow(shadow_ray, T_MAX)) {
                let half_vec = normalize(test_sun - ray.dir);
                let n_dot_h = max(dot(n, half_vec), 0.0);
                let spec_pow = mix(96.0, 8.0, roughness);
                let spec_col = mix(vec3<f32>(0.04), albedo, metallic);
                let spec = spec_col * pow(n_dot_h, spec_pow) * (1.0 - roughness * 0.7) * 1.5;
                radiance += (albedo * (1.0 - metallic * 0.7) * (n_dot_l / PI) + spec) * sun_radiance;
            }
        }

        // 2. Short-Range Ray-Traced Ambient Occlusion (RTAO) & Approximate Bounce Bleed
        let r1 = ign_noise(gid.x, gid.y, frame_idx);
        let r2 = fract(r1 + 0.6180339887 * f32((frame_idx & 31u) + 1u));
        let bounce_dir = sample_cosine_hemisphere(n, r1, r2);

        var ao_ray: Ray;
        ao_ray.origin = hit_pos;
        ao_ray.dir = bounce_dir;
        ao_ray.inv_dir = safe_inv_dir(ao_ray.dir);

        let ao_hit = trace_bvh(ao_ray, 8.5);
        var ao_factor = 1.0;
        var bounce_bleed = vec3<f32>(0.0);

        if (ao_hit.hit) {
            ao_factor = clamp(ao_hit.t / 4.2, 0.20, 1.0);
            let b_tri = triangles[ao_hit.tri_idx];
            let b_albedo = evaluate_surface_albedo(b_tri, ao_hit.u, ao_hit.v);
            bounce_bleed = b_albedo * 0.42;
        } else {
            bounce_bleed = vec3<f32>(0.32, 0.42, 0.58);
        }

        // 3. Analytical Hemisphere Sky/Floor Fill + RTAO Modulation
        let sky_hemi = 0.5 + 0.5 * n.y;
        let floor_hemi = 0.5 - 0.5 * n.y;
        let sky_light = vec3<f32>(0.34, 0.44, 0.62) * sky_hemi;
        let floor_bounce = vec3<f32>(0.45, 0.36, 0.26) * floor_hemi;
        let corridor_fill = vec3<f32>(0.18, 0.16, 0.14) * abs(n.z);

        let indirect_approx = (sky_light + floor_bounce + corridor_fill + bounce_bleed) * ao_factor;
        radiance += albedo * indirect_approx;
    }

    // Temporal accumulation with neighborhood-clamped motion blend on frame 0
    let prev = accum_buffer[pixel_idx].xyz;
    var accumulated = radiance;
    if (frame_idx == 0u) {
        let prev_clamped = clamp(prev, radiance * 0.55, radiance * 1.45 + vec3<f32>(0.08));
        accumulated = mix(radiance, prev_clamped, 0.22);
    } else {
        let weight = 1.0 / f32(min(frame_idx + 1u, 256u));
        accumulated = mix(prev, radiance, weight);
    }
    accum_buffer[pixel_idx] = vec4<f32>(accumulated, primary_depth);

    // ACES Filmic Tone Mapping + Linear-to-sRGB Gamma
    let mapped = pow(aces_tonemap(accumulated * 1.15), vec3<f32>(1.0 / 2.2));
    let out_alpha = select(0.0, primary_depth, frame_idx < 20u);
    textureStore(output_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(mapped, out_alpha));
}

// ============================================================================
// Fullscreen Edge-Preserving Cross-Bilateral Denoiser & Blit Pass
// ============================================================================

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@group(0) @binding(0) var display_tex: texture_2d<f32>;
@group(0) @binding(1) var display_sampler: sampler;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(display_tex));
    let center = textureSample(display_tex, display_sampler, in.uv);

    if (center.a <= 0.001) {
        return vec4<f32>(center.rgb, 1.0);
    }

    let inv_res = 1.0 / dims;
    let center_luma = dot(center.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let center_depth = center.a;

    var sum_color = center.rgb * 1.5;
    var sum_weight = 1.5;

    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            let offset = vec2<f32>(f32(dx), f32(dy)) * inv_res;
            let sample_val = textureSample(display_tex, display_sampler, in.uv + offset);
            let sample_luma = dot(sample_val.rgb, vec3<f32>(0.299, 0.587, 0.114));

            let depth_diff = abs(sample_val.a - center_depth);
            let luma_diff = abs(sample_luma - center_luma);

            let w_depth = exp(-depth_diff * depth_diff * 1200.0);
            let w_luma  = exp(-luma_diff * luma_diff * 18.0);
            let w_spatial = select(0.5, 0.75, dx == 0 || dy == 0);

            let w = w_spatial * w_depth * w_luma;
            sum_color += sample_val.rgb * w;
            sum_weight += w;
        }
    }

    return vec4<f32>(sum_color / sum_weight, 1.0);
}