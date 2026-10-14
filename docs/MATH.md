# Mathematical Derivations & Low-Level GPU Reference

This document serves as the mathematical and systems engineering companion to `src/main.rs` and `src/shader.wgsl`.

---

## 1. Scene-Graph World & Normal Matrix Transformation

In `Sponza.gltf`, meshes are organized in a hierarchical node tree where each node $i$ defines a local affine transformation matrix $\mathbf{M}_{\text{local}, i} \in \mathbb{R}^{4 \times 4}$. During recursive traversal (`traverse_gltf_node`), the global world matrix is accumulated via pre-multiplication:

$$\mathbf{M}_{\text{world}} = \mathbf{M}_{\text{parent}} \cdot \mathbf{M}_{\text{local}}$$

While vertex positions $\mathbf{p}$ transform directly as homogeneous points $\mathbf{p}' = \mathbf{M}_{\text{world}} \mathbf{p}$, **surface normals $\mathbf{n}$ cannot be multiplied by $\mathbf{M}_{\text{world}}$** if $\mathbf{M}_{\text{world}}$ contains non-uniform scaling (which shears the normal away from perpendicularity).

### Proof of the Inverse-Transpose Normal Matrix
Let $\mathbf{t} = \mathbf{p}_1 - \mathbf{p}_0$ be a tangent vector lying on a triangle surface, so $\mathbf{n}^T \mathbf{t} = 0$. After transforming the triangle by the upper-left $3 \times 3$ linear block $\mathbf{A}$ of $\mathbf{M}_{\text{world}}$, the transformed tangent is $\mathbf{t}' = \mathbf{A}\mathbf{t}$. We require the transformed normal $\mathbf{n}' = \mathbf{G}\mathbf{n}$ to remain orthogonal to $\mathbf{t}'$:

$$(\mathbf{n}')^T \mathbf{t}' = (\mathbf{G}\mathbf{n})^T (\mathbf{A}\mathbf{t}) = \mathbf{n}^T \mathbf{G}^T \mathbf{A} \mathbf{t} = 0$$

For this identity to hold for all tangents $\mathbf{t}$, we require $\mathbf{G}^T \mathbf{A} = \mathbf{I}$, which yields:

$$\mathbf{G} = \left(\mathbf{A}^{-1}\right)^T$$

In `src/main.rs`, this is computed once per glTF node:
```rust
let world_xform = parent_xform * local_xform;
let normal_xform = world_xform.inverse().transpose();
```

---

## 2. Branchless Ray-AABB Slab Intersection

An Axis-Aligned Bounding Box (AABB) is the intersection of three orthogonal slabs $[b_{\min, x}, b_{\max, x}]$, $[b_{\min, y}, b_{\max, y}]$, and $[b_{\min, z}, b_{\max, z}]$.

For a ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$, the distances to the two bounding planes along axis $i \in \{x, y, z\}$ are:

$$t_{0, i} = (b_{\min, i} - o_i) \cdot \frac{1}{d_i}, \qquad t_{1, i} = (b_{\max, i} - o_i) \cdot \frac{1}{d_i}$$

By precomputing `inv_dir` ($\frac{1}{d_x}, \frac{1}{d_y}, \frac{1}{d_z}$) once per ray with a branchless epsilon guard (`safe_inv_dir`), each AABB test requires only one vector subtraction, one vector multiplication, and component-wise `min`/`max` intrinsics:

$$t_{\text{enter}} = \max\left(t_{\min}, \max_{i \in \{x,y,z\}} \min(t_{0, i}, t_{1, i})\right)$$

$$t_{\text{exit}} = \min\left(t_{\max}, \min_{i \in \{x,y,z\}} \max(t_{0, i}, t_{1, i})\right)$$

The ray intersects the box within the active interval $[t_{\min}, t_{\max}]$ if and only if $t_{\text{enter}} \le t_{\text{exit}}$.

---

## 3. Malley's Method for Cosine-Weighted Hemisphere Sampling

To compute Lambertian indirect illumination without explicit cosine-weight variance spikes, we sample outgoing ray directions $\omega_i$ proportional to $\cos\theta = \mathbf{n} \cdot \omega_i$, with Probability Density Function (PDF):

$$p(\omega_i) = \frac{\cos\theta}{\pi}$$

Using **Malley's Method** (`sample_cosine_hemisphere`), we generate a uniform point on the unit disk $(r \cos\phi, r \sin\phi)$ from two uniform random variables $(\xi_1, \xi_2) \in [0, 1)^2$ via area-preserving concentric mapping:

$$\phi = 2\pi \xi_1, \qquad r = \sqrt{\xi_2}$$

And project the disk point vertically onto the unit hemisphere above normal $\mathbf{n}$:

$$\omega_i = \mathbf{T}(r \cos\phi) + \mathbf{B}(r \sin\phi) + \mathbf{n}\sqrt{\max(0, 1 - \xi_2)}$$

where $(\mathbf{T}, \mathbf{B}, \mathbf{n})$ forms an orthonormal Frisvad/Duff tangent frame around $\mathbf{n}$. Because the $\cos\theta$ in the Lambertian BRDF numerator cancels the $\cos\theta$ in the PDF denominator:

$$\frac{f_r(\omega_i, \omega_o) \, L_i(\omega_i) \cos\theta}{p(\omega_i)} = \frac{\frac{\rho}{\pi} L_i(\omega_i) \cos\theta}{\frac{\cos\theta}{\pi}} = \rho \, L_i(\omega_i)$$

every sample has uniform throughput weight $\rho$ (the surface albedo), eliminating grazing-angle fireflies.

---

## 4. ACES Filmic Tone Mapping Curve

High Dynamic Range (HDR) linear radiance $\mathbf{x} \in [0, \infty)^3$ accumulated in `accum_buffer` is compressed into Display-Referred $[0, 1]^3$ using Narkowicz's rational polynomial approximation of the Academy Color Encoding System (ACES) Reference Rendering Transform (RRT) + Output Device Transform (ODT):

$$\text{ACES}(\mathbf{x}) = \text{clamp}\left(\frac{\mathbf{x}(2.51\mathbf{x} + 0.03)}{\mathbf{x}(2.43\mathbf{x} + 0.59) + 0.14}, \, 0.0, \, 1.0\right)$$

This provides a filmic toe in deep courtyard shadows and smooth highlight roll-off on sunlit limestone columns before applying linear-to-sRGB transfer $\mathbf{c}_{\text{sRGB}} = \text{ACES}(\mathbf{x})^{1/2.2}$.