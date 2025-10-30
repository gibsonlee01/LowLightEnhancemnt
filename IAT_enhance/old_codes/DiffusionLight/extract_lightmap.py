import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# -------------------------
# SH basis (real SH up to l=2 -> 9 coefficients)
# input: x,y,z arrays (same shape)
# output: array with last dim = 9
# -------------------------
def sh_basis_grid(x, y, z):
    # constants for real SH (orthonormal-ish)
    Y0 = 0.282095 * np.ones_like(x)            # l=0, m=0
    Y1 = 0.488603 * y                           # l=1, m=-1
    Y2 = 0.488603 * z                           # l=1, m=0
    Y3 = 0.488603 * x                           # l=1, m=1
    Y4 = 1.092548 * x * y                       # l=2, m=-2
    Y5 = 1.092548 * y * z                       # l=2, m=-1
    Y6 = 0.315392 * (3.0 * z * z - 1.0)         # l=2, m=0
    Y7 = 1.092548 * x * z                       # l=2, m=1
    Y8 = 0.546274 * (x * x - y * y)             # l=2, m=2

    # stack into (..., 9)
    return np.stack([Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8], axis=-1)


# -------------------------
# envmap -> SH coefficients (vectorized)
# envmap: (H, W, C) float (can be HDR)
# n_coeff: up to 9 (we implement 9)
# returns coeffs shape (C, n_coeff)
# -------------------------
def envmap_to_sh(envmap, n_coeff=9):
    assert n_coeff <= 9, "This implementation supports up to 9 coeffs (l<=2)."
    env = np.asarray(envmap, dtype=np.float64)
    H, W, C = env.shape

    # theta, phi centers for each pixel (equirectangular)
    ys = (np.arange(H) + 0.5) / H            # [0,1)
    xs = (np.arange(W) + 0.5) / W            # [0,1)
    theta = np.pi * ys                       # 0..pi
    phi = 2.0 * np.pi * xs                   # 0..2pi

    # compute directional vectors grid
    sin_theta = np.sin(theta)                # shape (H,)
    cos_theta = np.cos(theta)                # shape (H,)
    sin_theta = sin_theta[:, None]           # (H,1)
    cos_theta = cos_theta[:, None]           # (H,1)

    cos_phi = np.cos(phi)[None, :]           # (1,W)
    sin_phi = np.sin(phi)[None, :]           # (1,W)

    x = (sin_theta * cos_phi)  # (H,W)
    y = (sin_theta * sin_phi)  # (H,W)
    z = (cos_theta * np.ones_like(phi)[None, :])  # (H,W)

    # SH basis grid (H,W,9)
    Y = sh_basis_grid(x, y, z)  # (H, W, 9)

    # area factor per pixel: dΩ ≈ sinθ * dθ * dφ,
    # with dθ ≈ π/H, dφ ≈ 2π/W -> area_factor = (π/H) * (2π/W) = 2 * π^2 / (H*W)
    area_scalar = 2.0 * (np.pi ** 2) / (H * W)  # scalar
    # we'll multiply Y by sin_theta already inside x,y,z calc; but we must still include area_scalar
    # note: we already used sin(theta) when making x,y,z (not for weight), must still include sin(theta) weight separately.
    # create sin_theta grid for weighting:
    sin_theta_grid = np.sin(theta)[:, None]  # (H,1) used for weighting

    # compute coefficients per channel
    coeffs = np.zeros((C, n_coeff), dtype=np.float64)
    # vectorized sum: sum over H and W
    # weight = sin_theta_grid * (dθ * dφ) i.e., multiply by area_scalar (which already includes dθ*dφ factor)
    weight = sin_theta_grid  # (H,1)
    # compute weighted inner products
    for c in range(C):
        # env[:,:,c] shape (H,W); expand to (H,W,1)
        val = env[:, :, c][:, :, None]  # (H,W,1)
        # multiply: val * Y * weight  -> shape (H,W,9)
        weighted = val * Y * weight[:, :, None]
        # sum over H,W
        s = weighted.sum(axis=(0, 1))   # (9,)
        coeffs[c, :] = s * area_scalar

    return coeffs  # (C, n_coeff)


# -------------------------
# SH coefficients -> envmap reconstruction
# coeffs: (C, n_coeff)
# returns envmap_sh (H,W,C)
# -------------------------
def sh_to_envmap(coeffs, H, W):
    n_coeff = coeffs.shape[1]
    # prepare direction grid same as above
    ys = (np.arange(H) + 0.5) / H
    xs = (np.arange(W) + 0.5) / W
    theta = np.pi * ys
    phi = 2.0 * np.pi * xs

    sin_theta = np.sin(theta)[:, None]
    cos_theta = np.cos(theta)[:, None]
    cos_phi = np.cos(phi)[None, :]
    sin_phi = np.sin(phi)[None, :]

    x = (sin_theta * cos_phi)
    y = (sin_theta * sin_phi)
    z = (cos_theta * np.ones_like(phi)[None, :])

    Y = sh_basis_grid(x, y, z)[:, :, :n_coeff]  # (H,W,n_coeff)

    out = np.zeros((H, W, coeffs.shape[0]), dtype=np.float64)
    for c in range(coeffs.shape[0]):
        # dot across coefficients
        out[:, :, c] = np.tensordot(Y, coeffs[c, :], axes=([2], [0]))

    return out


# -------------------------
# Visualize helper that uses above functions
# -------------------------
def visualize_envmap_sh(npy_path, n_coeff=9, save_png=True, normalize_for_display=True):
    envmap = np.load(npy_path).astype(np.float64)  # (H, W, 3) -- can be HDR
    H, W, C = envmap.shape
    print(f"Loaded envmap: shape={envmap.shape}, min={envmap.min():.6f}, max={envmap.max():.6f}")

    coeffs = envmap_to_sh(envmap, n_coeff=n_coeff)
    print("SH coeffs (per channel):")
    for c_idx, cname in enumerate(['R','G','B']):
        print(f"  {cname}: {coeffs[c_idx,:]}")

    envmap_sh = sh_to_envmap(coeffs, H, W)
    print(f"Reconstructed SH envmap: min={envmap_sh.min():.6f}, max={envmap_sh.max():.6f}")

    # for visualization, optionally normalize the SH image to [0,1] (preserves relative structure)
    if normalize_for_display:
        # avoid all-zero division
        mn, mx = envmap_sh.min(), envmap_sh.max()
        if mx - mn > 1e-8:
            disp_sh = (envmap_sh - mn) / (mx - mn)
        else:
            disp_sh = envmap_sh.copy()
    else:
        disp_sh = np.clip(envmap_sh, 0, 1)

    # show side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(np.clip(envmap, 0, None))
    axs[0].set_title("Original Envmap (clipped for display)")
    axs[0].axis("off")
    axs[1].imshow(disp_sh)
    axs[1].set_title(f"SH reconstructed (n_coeff={n_coeff})")
    axs[1].axis("off")
    plt.show()

    # save if required (save normalized SH for visibility)
    if save_png:
        base = "/content/drive/MyDrive/IAT_test/IAT_enhance/DiffusionLight"
        Image.fromarray(np.uint8(np.clip(envmap / max(1.0, envmap.max()), 0, 1) * 255)).save(base + "_orig.png")
        Image.fromarray(np.uint8(np.clip(disp_sh, 0, 1) * 255)).save(base + "_sh.png")
        print(f"Saved: {base}_orig.png, {base}_sh.png")


# ==== example usage ====
if __name__ == "__main__":
    npy_file = "/content/drive/MyDrive/LOL-v2/Real_captured/Train/Env_map/normal00001_env_map.npy"
    visualize_envmap_sh(npy_file, n_coeff=9, save_png=True)
