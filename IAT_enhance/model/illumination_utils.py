import torch
import torch.nn.functional as F

def estimate_illumination(img):
    """
    Estimate illumination map from RGB image
    
    Args:
        img: [B, 3, H, W] RGB image
    Returns:
        illum: [B, 1, H, W] Illumination map
    """
    # RGB to grayscale (luminance)
    # ITU-R BT.601 standard
    gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    
    # Gaussian blur to get low-frequency component (illumination)
    kernel_size = 15
    padding = kernel_size // 2
    
    # Average pooling as approximation of Gaussian blur
    illum = F.avg_pool2d(gray, kernel_size=kernel_size, stride=1, padding=padding)
    
    return illum

def rgb_to_hsv(rgb):
    """
    Convert RGB to HSV
    Args:
        rgb: [B, 3, H, W] in range [0, 1]
    Returns:
        hsv: [B, 3, H, W] where H in [0, 1], S in [0, 1], V in [0, 1]
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    
    max_val, max_idx = torch.max(rgb, dim=1)
    min_val, _ = torch.min(rgb, dim=1)
    diff = max_val - min_val
    
    # Saturation
    s = torch.where(max_val != 0, diff / (max_val + 1e-7), torch.zeros_like(max_val))
    
    # Hue
    h = torch.zeros_like(max_val)
    
    mask_r = (max_idx == 0)
    mask_g = (max_idx == 1)
    mask_b = (max_idx == 2)
    
    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / (diff[mask_r] + 1e-7)) % 360) / 360.0
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / (diff[mask_g] + 1e-7)) + 120) / 360.0
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / (diff[mask_b] + 1e-7)) + 240) / 360.0
    
    # Value
    v = max_val
    
    return torch.stack([h, s, v], dim=1)

def hsv_to_rgb(hsv):
    """
    Convert HSV to RGB
    Args:
        hsv: [B, 3, H, W]
    Returns:
        rgb: [B, 3, H, W]
    """
    h, s, v = hsv[:, 0] * 360, hsv[:, 1], hsv[:, 2]
    
    c = v * s
    x = c * (1 - torch.abs((h / 60) % 2 - 1))
    m = v - c
    
    rgb = torch.zeros_like(hsv)
    
    mask = (h >= 0) & (h < 60)
    rgb[:, 0][mask] = c[mask]
    rgb[:, 1][mask] = x[mask]
    rgb[:, 2][mask] = 0
    
    mask = (h >= 60) & (h < 120)
    rgb[:, 0][mask] = x[mask]
    rgb[:, 1][mask] = c[mask]
    rgb[:, 2][mask] = 0
    
    mask = (h >= 120) & (h < 180)
    rgb[:, 0][mask] = 0
    rgb[:, 1][mask] = c[mask]
    rgb[:, 2][mask] = x[mask]
    
    mask = (h >= 180) & (h < 240)
    rgb[:, 0][mask] = 0
    rgb[:, 1][mask] = x[mask]
    rgb[:, 2][mask] = c[mask]
    
    mask = (h >= 240) & (h < 300)
    rgb[:, 0][mask] = x[mask]
    rgb[:, 1][mask] = 0
    rgb[:, 2][mask] = c[mask]
    
    mask = (h >= 300) & (h < 360)
    rgb[:, 0][mask] = c[mask]
    rgb[:, 1][mask] = 0
    rgb[:, 2][mask] = x[mask]
    
    rgb = rgb + m.unsqueeze(1)
    
    return torch.clamp(rgb, 0, 1)

def apply_saturation_enhancement(base_rgb, sat_map, illum_map, boost_factor=0.5):
    """
    Apply illumination-guided saturation enhancement
    
    Args:
        base_rgb: [B, 3, H, W] - base model output
        sat_map: [B, 1, H, W] - predicted saturation adjustment map
        illum_map: [B, 1, H, W] - illumination map
        boost_factor: float - maximum boost amount
    Returns:
        enhanced_rgb: [B, 3, H, W]
    """
    # RGB to HSV
    hsv = rgb_to_hsv(base_rgb)
    h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    
    # Illumination-based weighting
    # Low illumination → High weight (more boost needed)
    illum_weight = 1.0 - illum_map
    
    # Calculate boost amount
    # sat_map: learned spatial adjustment
    # illum_weight: illumination-based prior
    boost = sat_map * illum_weight * boost_factor
    
    # Apply to saturation
    # (1 - s): already saturated regions get less boost
    s_enhanced = s + boost * (1.0 - s)
    s_enhanced = torch.clamp(s_enhanced, 0.0, 1.0)
    
    # HSV to RGB
    hsv_enhanced = torch.cat([h, s_enhanced, v], dim=1)
    enhanced_rgb = hsv_to_rgb(hsv_enhanced)
    
    return enhanced_rgb