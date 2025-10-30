import os
import glob
import torch
import numpy as np
import torchvision
from PIL import Image

def ball2envmap(chromeball, msaa_scale, envmap_height):
    I = np.array([1,0,0]) # incoming vector, pointing to the camera
    env_grid = create_envmap_grid(envmap_height * msaa_scale)
    reflect_vec = get_cartesian_from_spherical(env_grid[...,1], env_grid[...,0])
    normal = get_normal_vector(I[None,None], reflect_vec)
    pos = (normal + 1.0) / 2
    pos  = 1.0 - pos
    pos = pos[...,1:]
    with torch.no_grad():
        grid = torch.from_numpy(pos)[None].float()
        grid = grid * 2 - 1
        ball_image = chromeball.permute(0,3,1,2) # [1,3,H,W]
        envmap = torch.nn.functional.grid_sample(
            ball_image, grid, mode='bilinear',
            padding_mode='border', align_corners=True
        )
        envmap = torch.nn.functional.interpolate(
            envmap, size=(envmap_height, envmap_height*2),
            mode='bilinear', align_corners=False
        )
        envmap = envmap.permute(0,2,3,1) # [1,H,W,3]
    return envmap

def create_envmap_grid(size: int):
    theta = torch.linspace(0, np.pi * 2, size * 2)
    phi = torch.linspace(0, np.pi, size)
    theta, phi = torch.meshgrid(theta, phi, indexing='xy')
    theta_phi = torch.cat([theta[..., None], phi[..., None]], dim=-1)
    return theta_phi.numpy()

def get_cartesian_from_spherical(theta: np.array, phi: np.array, r = 1.0):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.concatenate([x[...,None],y[...,None],z[...,None]], axis=-1)

def get_normal_vector(incoming_vector: np.ndarray, reflect_vector: np.ndarray):
    N = (incoming_vector + reflect_vector) / np.linalg.norm(
        incoming_vector + reflect_vector, axis=-1, keepdims=True
    )
    return N



if __name__ == "__main__":
    # ==== 경로 설정 ====
    input_dir = "/content/drive/MyDrive/LOL-v2/Real_captured/Train/Chrome_ball"
    output_dir = "/content/drive/MyDrive/LOL-v2/Real_captured/Train/Env_map"
    os.makedirs(output_dir, exist_ok=True)

    # ==== 파라미터 ====
    BALL_SIZE = 256
    MSAA_SCALE = 4
    ENVMAP_SIZE = 256

    # ==== 처리 ====
    image_paths = glob.glob(os.path.join(input_dir, "*.png"))
    print(f"Found {len(image_paths)} images")

    for img_path in image_paths:
        # 원본 파일 읽기
        image = Image.open(img_path).convert("RGB")
        tensor_img = torchvision.transforms.functional.pil_to_tensor(image).float() / 255.0
        
        # 크롭: 중앙에서 BALL_SIZE 크롭
        c = tensor_img.shape[1] // 2
        ball_crop = tensor_img[:, c - BALL_SIZE//2 : c + BALL_SIZE//2,
                                c - BALL_SIZE//2 : c + BALL_SIZE//2]
        
        # envmap 생성
        envmap = ball2envmap(ball_crop.permute(1,2,0)[None], MSAA_SCALE, ENVMAP_SIZE)
        envmap_np = envmap.cpu().numpy()[0]  # (H, W, 3)
        
        # 저장 경로 변환
        base = os.path.basename(img_path)
        name, _ = os.path.splitext(base)
        name = name.replace("chrome_ball", "env_map")  # 파일명 치환
        save_path = os.path.join(output_dir, f"{name}.npy")
        
        # npy 저장
        np.save(save_path, envmap_np)
        print(f"Saved: {save_path}")
