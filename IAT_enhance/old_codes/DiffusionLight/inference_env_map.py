import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import os

from .chromeball_utils import init_chromeball_pipeline, generate_chromeball
from .generate_env_map import ball2envmap


def make_envmap_from_tensor(tensor_img, ev, seed, pipe, depth_estimator,
                            BALL_SIZE=256, MSAA_SCALE=4, ENVMAP_SIZE=256, device="cuda"):
    """
    tensor_img: torch.Tensor [3,H,W] (0~1 범위, cuda 가능) 또는 PIL.Image
    """
    with torch.no_grad():
        # ==========================
        # (1) Tensor → PIL 변환
        # ==========================
        if isinstance(tensor_img, torch.Tensor):
            assert tensor_img.ndim == 3 and tensor_img.shape[0] == 3, "Tensor input must be [3,H,W]"
            pil_image = torchvision.transforms.functional.to_pil_image(
                tensor_img.detach().cpu().clamp(0, 1)
            )
        elif isinstance(tensor_img, Image.Image):
            pil_image = tensor_img.convert("RGB")
        else:
            raise TypeError("tensor_img must be torch.Tensor [3,H,W] or PIL.Image")

        # ==========================
        # (2) 크롬볼 생성 (PIL 기반)
        # ==========================
        chromeball_img = generate_chromeball(
            pipe=pipe,
            depth_estimator=depth_estimator,
            input_image=pil_image,   # ✅ PIL만 넘김
            ev=ev,
            seed=seed,
        )

        # ==========================
        # (3) chromeball PIL → Tensor
        # ==========================
        chromeball_tensor = torchvision.transforms.functional.pil_to_tensor(chromeball_img).float() / 255.0

        # 중앙 crop
        c = chromeball_tensor.shape[1] // 2
        ball_crop = chromeball_tensor[:, 
                                      c - BALL_SIZE // 2 : c + BALL_SIZE // 2,
                                      c - BALL_SIZE // 2 : c + BALL_SIZE // 2]

        # envmap 생성
        envmap = ball2envmap(ball_crop.permute(1, 2, 0)[None], MSAA_SCALE, ENVMAP_SIZE)
        envmap = envmap.permute(0, 3, 1, 2)[0]  # (3,H,W)

        return envmap.to(device), chromeball_img

