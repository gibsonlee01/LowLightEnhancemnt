import torch
import torchvision.transforms.functional as TF
from PIL import Image
import os

from DiffusionLight.chromeball_utils import init_chromeball_pipeline, generate_chromeball

# 초기화
device = "cuda"
pipe, depth_estimator = init_chromeball_pipeline(device=device)

# 테스트용 경로
img_path = "/content/drive/MyDrive/LOL-v2/Real_captured/Train/Normal/normal00005.png"
save_dir = "/content/drive/MyDrive/LOL-v2/Test_image/"
os.makedirs(save_dir, exist_ok=True)

# =======================
# 1) PIL 이미지로 실행
# =======================
pil_img = Image.open(img_path).convert("RGB")
try:
    out_pil = generate_chromeball(pipe, depth_estimator, pil_img, ev=0)
    out_pil.save(os.path.join(save_dir, "chrome_pil.png"))
    print("✅ PIL 입력 성공: chrome_pil.png 저장 완료")
except Exception as e:
    print("❌ PIL 입력 실패:", e)

# =======================
# 2) Tensor 이미지로 실행
# =======================
tensor_img = TF.to_tensor(pil_img).cuda()  # [3,H,W], float
try:
    out_tensor = generate_chromeball(pipe, depth_estimator, tensor_img, ev=0)
    out_tensor.save(os.path.join(save_dir, "chrome_tensor.png"))
    print("✅ Tensor 입력 성공: chrome_tensor.png 저장 완료")
except Exception as e:
    print("❌ Tensor 입력 실패:", e)
