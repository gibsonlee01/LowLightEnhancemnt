import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ==== 시각화 함수 ====
def visualize_envmap(npy_path, save_png=True):
    # envmap 불러오기
    envmap = np.load(npy_path)  # shape: (H, W, 3)
    
    # 값 범위 [0,1] 클램프
    envmap = np.clip(envmap, 0, 1)
    
    # PIL 이미지로 변환 (0~255 uint8)
    envmap_img = Image.fromarray((envmap * 255).astype(np.uint8))
    
    # 디스플레이
    plt.figure(figsize=(10,5))
    plt.imshow(envmap_img)
    plt.axis("off")
    plt.title(os.path.basename(npy_path))
    plt.show()
    
    # PNG 저장 옵션
    if save_png:
        png_path = npy_path.replace(".npy", ".png")
        envmap_img.save(png_path)
        print(f"Saved visualization: {png_path}")

# ==== 예시 사용 ====
npy_file = "/content/drive/MyDrive/LOL-v2/Real_captured/Train/Env_map/normal00001_env_map.npy"
visualize_envmap(npy_file)
