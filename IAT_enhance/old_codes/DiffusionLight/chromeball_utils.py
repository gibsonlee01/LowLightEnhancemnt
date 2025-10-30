import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import pipeline
import torchvision

def init_chromeball_pipeline(
    device="cuda",
    turbo_lora_path="DiffusionLight/TurboLoRA",
    exposure_lora_path="DiffusionLight/ExposureLoRA",
    use_cpu_offload=False,
):
    """
    크롬볼 삽입용 파이프라인 초기화 (한 번만 실행)
    """
    # ControlNet 로드
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
        use_safetensors=True,
        variant="fp16" if "cuda" in str(device) else None,
    )
    controlnet = controlnet.to(device)

    # SDXL Inpaint + ControlNet
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
        add_watermarker=False,
        use_safetensors=True,
        variant="fp16" if "cuda" in str(device) else None,
    )
    pipe = pipe.to(device)

    # LoRA 로드
    pipe.load_lora_weights(turbo_lora_path, adapter_name="turbo")
    pipe.load_lora_weights(exposure_lora_path, adapter_name="exposure")

    # Scheduler 교체
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # 메모리 최적화
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()
    if use_cpu_offload:
        pipe.enable_sequential_cpu_offload()

    # ==========================
    # depth estimator
    # ==========================
    # pipeline은 int index로 GPU 지정 → device가 cuda:0이면 0, cpu면 -1
    device_index = 0 if "cuda" in str(device) else -1
    depth_estimator = pipeline("depth-estimation", device=device_index, use_fast=True)

    return pipe, depth_estimator

def generate_chromeball(
    pipe,
    depth_estimator,
    input_image,  # PIL.Image (RGB)
    prompt="a perfect mirrored reflective chrome ball sphere",
    prompt_black="a perfect black dark mirrored reflective chrome ball sphere",
    negative_prompt="matte, diffuse, flat, dull",
    seed=200,
    ev=0,
    lowest_ev=-5,
    switch_lora_timestep=800,
    turbo_lora_scale=1.0,
    exposure_lora_scale=0.75,
    condition_scale=0.5,
    num_steps=30,
    ball_size=256,
    dilate_pixel=10,
    save_path=None,
):
    """
    크롬볼 삽입 실행 (PIL 기반)
    - input_image: PIL.Image (RGB)
    - save_path: 지정 시 파일 저장, 아니면 PIL 반환
    """
    # ==========================
    # (1) 입력 이미지 준비 (1024x1024)
    # ==========================
    pil_for_depth = pil_square_image(input_image.convert("RGB"), (1024, 1024))
    init_image = pil_for_depth

    # ==========================
    # (2) depth estimation (PIL)
    # ==========================
    depth_image = depth_estimator(pil_for_depth)["depth"]
    depth_arr = np.asarray(depth_image).astype(np.uint8)
    if depth_arr.ndim == 3:
        depth_arr = depth_arr[..., 0]

    H, W = depth_arr.shape
    cx, cy = W // 2, H // 2
    half = ball_size // 2
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    # depth control image (3채널 변환)
    depth_mask_circle = get_circle_mask(ball_size).numpy()
    depth_arr_copy = depth_arr.copy()
    depth_arr_copy[y0:y1, x0:x1] = (
        depth_arr_copy[y0:y1, x0:x1] * (1 - depth_mask_circle) + (depth_mask_circle * 255)
    ).astype(np.uint8)

    depth_mask = Image.fromarray(depth_arr_copy).convert("RGB")

    # inpaint mask
    mask_image = np.zeros_like(depth_arr_copy)
    inpaint_mask = get_circle_mask(ball_size + dilate_pixel * 2).numpy()
    y0_d, y1_d = y0 - dilate_pixel, y1 + dilate_pixel
    x0_d, x1_d = x0 - dilate_pixel, x1 + dilate_pixel
    y0_d = max(0, y0_d); x0_d = max(0, x0_d)
    y1_d = min(H, y1_d); x1_d = min(W, x1_d)
    mask_region = inpaint_mask[0:y1_d-y0_d, 0:x1_d-x0_d]
    mask_image[y0_d:y1_d, x0_d:x1_d] = (mask_region * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_image).convert("L")  # Inpaint mask는 1채널 흑백

    # ==========================
    # (3) Prompt 인코딩
    # ==========================
    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(prompt)
    prompt_embeds_black, _, pooled_prompt_embeds_black, _ = pipe.encode_prompt(prompt_black)

    ratio = (ev - lowest_ev) / (0 - lowest_ev)
    prompt_embeds_interp = prompt_embeds * ratio + prompt_embeds_black * (1 - ratio)
    pooled_prompt_embeds_interp = pooled_prompt_embeds * ratio + pooled_prompt_embeds_black * (1 - ratio)

    # LoRA 적용
    apply_lora(pipe, "turbo", lora_scale=turbo_lora_scale)
    is_exposure_lora_loaded = {"v": False}

    def callback(pipeline, i, t, callback_kwargs):
        try:
            tt = int(t)
        except Exception:
            tt = int(round(float(t)))
        if not is_exposure_lora_loaded["v"] and tt <= switch_lora_timestep:
            apply_lora(pipeline, "exposure", lora_scale=exposure_lora_scale)
            is_exposure_lora_loaded["v"] = True
        return callback_kwargs

    generator = torch.Generator(device=pipe.unet.device).manual_seed(seed)

    # ==========================
    # (4) Pipeline 호출 (PIL 입력)
    # ==========================
    out = pipe(
        prompt_embeds=prompt_embeds_interp,
        pooled_prompt_embeds=pooled_prompt_embeds_interp,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        image=init_image,        # PIL
        mask_image=mask_image,   # PIL (L)
        control_image=depth_mask, # PIL (RGB)
        controlnet_conditioning_scale=condition_scale,
        callback_on_step_end=callback,
        generator=generator,
    )

    out_img = out["images"][0]  # PIL.Image 반환

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out_img.save(save_path)
        return save_path
    else:
        return out_img


#=================================#


def pil_square_image(image, desired_size = (512,512), interpolation=Image.LANCZOS):
    """
    Make top-bottom border
    """
    if image.size == desired_size:
        return image

    scale_factor = min(desired_size[0] / image.width, desired_size[1] / image.height)

    resized_image = image.resize((int(image.width * scale_factor), int(image.height * scale_factor)), interpolation)

    new_image = Image.new("RGB", desired_size, color=(0, 0, 0))

    new_image.paste(resized_image, ((desired_size[0] - resized_image.width) // 2, (desired_size[1] - resized_image.height) // 2))

    return new_image

def get_circle_mask(size=256):
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(1, -1, size)
    y, x = torch.meshgrid(y, x)
    z = (1 - x**2 - y**2)
    mask = z >= 0
    return mask

# function to apply LoRA weights
def apply_lora(pipe, adapter_name, lora_scale):
    pipe.unfuse_lora()     # unload previous lora weights (if any)
    pipe.set_adapters(adapter_name)
    pipe.fuse_lora(lora_scale=lora_scale)

