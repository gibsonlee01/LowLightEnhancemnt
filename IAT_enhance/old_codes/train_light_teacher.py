import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from torchvision.models import vgg16
from data_loaders.lol import lowlight_loader, lowlight_loader_sh
from model.IAT_main import IAT
from IQA_pytorch import SSIM
from utils import PSNR, validation, LossNetwork
from torch.utils.tensorboard import SummaryWriter
from model.light_net import LightEstimationNet


# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default="0")
parser.add_argument('--img_path', type=str, default="/content/drive/MyDrive/LOL-v2/Real_captured/Train/Low/")
parser.add_argument('--img_val_path', type=str, default="/content/drive/MyDrive/LOL-v2/Real_captured/Test/Low/")
parser.add_argument('--normalize', action="store_false")
parser.add_argument('--model_type', type=str, default='s')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--pretrain_dir', type=str, default="/content/drive/MyDrive/IAT_test/IAT_enhance/best_Epoch_lol.pth")
parser.add_argument('--lightnet_dir', type=str, default="/content/drive/MyDrive/IAT_test/IAT_enhance/workdirs/lightnet_pretrain/lightnet_best.pth")
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--display_iter', type=int, default=10)
parser.add_argument('--snapshots_folder', type=str, default="workdirs/IAT_joint")
parser.add_argument('--warmup_epochs', type=int, default=5, help="Epochs without env loss")
parser.add_argument('--lambda_env_start', type=float, default=0.001, help="Starting env loss weight")
parser.add_argument('--lambda_env_end', type=float, default=0.3, help="Final env loss weight")
parser.add_argument('--check_lightnet', action='store_true', help="Validate LightNet quality on GT")

config = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
os.makedirs(config.snapshots_folder, exist_ok=True)

# -----------------------------
# Tensorboard
# -----------------------------
writer = SummaryWriter(log_dir=os.path.join(config.snapshots_folder, "tensorboard_logs"))

# -----------------------------
# Model Setting
# -----------------------------
model = IAT(type=config.model_type).cuda()
if config.pretrain_dir is not None:
    model.load_state_dict(torch.load(config.pretrain_dir))
    print(f"✅ Loaded IAT pretrain from {config.pretrain_dir}")

# LightNet 불러오기 (freeze)
light_net = LightEstimationNet().cuda()
light_net.load_state_dict(torch.load(config.lightnet_dir))
light_net.eval()
for p in light_net.parameters():
    p.requires_grad = False
print(f"✅ Loaded frozen LightNet from {config.lightnet_dir}")

# -----------------------------
# Data Setting
# -----------------------------
train_dataset = lowlight_loader_sh(
    images_path=config.img_path,
    normalize=config.normalize,
    envmap_root=config.img_path.replace('Low','Env_map')
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True,
    num_workers=8, pin_memory=True
)
val_dataset = lowlight_loader(
    images_path=config.img_val_path, mode='test', normalize=config.normalize
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

# -----------------------------
# Loss & Optimizer
# -----------------------------
vgg_model = vgg16(pretrained=True).features[:16].cuda()
for p in vgg_model.parameters():
    p.requires_grad = False
loss_network = LossNetwork(vgg_model).eval()

L1_loss = nn.L1Loss()

optimizer = torch.optim.Adam(
    model.parameters(), lr=config.lr, weight_decay=config.weight_decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

# Metrics
ssim = SSIM()
psnr = PSNR()
psnr_high = 0

# -----------------------------
# LightNet Quality Check (Optional)
# -----------------------------
if config.check_lightnet:
    print("\n🔍 Checking LightNet quality on GT images...")
    light_net_errors = []
    with torch.no_grad():
        for i, imgs in enumerate(train_loader):
            if i >= 10:  # Check first 10 batches
                break
            _, high_img, gt_sh = imgs[0].cuda(), imgs[1].cuda(), imgs[2].cuda()
            pred_sh_gt = light_net(high_img)
            error = L1_loss(pred_sh_gt, gt_sh).item()
            light_net_errors.append(error)
    avg_error = sum(light_net_errors) / len(light_net_errors)
    print(f"📊 LightNet average error on GT: {avg_error:.4f}")
    if avg_error > 0.1:
        print("⚠️  Warning: LightNet prediction error is high. Consider retraining LightNet.")
    print()

# -----------------------------
# Training Loop
# -----------------------------
print("######## Start Joint Training (IAT + Frozen LightNet) ########")
print(f"Warmup epochs: {config.warmup_epochs}")
print(f"Env loss weight: {config.lambda_env_start} → {config.lambda_env_end}\n")

for epoch in range(config.num_epochs):
    model.train()
    
    # Progressive lambda_env scheduling
    if epoch < config.warmup_epochs:
        lambda_env = 0.0  # No env loss during warmup
    else:
        # Linear increase from lambda_env_start to lambda_env_end
        progress = (epoch - config.warmup_epochs) / (config.num_epochs - config.warmup_epochs)
        lambda_env = config.lambda_env_start + progress * (config.lambda_env_end - config.lambda_env_start)
    
    epoch_loss = 0
    epoch_loss_recon = 0
    epoch_loss_vgg = 0
    epoch_loss_env = 0
    
    for iteration, imgs in enumerate(train_loader):
        low_img, high_img, gt_sh = imgs[0].cuda(), imgs[1].cuda(), imgs[2].cuda()

        optimizer.zero_grad()
        mul, add, enhance_img = model(low_img)

        # Frozen LightNet → SH 추정
        pred_sh = light_net(enhance_img)

        # === Loss Computation ===
        # 1. Reconstruction Loss (Pixel-level accuracy)
        loss_recon = F.smooth_l1_loss(enhance_img, high_img)
        
        # 2. Perceptual Loss (Texture and detail preservation)
        loss_vgg = loss_network(enhance_img, high_img)
        
        # 3. Environment Loss (Global lighting consistency)
        loss_env = L1_loss(pred_sh, gt_sh)
        
        # Combined Loss
        loss = loss_recon + lambda_env * loss_env

        loss.backward()
        
        # Gradient norm check (optional debugging)
        if iteration == 0:
            total_grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            print(f"[Epoch {epoch}] Gradient norm: {total_grad_norm:.4f}")
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses
        epoch_loss += loss.item()
        epoch_loss_recon += loss_recon.item()
        epoch_loss_vgg += loss_vgg.item()
        epoch_loss_env += loss_env.item()

        # Display progress
        if (iteration + 1) % config.display_iter == 0:
            print(f"[Epoch {epoch}] Iter {iteration+1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Recon: {loss_recon.item():.4f} | "
                  f"VGG: {loss_vgg.item():.4f} | "
                  f"Env: {loss_env.item():.4f} (λ={lambda_env:.3f})")
            
            global_step = epoch * len(train_loader) + iteration
            writer.add_scalar("Loss/recon", loss_recon.item(), global_step)
            writer.add_scalar("Loss/vgg", loss_vgg.item(), global_step)
            writer.add_scalar("Loss/env", loss_env.item(), global_step)
            writer.add_scalar("Loss/total", loss.item(), global_step)
            writer.add_scalar("Hyperparams/lambda_env", lambda_env, global_step)

    # Epoch-level logging
    avg_loss = epoch_loss / len(train_loader)
    avg_loss_recon = epoch_loss_recon / len(train_loader)
    avg_loss_vgg = epoch_loss_vgg / len(train_loader)
    avg_loss_env = epoch_loss_env / len(train_loader)
    
    print(f"\n[Epoch {epoch} Summary]")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  - Recon: {avg_loss_recon:.4f}")
    print(f"  - VGG: {avg_loss_vgg:.4f}")
    print(f"  - Env: {avg_loss_env:.4f}")
    
    writer.add_scalar("Epoch/loss_total", avg_loss, epoch)
    writer.add_scalar("Epoch/loss_recon", avg_loss_recon, epoch)
    writer.add_scalar("Epoch/loss_vgg", avg_loss_vgg, epoch)
    writer.add_scalar("Epoch/loss_env", avg_loss_env, epoch)

    # Validation
    print("  Running validation...")
    model.eval()
    SSIM_mean, PSNR_mean = validation(model, val_loader)
    writer.add_scalar("Val/SSIM", SSIM_mean, epoch)
    writer.add_scalar("Val/PSNR", PSNR_mean, epoch)
    
    print(f"  Val SSIM: {SSIM_mean:.4f}, PSNR: {PSNR_mean:.4f}")
    
    with open(config.snapshots_folder + '/joint_training_log.txt', 'a+') as f:
        f.write(f"Epoch {epoch}: SSIM={SSIM_mean:.4f}, PSNR={PSNR_mean:.4f}, "
                f"Loss={avg_loss:.4f}, lambda_env={lambda_env:.3f}\n")

    # Save best model
    if PSNR_mean > psnr_high:
        psnr_high = PSNR_mean
        torch.save(model.state_dict(), os.path.join(config.snapshots_folder, "IAT_best.pth"))
        print(f"  ✅ Saved new best model | PSNR={PSNR_mean:.4f}\n")
    else:
        print()

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), 
                   os.path.join(config.snapshots_folder, f"IAT_epoch_{epoch}.pth"))

    scheduler.step()

print(f"\n{'='*60}")
print(f"Training completed!")
print(f"Best PSNR: {psnr_high:.4f}")
print(f"Model saved to: {config.snapshots_folder}")
print(f"{'='*60}")

writer.close()