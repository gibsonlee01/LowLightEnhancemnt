import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import os
import argparse
from data_loaders.lol import lowlight_loader_sh
from torch.utils.tensorboard import SummaryWriter
from model.light_net import LightEstimationNet


# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default="0")
parser.add_argument('--img_path', type=str, default="/content/drive/MyDrive/LOL-v2/Real_captured/Train/Low/")
parser.add_argument('--normalize', action="store_false", help="Default Normalize in LOL training.")
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--display_iter', type=int, default=50)
parser.add_argument('--snapshots_folder', type=str, default="workdirs/lightnet_pretrain")

config = parser.parse_args()

# -----------------------------
# Tensorboard setting
# -----------------------------
tensorboard_log_dir = os.path.join(config.snapshots_folder, 'tensorboard_logs')
os.makedirs(tensorboard_log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=tensorboard_log_dir)

print(config)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

if not os.path.exists(config.snapshots_folder):
    os.makedirs(config.snapshots_folder)

# -----------------------------
# Model Setting
# -----------------------------
light_net = LightEstimationNet().cuda()

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

# -----------------------------
# Loss & Optimizer
# -----------------------------
L1_loss = nn.L1Loss()
optimizer = torch.optim.Adam(
    light_net.parameters(), lr=config.lr, weight_decay=config.weight_decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

device = next(light_net.parameters()).device
print('the device is:', device)

# -----------------------------
# Training Loop
# -----------------------------
print('######## Start LightNet Pretraining #########')
best_loss = float("inf")
best_model_path = os.path.join(config.snapshots_folder, "lightnet_best.pth")

for epoch in range(config.num_epochs):
    light_net.train()
    epoch_loss = 0.0
    for iteration, imgs in enumerate(train_loader):
        low_img, high_img, gt_sh = imgs[0].cuda(), imgs[1].cuda(), imgs[2].cuda()

        # 일반적으로 high_img → SH 사용
        input_img = high_img  

        optimizer.zero_grad()
        pred_sh = light_net(input_img)
        loss_env = L1_loss(pred_sh, gt_sh)
        loss_env.backward()
        optimizer.step()

        epoch_loss += loss_env.item()

        if (iteration + 1) % config.display_iter == 0:
            print(f"[Epoch {epoch}] Iter {iteration+1} | Env Loss: {loss_env.item():.4f}")
            global_step = epoch * len(train_loader) + iteration
            writer.add_scalar("Loss/env", loss_env.item(), global_step)

    scheduler.step()

    # Epoch 평균 loss 기준 best 저장
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch} | Avg Env Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(light_net.state_dict(), best_model_path)
        print(f"✅ New best model saved at epoch {epoch} with loss {avg_loss:.4f}")

writer.close()
