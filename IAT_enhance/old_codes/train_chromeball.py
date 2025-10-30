import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F

import os
import sys
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models import vgg16

from data_loaders.lol import lowlight_loader, lowlight_loader_env
from model.IAT_main import IAT
from IQA_pytorch import SSIM
from utils import PSNR, adjust_learning_rate, validation, LossNetwork, visualization
from DiffusionLight.chromeball_utils import init_chromeball_pipeline
from DiffusionLight.inference_env_map import make_envmap_from_tensor 
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--img_path', type=str, default="/content/drive/MyDrive/LOL-v2/Real_captured/Train/Low/")
parser.add_argument('--img_val_path', type=str, default="/content/drive/MyDrive/LOL-v2/Real_captured/Test/Low/")
parser.add_argument("--normalize", action="store_false", help="Default Normalize in LOL training.")
parser.add_argument('--model_type', type=str, default='s')

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--pretrain_dir', type=str, default="/content/drive/MyDrive/IAT_test/IAT_enhance/best_Epoch_lol.pth")

parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--display_iter', type=int, default=10)
parser.add_argument('--snapshots_folder', type=str, default="workdirs/snapshots_folder_lol")

config = parser.parse_args()

#========tensorboard setting========#
tensorboard_log_dir = os.path.join(config.snapshots_folder, 'tensorboard_logs')
writer = SummaryWriter(log_dir=tensorboard_log_dir)

print(config)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

if not os.path.exists(config.snapshots_folder):
    os.makedirs(config.snapshots_folder)

# Model Setting
model = IAT(type=config.model_type).cuda()
if config.pretrain_dir is not None:
    model.load_state_dict(torch.load(config.pretrain_dir))

# Data Setting
train_dataset = lowlight_loader_env(images_path=config.img_path, normalize=config.normalize,
                                    envmap_root=config.img_path.replace('Low','Env_map'))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                                           pin_memory=True)
val_dataset = lowlight_loader(images_path=config.img_val_path, mode='test', normalize=config.normalize)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

# Loss & Optimizer Setting & Metric
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.cuda()

for param in vgg_model.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

device = next(model.parameters()).device
print('the device is:', device)

# chrome ball pipeline init 
pipe, depth_estimator = init_chromeball_pipeline(device=device)

L1_loss = nn.L1Loss()
L1_smooth_loss = F.smooth_l1_loss

loss_network = LossNetwork(vgg_model)
loss_network.eval()
lambda_env = 0.2

ssim = SSIM()
psnr = PSNR()
ssim_high = 0
psnr_high = 0

model.train()

print('######## Start IAT Training #########')
for epoch in range(config.num_epochs):
    print('the epoch is:', epoch)
    for iteration, imgs in enumerate(train_loader):
        # 데이터 로드
        low_img, high_img, gt_envmap = imgs[0].cuda(), imgs[1].cuda(), imgs[2].cuda()

        optimizer.zero_grad()
        model.train()
        mul, add, enhance_img = model(low_img)

        # 기존 loss
        loss_recon = L1_smooth_loss(enhance_img, high_img)
        loss_vgg = loss_network(enhance_img, high_img)

        # ==========================
        # Envmap loss (10 epoch마다만 계산)
        # ==========================
        loss_env = torch.tensor(0.0, device=device)  # 기본 0 loss

        # if epoch >= 150:   # 매 10 epoch마다 envmap loss 활성화
        pred_envmaps = []
        with torch.no_grad():
            for b in range(enhance_img.shape[0]):
                envmap, chromeball_img = make_envmap_from_tensor(
                    tensor_img=enhance_img[b].detach().cpu(),
                    ev=0.0,
                    seed=200,
                    pipe=pipe,
                    depth_estimator=depth_estimator,
                    BALL_SIZE=256,
                    MSAA_SCALE=4,
                    ENVMAP_SIZE=256,
                    device=device,
                )
                pred_envmaps.append(envmap)

                if epoch == 0 and iteration == 0 and b == 0:
                    print("🔍 [DEBUG] pred_envmap shape:", envmap.shape)
                    print("🔍 [DEBUG] gt_envmap shape:", gt_envmap[b].shape)

                # epoch마다 크롬볼 저장 (첫 iter/batch만)
                if iteration == 0 and b == 0:
                    save_dir = os.path.join(config.snapshots_folder, "chromeball")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f'epoch{epoch}_iter{iteration}_b{b}_chromeball.png')
                    chromeball_img.save(save_path)

            pred_envmaps = torch.stack(pred_envmaps, dim=0)
            loss_env = L1_loss(pred_envmaps, gt_envmap)

        # 최종 loss
        loss = loss_recon + 0.04 * loss_vgg + lambda_env * loss_env

        loss.backward()
        optimizer.step()

        if ((iteration + 1) % config.display_iter) == 0:
            print(f"Iter {iteration+1} | Loss: {loss.item():.4f} | Env Loss: {loss_env.item():.4f}")
            
            global_step = epoch * len(train_loader) + iteration
            writer.add_scalar("Loss/total", loss.item(), global_step)
            writer.add_scalar("Loss/recon", loss_recon.item(), global_step)
            writer.add_scalar("Loss/vgg", loss_vgg.item(), global_step)
            writer.add_scalar("Loss/env", loss_env.item(), global_step)

    # Validation
    model.eval()
    SSIM_mean, PSNR_mean = validation(model, val_loader)
    writer.add_scalar("Val/SSIM", SSIM_mean, epoch)
    writer.add_scalar("Val/PSNR", PSNR_mean, epoch)

    with open(config.snapshots_folder + '/chromeball_train_log.txt', 'a+') as f:
        f.write(f"epoch {epoch}: SSIM={SSIM_mean:.4f}, PSNR={PSNR_mean:.4f}\n")
        
    if SSIM_mean > ssim_high:
        ssim_high = SSIM_mean
        print('the highest SSIM value is:', str(ssim_high))
        torch.save(model.state_dict(), os.path.join(config.snapshots_folder, "chromeball_best_Epoch" + '.pth'))
    
    scheduler.step()

    writer.close()
    f.close()
