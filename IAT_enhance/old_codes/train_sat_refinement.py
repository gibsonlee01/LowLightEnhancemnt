import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import numpy as np
from tqdm import tqdm
from torchvision.models import vgg16
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from data_loaders.lol import lowlight_loader
from model.IAT_sat import IAT_SatRefinement
from utils import ( PSNR, adjust_learning_rate, LossNetwork )
from model.illumination_utils import (rgb_to_hsv, estimate_illumination)
from IQA_pytorch import SSIM

# ==================== Arguments ====================
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--img_path', type=str, 
                    default="/content/drive/MyDrive/LOL-v2/Real_captured/Train/Low/")
parser.add_argument('--img_val_path', type=str, 
                    default="/content/drive/MyDrive/LOL-v2/Real_captured/Test/Low/")
parser.add_argument("--normalize", action="store_false", 
                    help="Default Normalize in LOL training.")

# Experiment ID
parser.add_argument('--exp_id', type=str, required=True,
                    help='Experiment ID for organizing results (e.g., exp1, conservative, etc.)')

# Model settings
parser.add_argument('--model_type', type=str, default='lol')
parser.add_argument('--pretrained_iat', type=str,
                    help='Path to pretrained IAT weights', 
                    default="/content/drive/MyDrive/IAT_test/IAT_enhance/best_Epoch_lol.pth")
parser.add_argument('--resume_sat', type=str, default=None,
                    help='Path to resume saturation module training')
parser.add_argument('--boost_factor', type=float, default=0.3,
                    help='Saturation boost factor')

# Training settings
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--display_iter', type=int, default=10)
parser.add_argument('--grad_clip', type=float, default=1.0,
                    help='Gradient clipping max norm')

# Loss weights
parser.add_argument('--w_recon', type=float, default=1.0)
parser.add_argument('--w_sat', type=float, default=0.5)
parser.add_argument('--w_weighted_sat', type=float, default=0.5)
parser.add_argument('--w_percep', type=float, default=0.3)
parser.add_argument('--w_over_sat', type=float, default=0.1)

# Save settings
parser.add_argument('--workdir', type=str, default="workdirs",
                    help='Base working directory')
parser.add_argument('--log_interval', type=int, default=50,
                    help='Tensorboard logging interval for scalars')

config = parser.parse_args()

# Create experiment-specific folder
config.snapshots_folder = os.path.join(config.workdir, config.exp_id)
if not os.path.exists(config.snapshots_folder):
    os.makedirs(config.snapshots_folder)

# Save config to file
config_str = "\n".join([f"{k}: {v}" for k, v in vars(config).items()])
with open(os.path.join(config.snapshots_folder, 'config.txt'), 'w') as f:
    f.write(config_str)

print("="*50)
print(f"Experiment ID: {config.exp_id}")
print(f"Save Directory: {config.snapshots_folder}")
print("="*50)
print(config_str)
print("="*50)

os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

# Tensorboard writer
writer = SummaryWriter(log_dir=os.path.join(config.snapshots_folder, 'tensorboard'))

# ==================== Simplified Visualization Helper ====================
def log_validation_images(writer, low_img, base_output, enhanced_img, gt, illum_map, sat_map, epoch):
    """
    Log only essential comparison images for validation
    Only logs the first sample
    """
    # Ensure all images are in [0, 1] range
    low_img = torch.clamp(low_img, 0, 1)
    base_output = torch.clamp(base_output, 0, 1)
    enhanced_img = torch.clamp(enhanced_img, 0, 1)
    gt = torch.clamp(gt, 0, 1)
    
    # 1. Main comparison: GT vs Base vs Enhanced
    comparison_grid = vutils.make_grid(
        torch.cat([gt, base_output, enhanced_img], dim=0),
        nrow=3,
        normalize=False,
        padding=10,
        pad_value=1.0
    )
    writer.add_image('val/comparison_GT_Base_Enhanced', comparison_grid, epoch)
    
    # 2. Illumination and Saturation maps
    illum_vis = illum_map.repeat(1, 3, 1, 1)
    sat_map_vis = sat_map.repeat(1, 3, 1, 1)
    
    maps_grid = vutils.make_grid(
        torch.cat([illum_vis, sat_map_vis], dim=0),
        nrow=2,
        normalize=False,
        padding=10,
        pad_value=1.0
    )
    writer.add_image('val/maps_Illum_SatMap', maps_grid, epoch)

# ==================== Loss Function ====================
class SaturationRefinementLoss(nn.Module):
    def __init__(self, vgg_model, weights):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.loss_network = LossNetwork(vgg_model)
        self.loss_network.eval()
        
        self.w_recon = weights['recon']
        self.w_sat = weights['sat']
        self.w_weighted_sat = weights['weighted_sat']
        self.w_percep = weights['percep']
        self.w_over_sat = weights['over_sat']
    
    def forward(self, pred, gt, illum_map):
        loss_recon = self.l1(pred, gt)
        
        sat_pred = rgb_to_hsv(pred)[:, 1:2]
        sat_gt = rgb_to_hsv(gt)[:, 1:2]
        loss_sat = self.l1(sat_pred, sat_gt)
        
        illum_weight = 1.0 - illum_map
        weighted_error = illum_weight * torch.abs(sat_pred - sat_gt)
        loss_weighted_sat = weighted_error.mean()
        
        loss_percep = self.loss_network(pred, gt)
        
        over_sat_mask = (sat_pred > 0.95).float()
        loss_over_sat = over_sat_mask.mean()
        
        total_loss = (
            self.w_recon * loss_recon +
            self.w_sat * loss_sat +
            self.w_weighted_sat * loss_weighted_sat +
            self.w_percep * loss_percep +
            self.w_over_sat * loss_over_sat
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'recon': loss_recon.item(),
            'sat': loss_sat.item(),
            'weighted_sat': loss_weighted_sat.item(),
            'percep': loss_percep.item(),
            'over_sat': loss_over_sat.item()
        }
        
        return total_loss, loss_dict

# ==================== Validation Function ====================
def validation(model, val_loader, writer, epoch):
    """Validation with PSNR, SSIM and minimal visualization"""
    model.eval()
    
    psnr_list = []
    ssim_list = []
    sat_diff_list = []
    
    ssim_fn = SSIM()
    psnr_fn = PSNR()
    
    first_sample_saved = False
    
    with torch.no_grad():
        for idx, imgs in enumerate(tqdm(val_loader, desc='Validation')):
            low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
            
            enhanced_img, intermediate = model(low_img, return_intermediate=True)
            
            for b in range(low_img.size(0)):
                psnr_val = psnr_fn(enhanced_img[b:b+1], high_img[b:b+1])
                ssim_val = ssim_fn(enhanced_img[b:b+1], high_img[b:b+1], as_loss=False)
                
                psnr_list.append(psnr_val.item())
                ssim_list.append(ssim_val.item())
                
                sat_pred = rgb_to_hsv(enhanced_img[b:b+1])[:, 1].mean()
                sat_gt = rgb_to_hsv(high_img[b:b+1])[:, 1].mean()
                sat_diff_list.append(torch.abs(sat_pred - sat_gt).item())
            
            if not first_sample_saved:
                log_validation_images(
                    writer,
                    low_img=low_img[0:1],
                    base_output=intermediate['base_output'][0:1],
                    enhanced_img=enhanced_img[0:1],
                    gt=high_img[0:1],
                    illum_map=intermediate['illum_map'][0:1],
                    sat_map=intermediate['sat_map'][0:1],
                    epoch=epoch
                )
                first_sample_saved = True
    
    psnr_mean = np.mean(psnr_list)
    ssim_mean = np.mean(ssim_list)
    sat_diff_mean = np.mean(sat_diff_list)
    
    writer.add_scalar('val/PSNR', psnr_mean, epoch)
    writer.add_scalar('val/SSIM', ssim_mean, epoch)
    writer.add_scalar('val/Sat_Diff', sat_diff_mean, epoch)
    
    print(f'Validation - PSNR: {psnr_mean:.4f}, SSIM: {ssim_mean:.4f}, Sat Diff: {sat_diff_mean:.4f}')
    
    return psnr_mean, ssim_mean, sat_diff_mean

# ==================== Main Training ====================
def main():
    # Model Setup
    print('========== Loading Model ==========')
    model = IAT_SatRefinement.from_pretrained(
        pretrained_path=config.pretrained_iat,
        type=config.model_type,
        boost_factor=config.boost_factor
    ).cuda()
    
    if config.resume_sat is not None:
        model.load_sat_module(config.resume_sat)
        print(f'Resumed from: {config.resume_sat}')
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Frozen parameters: {total_params - trainable_params:,}')
    
    writer.add_text('model/architecture', str(model.sat_module), 0)
    writer.add_text('model/parameters', 
                   f'Total: {total_params:,}, Trainable: {trainable_params:,}', 0)
    
    # Data Loading
    print('========== Loading Data ==========')
    train_dataset = lowlight_loader(
        images_path=config.img_path, 
        normalize=config.normalize
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True
    )
    
    val_dataset = lowlight_loader(
        images_path=config.img_val_path, 
        mode='test', 
        normalize=config.normalize
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    # Loss & Optimizer Setup
    print('========== Setting up Training ==========')
    vgg_model = vgg16(pretrained=True).features[:16].cuda()
    for param in vgg_model.parameters():
        param.requires_grad = False
    
    loss_weights = {
        'recon': config.w_recon,
        'sat': config.w_sat,
        'weighted_sat': config.w_weighted_sat,
        'percep': config.w_percep,
        'over_sat': config.w_over_sat
    }
    
    writer.add_text('hyperparameters/loss_weights', str(loss_weights), 0)
    writer.add_text('hyperparameters/optimizer', 
                   f'Adam, lr={config.lr}, weight_decay={config.weight_decay}', 0)
    writer.add_text('hyperparameters/grad_clip', f'{config.grad_clip}', 0)
    writer.add_text('hyperparameters/boost_factor', f'{config.boost_factor}', 0)
    
    criterion = SaturationRefinementLoss(vgg_model, loss_weights)
    
    optimizer = torch.optim.Adam(
        model.sat_module.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.num_epochs
    )
    
    # Training
    print('========== Start Training ==========')
    best_psnr = 0
    best_ssim = 0
    best_sat_diff = float('inf')
    
    global_step = 0
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = {
            'total': [], 'recon': [], 'sat': [], 
            'weighted_sat': [], 'percep': [], 'over_sat': []
        }
        grad_norms = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        for iteration, imgs in enumerate(pbar):
            low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
            
            optimizer.zero_grad()
            
            # Forward
            enhanced_img, intermediate = model(low_img, return_intermediate=True)
            illum_map = intermediate['illum_map']
            
            # Loss
            loss, loss_dict = criterion(enhanced_img, high_img, illum_map)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.sat_module.parameters(), 
                max_norm=config.grad_clip
            )
            grad_norms.append(grad_norm.item())
            
            optimizer.step()
            
            # Record losses
            for key in loss_dict:
                epoch_losses[key].append(loss_dict[key])
            
            # Tensorboard logging - Scalars
            if global_step % config.log_interval == 0:
                for key, val in loss_dict.items():
                    writer.add_scalar(f'train/loss_{key}', val, global_step)
                
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('train/learning_rate', current_lr, global_step)
                writer.add_scalar('train/grad_norm', grad_norm.item(), global_step)
                
                # Saturation map statistics
                sat_map_mean = intermediate['sat_map'].mean().item()
                sat_map_std = intermediate['sat_map'].std().item()
                writer.add_scalar('train/sat_map_mean', sat_map_mean, global_step)
                writer.add_scalar('train/sat_map_std', sat_map_std, global_step)
            
            # Display
            if (iteration + 1) % config.display_iter == 0:
                pbar.set_postfix(loss_dict)
            
            global_step += 1
        
        # Epoch summary
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        avg_grad_norm = np.mean(grad_norms)
        
        print(f"\nEpoch {epoch+1} Summary:")
        for key, val in avg_losses.items():
            print(f"  {key}: {val:.4f}")
            writer.add_scalar(f'epoch/loss_{key}', val, epoch)
        print(f"  avg_grad_norm: {avg_grad_norm:.4f}")
        writer.add_scalar('epoch/avg_grad_norm', avg_grad_norm, epoch)
        
        # Validation
        print('Running validation...')
        psnr_mean, ssim_mean, sat_diff_mean = validation(model, val_loader, writer, epoch)
        
        # Log to file
        with open(os.path.join(config.snapshots_folder, 'log.txt'), 'a+') as f:
            f.write(f"Epoch {epoch+1}: ")
            f.write(f"PSNR={psnr_mean:.4f}, SSIM={ssim_mean:.4f}, ")
            f.write(f"SatDiff={sat_diff_mean:.4f}, Loss={avg_losses['total']:.4f}, ")
            f.write(f"GradNorm={avg_grad_norm:.4f}\n")
        
        # Save best models
        if psnr_mean > best_psnr:
            best_psnr = psnr_mean
            model.save_sat_module(
                os.path.join(config.snapshots_folder, 'best_psnr.pth')
            )
            print(f'Saved best PSNR model: {best_psnr:.4f}')
            writer.add_text('checkpoints/best_psnr', 
                          f'Epoch {epoch+1}, PSNR={best_psnr:.4f}', epoch)
        
        if ssim_mean > best_ssim:
            best_ssim = ssim_mean
            model.save_sat_module(
                os.path.join(config.snapshots_folder, 'best_ssim.pth')
            )
            print(f'Saved best SSIM model: {best_ssim:.4f}')
            writer.add_text('checkpoints/best_ssim', 
                          f'Epoch {epoch+1}, SSIM={best_ssim:.4f}', epoch)
        
        if sat_diff_mean < best_sat_diff:
            best_sat_diff = sat_diff_mean
            model.save_sat_module(
                os.path.join(config.snapshots_folder, 'best_sat.pth')
            )
            print(f'Saved best Sat Diff model: {best_sat_diff:.4f}')
            writer.add_text('checkpoints/best_sat', 
                          f'Epoch {epoch+1}, SatDiff={best_sat_diff:.4f}', epoch)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.save_sat_module(
                os.path.join(config.snapshots_folder, f'epoch_{epoch+1}.pth')
            )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning rate: {current_lr:.6f}\n')
    
    print('========== Training Finished ==========')
    print(f'Best PSNR: {best_psnr:.4f}')
    print(f'Best SSIM: {best_ssim:.4f}')
    print(f'Best Sat Diff: {best_sat_diff:.4f}')
    
    writer.close()

if __name__ == '__main__':
    main()