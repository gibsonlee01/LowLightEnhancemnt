import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import math
from torchvision.models import vgg16

from data_loaders.lol import lowlight_loader
from model.IAT_main import IAT, Local_pred_S
from IQA_pytorch import SSIM
from utils import PSNR, validation, LossNetwork
from torch.utils.tensorboard import SummaryWriter


# ==================== Illumination Map Extraction ====================
class IlluminationExtractor:
    """Retinex 이론 기반 조명맵 추출"""
    
    @staticmethod
    def estimate_illumination(image, method='max'):
        """
        조명맵 추출
        Args:
            image: [B, 3, H, W], range [0, 1]
            method: 'max', 'gray', 'avg'
        Returns:
            illum_map: [B, 1, H, W]
        """
        if method == 'max':
            # Max channel (가장 밝은 채널)
            illum = torch.max(image, dim=1, keepdim=True)[0]
        elif method == 'gray':
            # Luminance (Y = 0.299R + 0.587G + 0.114B)
            weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(image.device)
            illum = (image * weights).sum(dim=1, keepdim=True)
        elif method == 'avg':
            # Average channel
            illum = image.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return illum
    
    @staticmethod
    def smooth_illumination(illum, kernel_size=15):
        """
        조명맵 스무딩 (Guided filter 대신 Gaussian blur 사용)
        """
        # Gaussian kernel 생성
        sigma = kernel_size / 6.0
        kernel = torch.zeros(1, 1, kernel_size, kernel_size).to(illum.device)
        center = kernel_size // 2
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[0, 0, i, j] = math.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        kernel = kernel / kernel.sum()
        
        # Apply Gaussian smoothing
        padding = kernel_size // 2
        smoothed = F.conv2d(illum, kernel, padding=padding)
        
        return smoothed
    
    @staticmethod
    def compute_importance_map(img_low, method='max', smooth=True):
        """
        조명 기반 importance map 계산
        어두운 영역 = 높은 가중치
        """
        # 조명맵 추출
        illum = IlluminationExtractor.estimate_illumination(img_low, method)
        
        # 스무딩 (optional)
        if smooth:
            illum = IlluminationExtractor.smooth_illumination(illum)
        
        # Importance: 어두울수록 높은 가중치
        # W = exp(-k * I), where I is illumination
        k = 3.0  # sensitivity parameter
        importance = torch.exp(-k * illum)
        
        # Normalize to [0.5, 2.0] for stability
        importance = importance / (importance.mean(dim=[2, 3], keepdim=True) + 1e-6)
        importance = torch.clamp(importance, 0.5, 2.0)
        
        return importance, illum


# ==================== Student Model ====================
class LightweightGlobal(nn.Module):
    """Teacher의 Global_pred를 대체하는 경량 네트워크"""
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Feature extraction
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Gamma prediction
        self.gamma_fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Color matrix prediction (3x3)
        self.color_fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 9)
        )
        
        # Initialize color prediction to identity
        nn.init.zeros_(self.color_fc[-1].weight)
        nn.init.zeros_(self.color_fc[-1].bias)
        
    def forward(self, x):
        feat = self.feature(x)  # [B, 32, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [B, 32]
        
        # Gamma: [B, 1] (range 0.5~1.5)
        gamma = 0.5 + 1.0 * self.gamma_fc(feat)
        
        # Color: [B, 3, 3] (residual connection to identity)
        color = self.color_fc(feat).view(-1, 3, 3)
        identity = torch.eye(3).unsqueeze(0).to(x.device)
        color = color * 0.1 + identity
        
        return gamma, color


class IAT_Student(nn.Module):
    """Student model: Local_pred_S + Lightweight Global"""
    def __init__(self, in_dim=3, type='lol'):
        super().__init__()
        self.local_net = Local_pred_S(in_dim=in_dim)
        self.global_net = LightweightGlobal(in_channels=in_dim)
        
    def apply_color(self, image, ccm):
        """Teacher와 동일한 color transformation"""
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)
    
    def forward(self, img_low):
        # Local enhancement
        mul, add = self.local_net(img_low)
        img_high = img_low.mul(mul).add(add)
        
        # Global refinement
        gamma, color = self.global_net(img_low)
        b = img_high.shape[0]
        img_high = img_high.permute(0, 2, 3, 1)  # [B,C,H,W] → [B,H,W,C]
        
        # Apply color and gamma (same as teacher)
        img_high = torch.stack([
            self.apply_color(img_high[i], color[i])**gamma[i] 
            for i in range(b)
        ], dim=0)
        
        img_high = img_high.permute(0, 3, 1, 2)  # [B,H,W,C] → [B,C,H,W]
        
        return mul, add, img_high


# ==================== Distillation Trainer ====================
class IlluminationDistillationTrainer:
    def __init__(self, config):
        self.config = config
        
        # Teacher (frozen)
        self.teacher = IAT(type=config.model_type).cuda()
        if config.teacher_path:
            state_dict = torch.load(config.teacher_path)
            self.teacher.load_state_dict(state_dict)
            print(f"✅ Loaded teacher from: {config.teacher_path}")
        
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # Student
        self.student = IAT_Student(type=config.model_type).cuda()
        
        # Model size comparison
        teacher_params = sum(p.numel() for p in self.teacher.parameters()) / 1e6
        student_params = sum(p.numel() for p in self.student.parameters()) / 1e6
        print(f"📊 Teacher: {teacher_params:.2f}M params")
        print(f"📊 Student: {student_params:.2f}M params")
        print(f"📊 Compression: {teacher_params/student_params:.2f}x")
        
        # Optimizer & Scheduler
        self.optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        # Loss functions
        self.L1_smooth = F.smooth_l1_loss
        vgg_model = vgg16(pretrained=True).features[:16].cuda()
        for p in vgg_model.parameters():
            p.requires_grad = False
        self.loss_network = LossNetwork(vgg_model).eval()
        
        # Illumination extractor
        self.illum_extractor = IlluminationExtractor()
        
        # Metrics
        self.ssim = SSIM()
        self.psnr = PSNR()
        
        # Tensorboard
        self.writer = SummaryWriter(
            log_dir=os.path.join(config.snapshots_folder, "tensorboard")
        )
    
    def weighted_feature_loss(self, feat_s, feat_t, weight_map):
        """Importance-weighted feature matching"""
        # Resize weight map to match feature size if needed
        if feat_s.shape != weight_map.shape:
            weight_map = F.interpolate(
                weight_map, 
                size=feat_s.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        diff = (feat_s - feat_t) ** 2
        weighted_diff = weight_map * diff
        return weighted_diff.mean()
    
    def compute_loss(self, student_out, teacher_out, gt_img, img_low, epoch):
        """
        Hierarchical distillation loss with illumination-aware weighting
        """
        mul_s, add_s, img_s = student_out
        mul_t, add_t, img_t = teacher_out
        
        losses = {}
        
        # ===== 1. Extract illumination-based importance map =====
        importance_map, illum_map = self.illum_extractor.compute_importance_map(
            img_low, 
            method='max',
            smooth=True
        )
        
        # ===== 2. Task loss (reconstruction) =====
        losses['recon'] = self.L1_smooth(img_s, gt_img)
        losses['percep'] = self.loss_network(img_s, gt_img)
        
        # ===== 3. Illumination-weighted feature distillation =====
        losses['mul_distill'] = self.weighted_feature_loss(mul_s, mul_t, importance_map)
        losses['add_distill'] = self.weighted_feature_loss(add_s, add_t, importance_map)
        
        # ===== 4. Output-level distillation =====
        losses['output_distill'] = self.L1_smooth(img_s, img_t)
        
        # ===== 5. Progressive weighting =====
        warmup = self.config.warmup_epochs
        total = self.config.num_epochs
        
        if epoch < warmup:
            # Warmup: focus on task loss
            alpha_feat = 0.1 * (epoch / warmup)  # 0 → 0.1
            alpha_output = 0.05 * (epoch / warmup)  # 0 → 0.05
        else:
            # Main training: gradually increase distillation
            progress = (epoch - warmup) / (total - warmup)
            alpha_feat = 0.1 + 0.4 * progress  # 0.1 → 0.5
            alpha_output = 0.05 + 0.15 * progress  # 0.05 → 0.2
        
        # ===== 6. Total loss =====
        total_loss = (
            1.0 * losses['recon'] +
            0.04 * losses['percep'] +
            alpha_feat * (losses['mul_distill'] + losses['add_distill']) +
            alpha_output * losses['output_distill']
        )
        
        losses['total'] = total_loss
        losses['alpha_feat'] = alpha_feat
        losses['alpha_output'] = alpha_output
        losses['importance_mean'] = importance_map.mean().item()
        
        return total_loss, losses, importance_map, illum_map
    
    def train_epoch(self, train_loader, epoch):
        self.student.train()
        epoch_losses = {}
        
        for iteration, imgs in enumerate(train_loader):
            low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
            
            self.optimizer.zero_grad()
            
            # Teacher forward (no grad)
            with torch.no_grad():
                mul_t, add_t, img_t = self.teacher(low_img)
                teacher_out = (mul_t, add_t, img_t)
            
            # Student forward
            mul_s, add_s, img_s = self.student(low_img)
            student_out = (mul_s, add_s, img_s)
            
            # Compute loss
            total_loss, losses, importance_map, illum_map = self.compute_loss(
                student_out, teacher_out, high_img, low_img, epoch
            )
            
            # Backward
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), 
                max_norm=5.0
            )
            self.optimizer.step()
            
            # Accumulate losses
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v.item() if isinstance(v, torch.Tensor) else v)
            
            # Logging
            if (iteration + 1) % self.config.display_iter == 0:
                print(f"[Epoch {epoch}] Iter {iteration+1}/{len(train_loader)}")
                print(f"  Loss: {losses['total'].item():.4f} | "
                      f"Recon: {losses['recon'].item():.4f} | "
                      f"Mul: {losses['mul_distill'].item():.4f} | "
                      f"Add: {losses['add_distill'].item():.4f}")
                print(f"  α_feat: {losses['alpha_feat']:.3f} | "
                      f"α_out: {losses['alpha_output']:.3f} | "
                      f"Grad: {grad_norm:.3f}")
                
                # Tensorboard logging
                global_step = epoch * len(train_loader) + iteration
                for k, v in losses.items():
                    self.writer.add_scalar(f"Train/{k}", v.item() if isinstance(v, torch.Tensor) else v, global_step)
            
            # Visualization (first iteration of every 10 epochs)
            if iteration == 0 and epoch % 10 == 0:
                with torch.no_grad():
                    # Clamp to [0, 1] for visualization
                    low_vis = torch.clamp(low_img[0], 0, 1)
                    high_vis = torch.clamp(high_img[0], 0, 1)
                    teacher_vis = torch.clamp(img_t[0], 0, 1)
                    student_vis = torch.clamp(img_s[0], 0, 1)
                    
                    self.writer.add_image('Images/1_input', low_vis, epoch)
                    self.writer.add_image('Images/2_ground_truth', high_vis, epoch)
                    self.writer.add_image('Images/3_teacher', teacher_vis, epoch)
                    self.writer.add_image('Images/4_student', student_vis, epoch)
                    self.writer.add_image('Maps/illumination', illum_map[0], epoch)
                    self.writer.add_image('Maps/importance', importance_map[0], epoch)
        
        # Epoch average
        avg_losses = {k: sum(v)/len(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    def validate(self, val_loader, epoch):
        self.student.eval()
        with torch.no_grad():
            psnr_mean, ssim_mean = validation(self.student, val_loader)
        
        self.writer.add_scalar("Val/PSNR", psnr_mean, epoch)
        self.writer.add_scalar("Val/SSIM", ssim_mean, epoch)
        
        return psnr_mean, ssim_mean
    
    def train(self, train_loader, val_loader):
        best_psnr = 0
        
        print("\n" + "="*70)
        print("  Illumination-Aware Knowledge Distillation")
        print("="*70)
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Warmup epochs: {self.config.warmup_epochs}")
        print("="*70 + "\n")
        
        for epoch in range(self.config.num_epochs):
            # Train
            avg_losses = self.train_epoch(train_loader, epoch)
            
            # Summary
            print(f"\n{'='*70}")
            print(f"Epoch {epoch} Summary:")
            print(f"  Total: {avg_losses['total']:.4f}")
            print(f"  Recon: {avg_losses['recon']:.4f}")
            print(f"  Mul distill: {avg_losses['mul_distill']:.4f}")
            print(f"  Add distill: {avg_losses['add_distill']:.4f}")
            print(f"  Output distill: {avg_losses['output_distill']:.4f}")
            print(f"  Importance mean: {avg_losses['importance_mean']:.3f}")
            
            # Validate
            psnr_mean, ssim_mean = self.validate(val_loader, epoch)
            print(f"  Val PSNR: {psnr_mean:.4f} | Val SSIM: {ssim_mean:.4f}")
            print(f"{'='*70}\n")
            
            # Save log
            with open(os.path.join(self.config.snapshots_folder, 'log.txt'), 'a+') as f:
                f.write(f"Epoch {epoch}: PSNR={psnr_mean:.4f}, SSIM={ssim_mean:.4f}, "
                       f"Loss={avg_losses['total']:.4f}\n")
            
            # Save best
            if psnr_mean > best_psnr:
                best_psnr = psnr_mean
                torch.save(
                    self.student.state_dict(),
                    os.path.join(self.config.snapshots_folder, "student_best.pth")
                )
                print(f"✅ Saved best model | PSNR={psnr_mean:.4f}\n")
            
            # Checkpoint
            if (epoch + 1) % 20 == 0:
                torch.save(
                    self.student.state_dict(),
                    os.path.join(self.config.snapshots_folder, f"student_epoch_{epoch}.pth")
                )
            
            self.scheduler.step()
        
        print(f"\n{'='*70}")
        print(f"Training Completed! Best PSNR: {best_psnr:.4f}")
        print(f"{'='*70}\n")
        
        self.writer.close()


# ==================== Main ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Hardware
    parser.add_argument('--gpu_id', type=str, default='0')
    
    # Data
    parser.add_argument('--img_path', type=str, 
                       default="/content/drive/MyDrive/LOL-v2/Real_captured/Train/Low/")
    parser.add_argument('--img_val_path', type=str, 
                       default="/content/drive/MyDrive/LOL-v2/Real_captured/Test/Low/")
    parser.add_argument("--normalize", action="store_false")
    
    # Model
    parser.add_argument('--model_type', type=str, default='lol')
    parser.add_argument('--teacher_path', type=str, required=True,
                       help="Path to pretrained teacher IAT checkpoint")
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    
    # Logging
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, 
                       default="workdirs/IAT_illumination_distill")
    
    config = parser.parse_args()
    
    print(config)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    os.makedirs(config.snapshots_folder, exist_ok=True)
    
    # Data loaders
    train_dataset = lowlight_loader(images_path=config.img_path, normalize=config.normalize)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=8, pin_memory=True
    )
    
    val_dataset = lowlight_loader(images_path=config.img_val_path, mode='test', 
                                  normalize=config.normalize)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=8, pin_memory=True
    )
    
    # Train
    trainer = IlluminationDistillationTrainer(config)
    trainer.train(train_loader, val_loader)