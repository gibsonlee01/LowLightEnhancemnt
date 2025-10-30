import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from torchvision.models import vgg16

from data_loaders.lol import lowlight_loader
from model.IAT_main import IAT
from IQA_pytorch import SSIM
from utils import PSNR, validation, LossNetwork
from torch.utils.tensorboard import SummaryWriter


# ==================== Student Model (기존과 동일) ====================
class LightweightGammaPredictor(nn.Module):
    """
    Global_net(Cross Attention)을 대체하는 경량 네트워크
    Teacher와 동일하게 단일 gamma 값 예측
    """
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Lightweight CNN for single gamma prediction
        self.gamma_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global pooling
            nn.Flatten(),
            nn.Linear(32, 1),  # Single gamma value
            nn.Sigmoid()  # Gamma in [0, 1]
        )
        
        # Lightweight color matrix prediction
        self.color_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 9),  # 3x3 color matrix
        )
        
    def forward(self, x):
        # Gamma: [B, 1] - 단일 값
        gamma = self.gamma_net(x)
        
        # Color matrix: [B, 3, 3]
        color = self.color_net(x)
        color = color.view(-1, 3, 3)
        
        # Initialize as identity matrix
        identity = torch.eye(3).unsqueeze(0).to(x.device)
        color = color + identity  # Residual connection
        
        return gamma, color


class IAT_Student(nn.Module):
    """
    Student: Local_net + Lightweight Global predictor
    """
    def __init__(self, in_dim=3, type='lol'):
        super().__init__()
        
        # Local net (동일하게 유지)
        from model.IAT_main import Local_pred_S
        self.local_net = Local_pred_S(in_dim=in_dim)
        
        # Lightweight global predictor (Global_net 대체)
        self.lightweight_global = LightweightGammaPredictor(in_channels=in_dim)
        
    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)
    
    def forward(self, img_low, return_intermediates=False):
        # Local enhancement
        mul, add = self.local_net(img_low)
        img_local = img_low.mul(mul).add(add)
        
        # Lightweight global refinement
        gamma, color = self.lightweight_global(img_low)  # gamma: [B, 1]
        
        b = img_local.shape[0]
        img_high = img_local.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        # Apply color correction and gamma (각 배치별로)
        img_high = torch.stack([
            self.apply_color(img_high[i], color[i]) ** gamma[i]
            for i in range(b)
        ], dim=0)
        
        img_high = img_high.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        if return_intermediates:
            return {
                'mul': mul,
                'add': add,
                'img_local': img_local,
                'gamma': gamma,
                'color': color,
                'img_high': img_high
            }
        
        return mul, add, img_high


# ==================== Illumination Map Utilities ====================
def extract_illumination_map(image):
    """
    Simple illumination map extraction
    Returns illumination map from RGB image
    """
    # Method 1: Max channel (simple but effective)
    illum_max = torch.max(image, dim=1, keepdim=True)[0]
    
    # Method 2: Weighted RGB (considering human perception)
    # weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(image.device)
    # illum_weighted = (image * weights).sum(dim=1, keepdim=True)
    
    return illum_max


def compute_illumination_loss(pred, target):
    """
    Illumination consistency loss
    """
    # Extract illumination maps
    illum_pred = extract_illumination_map(pred)
    illum_target = extract_illumination_map(target)
    
    # L1 loss for illumination matching
    loss_illum = F.l1_loss(illum_pred, illum_target)
    
    # Optional: Multi-scale illumination matching
    loss_multiscale = 0
    for scale in [2, 4]:
        pred_down = F.avg_pool2d(illum_pred, scale)
        target_down = F.avg_pool2d(illum_target, scale)
        loss_multiscale += F.l1_loss(pred_down, target_down) / scale
    
    return loss_illum + 0.3 * loss_multiscale


# ==================== Distillation Trainer with Illumination ====================
class IAT_DistillationTrainer:
    def __init__(self, config):
        self.config = config
        
        # Teacher: Pretrained full IAT (frozen)
        self.teacher = IAT(type=config.model_type, with_global=True).cuda()
        if config.teacher_path is not None:
            self.teacher.load_state_dict(torch.load(config.teacher_path))
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        print(f"✅ Loaded Teacher from {config.teacher_path}")
        
        # Student: Lightweight IAT
        self.student = IAT_Student(type=config.model_type).cuda()
        print(f"📊 Teacher params: {sum(p.numel() for p in self.teacher.parameters())/1e6:.2f}M")
        print(f"📊 Student params: {sum(p.numel() for p in self.student.parameters())/1e6:.2f}M")
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        # Loss functions
        self.L1_loss = nn.L1Loss()
        vgg_model = vgg16(pretrained=True).features[:16].cuda()
        for p in vgg_model.parameters():
            p.requires_grad = False
        self.loss_network = LossNetwork(vgg_model).eval()
        
        # Metrics
        self.ssim = SSIM()
        self.psnr = PSNR()
        
        # Tensorboard
        self.writer = SummaryWriter(
            log_dir=os.path.join(config.snapshots_folder, "tensorboard")
        )
        
    def extract_teacher_knowledge(self, img_low):
        """Teacher의 intermediate outputs 추출"""
        with torch.no_grad():
            # Local outputs
            mul_t, add_t = self.teacher.local_net(img_low)
            img_local_t = img_low.mul(mul_t).add(add_t)
            
            # Global outputs
            gamma_t, color_t = self.teacher.global_net(img_low)
            
            # Final output
            _, _, img_high_t = self.teacher(img_low)
            
        return {
            'mul': mul_t,
            'add': add_t,
            'img_local': img_local_t,
            'gamma': gamma_t,  # [B, 1]
            'color': color_t,  # [B, 3, 3]
            'img_high': img_high_t
        }
    
    def compute_distillation_loss(self, student_out, teacher_out, high_img, epoch):
        """
        Multi-level distillation loss with illumination
        """
        losses = {}
        
        # ========== 기존 Losses (그대로 유지) ==========
        # 1. GT Reconstruction Loss (Primary)
        losses['gt'] = F.smooth_l1_loss(student_out['img_high'], high_img)
        
        # 2. Output-level Distillation
        losses['output'] = self.L1_loss(student_out['img_high'], teacher_out['img_high'])
        
        # 3. Gamma Distillation
        losses['gamma'] = self.L1_loss(student_out['gamma'], teacher_out['gamma'])
        
        # 4. Color Matrix Distillation
        losses['color'] = self.L1_loss(student_out['color'], teacher_out['color'])
        
        # 5. Local Enhancement Consistency
        losses['mul'] = self.L1_loss(student_out['mul'], teacher_out['mul'])
        losses['add'] = self.L1_loss(student_out['add'], teacher_out['add'])
        
        # 6. Intermediate Image Distillation
        losses['img_local'] = self.L1_loss(student_out['img_local'], teacher_out['img_local'])
        
        # 7. Perceptual Loss
        losses['perceptual'] = self.loss_network(student_out['img_high'], high_img)
        
        # 8. Perceptual Distillation
        losses['perceptual_distill'] = self.loss_network(
            student_out['img_high'], teacher_out['img_high']
        )
        
        # ========== NEW: Illumination Loss ==========
        # 9. Illumination consistency with GT
        losses['illum_gt'] = compute_illumination_loss(student_out['img_high'], high_img)
        
        # 10. Illumination distillation from teacher
        losses['illum_teacher'] = compute_illumination_loss(
            student_out['img_high'], teacher_out['img_high']
        )
        
        # ========== Progressive Weighting ==========
        warmup_epochs = self.config.warmup_epochs
        
        if epoch < warmup_epochs:
            alpha_distill = 0.1 * (epoch / warmup_epochs)
            alpha_illum = 0.2 * (epoch / warmup_epochs)  # Illumination weight
        else:
            progress = (epoch - warmup_epochs) / (self.config.num_epochs - warmup_epochs)
            alpha_distill = 0.1 + 0.4 * progress
            alpha_illum = 0.2 + 0.2 * progress  # 0.2 → 0.4
        
        # Combined Loss (illumination loss 추가)
        total_loss = (
            1.0 * losses['gt'] +                      # GT (primary)
            0.04 * losses['perceptual'] +             # Perceptual quality
            alpha_distill * losses['output'] +        # Output mimicking
            alpha_distill * losses['gamma'] +         # Gamma distillation
            alpha_distill * 0.5 * losses['color'] +   # Color distillation
            alpha_distill * 0.3 * losses['img_local'] + # Local consistency
            alpha_distill * 0.2 * losses['mul'] +     # Mul parameter
            alpha_distill * 0.2 * losses['add'] +     # Add parameter
            alpha_distill * 0.1 * losses['perceptual_distill'] + # Perceptual alignment
            alpha_illum * losses['illum_gt'] +        # NEW: Illumination with GT
            alpha_illum * 0.5 * losses['illum_teacher']  # NEW: Illumination distillation
        )
        
        losses['total'] = total_loss
        losses['alpha_distill'] = alpha_distill
        losses['alpha_illum'] = alpha_illum
        
        return total_loss, losses
    
    def train_epoch(self, train_loader, epoch):
        self.student.train()
        epoch_losses = {}
        
        for iteration, imgs in enumerate(train_loader):
            low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
            
            # Forward
            self.optimizer.zero_grad()
            
            # Teacher knowledge extraction
            teacher_out = self.extract_teacher_knowledge(low_img)
            
            # Student prediction
            student_out = self.student(low_img, return_intermediates=True)
            
            # Compute loss
            total_loss, losses = self.compute_distillation_loss(
                student_out, teacher_out, high_img, epoch
            )
            
            # Backward
            total_loss.backward()
            
            # Gradient clipping (stability)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # Logging
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                if isinstance(v, torch.Tensor):
                    epoch_losses[k].append(v.item())
                else:
                    epoch_losses[k].append(v)
            
            if (iteration + 1) % self.config.display_iter == 0:
                print(f"[Epoch {epoch}] Iter {iteration+1}/{len(train_loader)}")
                print(f"  Total: {losses['total'].item():.4f} | "
                      f"GT: {losses['gt'].item():.4f} | "
                      f"Illum_GT: {losses['illum_gt'].item():.4f} | "  # NEW
                      f"Gamma: {losses['gamma'].item():.4f} | "
                      f"α_dist: {losses['alpha_distill']:.3f} | "
                      f"α_illum: {losses['alpha_illum']:.3f}")  # NEW
                
                global_step = epoch * len(train_loader) + iteration
                for k, v in losses.items():
                    if isinstance(v, torch.Tensor):
                        self.writer.add_scalar(f"Train/{k}", v.item(), global_step)
                    else:
                        self.writer.add_scalar(f"Train/{k}", v, global_step)
        
        # Epoch summary
        avg_losses = {k: sum(v)/len(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    def validate(self, val_loader, epoch):
        self.student.eval()
        
        PSNR_mean, SSIM_mean = validation(self.student, val_loader)
        
        self.writer.add_scalar("Val/PSNR", PSNR_mean, epoch)
        self.writer.add_scalar("Val/SSIM", SSIM_mean, epoch)
        
        return PSNR_mean, SSIM_mean
    
    def train(self, train_loader, val_loader):
        best_psnr = 0
        
        print("######## Start IAT Distillation Training with Illumination #########")
        print(f"Teacher params: {sum(p.numel() for p in self.teacher.parameters())/1e6:.2f}M")
        print(f"Student params: {sum(p.numel() for p in self.student.parameters())/1e6:.2f}M")
        print(f"Warmup epochs: {self.config.warmup_epochs}\n")
        
        for epoch in range(self.config.num_epochs):
            # Train
            avg_losses = self.train_epoch(train_loader, epoch)
            
            print(f"\n[Epoch {epoch} Summary]")
            print(f"  Avg Total Loss: {avg_losses['total']:.4f}")
            print(f"  - GT: {avg_losses['gt']:.4f}")
            print(f"  - Illum_GT: {avg_losses['illum_gt']:.4f}")  # NEW
            print(f"  - Gamma: {avg_losses['gamma']:.4f}")
            print(f"  - Alpha_distill: {avg_losses['alpha_distill']:.3f}")
            print(f"  - Alpha_illum: {avg_losses['alpha_illum']:.3f}")  # NEW
            
            # Validate
            print("  Running validation...")
            PSNR_mean, SSIM_mean = self.validate(val_loader, epoch)
            print(f"  Val PSNR: {PSNR_mean:.4f}, SSIM: {SSIM_mean:.4f}")
            
            # Save log
            with open(os.path.join(self.config.snapshots_folder, 'distill_log.txt'), 'a+') as f:
                f.write(f"Epoch {epoch}: PSNR={PSNR_mean:.4f}, SSIM={SSIM_mean:.4f}, "
                       f"Loss={avg_losses['total']:.4f}, Illum={avg_losses['illum_gt']:.4f}\n")
            
            # Save best
            if PSNR_mean > best_psnr:
                best_psnr = PSNR_mean
                torch.save(self.student.state_dict(),
                          os.path.join(self.config.snapshots_folder, "student_best.pth"))
                print(f"  ✅ Saved best student model | PSNR={PSNR_mean:.4f}\n")
            else:
                print()
            
            # Checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save(self.student.state_dict(),
                          os.path.join(self.config.snapshots_folder, f"student_epoch_{epoch}.pth"))
            
            self.scheduler.step()
        
        print(f"\n{'='*60}")
        print(f"Training completed! Best PSNR: {best_psnr:.4f}")
        print(f"{'='*60}")
        
        self.writer.close()


# ==================== Main ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--img_path', type=str, default="/content/drive/MyDrive/LOL-v2/Real_captured/Train/Low/")
    parser.add_argument('--img_val_path', type=str, default="/content/drive/MyDrive/LOL-v2/Real_captured/Test/Low/")
    parser.add_argument("--normalize", action="store_false")
    parser.add_argument('--model_type', type=str, default='s')
    
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    
    # Teacher model path (필수!)
    parser.add_argument('--teacher_path', type=str, required=True,
                       help="Path to pretrained teacher IAT model")
    
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="workdirs/IAT_distillation_illum")
    
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
    
    val_dataset = lowlight_loader(images_path=config.img_val_path, mode='test', normalize=config.normalize)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=8, pin_memory=True
    )
    
    # Train
    trainer = IAT_DistillationTrainer(config)
    trainer.train(train_loader, val_loader)