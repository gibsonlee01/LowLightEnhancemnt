import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_

class SaturationRefinementModule(nn.Module):
    """
    Lightweight saturation refinement module
    Input: base_output (3ch) + illumination map (1ch) = 4ch
    Output: saturation adjustment map (1ch)
    """
    def __init__(self, in_channels=4, hidden_dim=32):
        super(SaturationRefinementModule, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.conv_out = nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, base_output, illum_map):
        """
        Args:
            base_output: [B, 3, H, W] - base model output
            illum_map: [B, 1, H, W] - illumination map
        Returns:
            sat_map: [B, 1, H, W] - saturation adjustment map in [0, 1]
        """
        # Concatenate inputs
        x = torch.cat([base_output, illum_map], dim=1)  # [B, 4, H, W]
        
        # Forward through network
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        sat_map = self.sigmoid(self.conv_out(x))
        
        return sat_map