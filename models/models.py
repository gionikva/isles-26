from torch.nn import Module
import torch.nn as nn
import torch
# from torch.nn import Conv3d


   
class GhostConv3D(Module):
    def __init__(self, in_channels, out_channels=8, ratio=2):
        super().__init__()
        self.out_channels = out_channels
        int_channels = out_channels // ratio
        new_channels = int_channels * (ratio - 1)
        
        pr_kernel_size = 3
        pr_stride = 2
        pr_padding = 1
        
        dw_kernel_size = 3
        dw_stride = 1
        dw_padding = 1
        
        # n. of groups for group norm
        n_groups = 4
        
        self.primary = nn.Sequential(
            nn.Conv3d(in_channels, int_channels, pr_kernel_size, pr_stride, pr_padding, bias=False),
            nn.GELU()
        )
      
        
        self.depthwise = nn.Sequential(
            nn.Conv3d(int_channels, new_channels, dw_kernel_size, dw_stride, dw_padding, groups=int_channels, bias=False),
            nn.GELU()
        )
        
        self.group_norm = nn.GroupNorm(n_groups, out_channels)
        
    def forward(self, X):
        x1 = self.primary(X)
        x2 = self.depthwise(x1)
        cat = torch.cat([x1, x2], dim=1)
        out = self.group_norm(cat)
        return out
    
class GlobalAnchorDetector(Module):
    def __init__(self, in_channels, out_channels, num_anchors = 8):
        super().__init__()
        
        self.num_anchors = num_anchors
        
        kernel = 3
        stride = 2
        padding = 1
        
        # n. of groups for GN
        n_groups = 4
        
        # mlp hidden layer size
        hidden_width = 128
        
        self.phi = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel, stride, padding),
            nn.GroupNorm(n_groups, out_channels),
            nn.SiLU()
        )
        
        self.pool = nn.AdaptiveAvgPool3d([1, 1, 1])
        
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 3 * num_anchors)
        )
        
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x1 = self.phi(x)
        x2 = self.pool(x1)
        x2 = x2.flatten(start_dim = 1)
        x3 = self.mlp(x2)
        logits = self.sigmoid(x3)
        return logits.reshape(batch_size, self.num_anchors, 3)
        # self.mlp = 

class LightMedSeg(Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        pass
    
 