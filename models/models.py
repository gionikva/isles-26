from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import torch
# from torch.nn import Conv3d


   
class GhostConv3D(Module):
    def __init__(self, in_channels, out_channels=8, downscale=True, ratio=2):
        super().__init__()
        self.out_channels = out_channels
        int_channels = out_channels // ratio
        new_channels = int_channels * (ratio - 1)
        
        
        pr_kernel_size = 3
        pr_stride = 2 if downscale else 1
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
        S = logits.reshape(batch_size, self.num_anchors, 3)
        return S
        
class SpatialAnchorFiLM(Module):
    def __init__(self, in_channels, num_anchors = 8):
        super().__init__()
        
        self.gamma_proj = nn.Linear(num_anchors * 3, in_channels)
        self.beta_proj = nn.Linear(num_anchors * 3, in_channels)
        
    def forward(self, S, gconvout):
        
        s = S.flatten(start_dim=1)
        gamma = self.gamma_proj(s)
        beta = self.beta_proj(s)
        
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]
        
        F_cond = (gamma + 1) * gconvout + beta
        return F_cond
                
class Encoder(Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
       
        self.ghost_conv = GhostConv3D(in_channels, out_channels, downscale=False)
        self.spatial_film = SpatialAnchorFiLM(out_channels)
        
        self.detail_preserve = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, grups=out_channels),
            nn.GroupNorm(4, out_channels),
            nn.SiLU()
        )
        
        self.smoothing = nn.Conv3d(out_channels, out_channels, 1, 1, 0)
        
        reduction_ratio = max(4, out_channels // 8)
        
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(out_channels, out_channels // reduction_ratio),
            nn.SiLU(),
            nn.Linear(out_channels // reduction_ratio, out_channels),
            nn.Sigmoid()
        )
        
        self.max_pool = nn.MaxPool3d(2, 2) if downsample else nn.Identity()

    def forward(self, f0_prime, S, T):
        convout = self.ghost_conv(f0_prime)
        conditioned = self.spatial_film(S, convout)
        
        stage_resolution = conditioned.shape[2:]
        T_resized = F.interpolate(T, size=stage_resolution, mode='trilinear')
        
        z_detail = self.detail_preserve(conditioned)
        z_smooth = self.smoothing(conditioned)
        
        texture_aware = T_resized * z_detail + (1 - T_resized) * z_smooth
        
        
        E_i = self.bottleneck(texture_aware) * texture_aware
        
        out = self.max_pool(E_i)
        return out, E_i

class MultiScaleSkipFusion(Module):
    def __init__(self):
        super().__init__()
        
        internal_channels = 64
        concat_channels = 256
        n_stages = 4
        
        self.proj_1 = nn.Conv3d(in_channels=8, out_channels=internal_channels, kernel_size=1, stride=1, padding=0)
        self.proj_2 = nn.Conv3d(in_channels=16, out_channels=internal_channels, kernel_size=1, stride=1, padding=0)
        self.proj_3 = nn.Conv3d(in_channels=32, out_channels=internal_channels, kernel_size=1, stride=1, padding=0)
        self.proj_4 = nn.Conv3d(in_channels=64, out_channels=internal_channels, kernel_size=1, stride=1, padding=0)

        self.controller_1 = nn.Sequential(
            nn.Conv3d(in_channels=concat_channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv3d(in_channels=internal_channels, out_channels=n_stages, kernel_size=1, stride=1, padding=0),
        )
        
        self.controller_2 = nn.Sequential(
            nn.Conv3d(in_channels=concat_channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv3d(in_channels=internal_channels, out_channels=n_stages, kernel_size=1, stride=1, padding=0),
        )
        
        self.out_proj_1 = nn.Conv3d(in_channels=internal_channels, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.out_proj_2 = nn.Conv3d(in_channels=internal_channels, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.out_proj_3 = nn.Conv3d(in_channels=internal_channels, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.out_proj_4 = nn.Conv3d(in_channels=internal_channels, out_channels=8, kernel_size=1, stride=1, padding=0)


    def forward(self, E_1, E_2, E_3, E_4):
        
        E_hat_1 = self.proj_1(E_1)
        E_hat_2 = self.proj_2(E_2)
        E_hat_3 = self.proj_3(E_3) 
        E_hat_4 = self.proj_4(E_4)
        
        skip_4 = self.out_proj_4(E_hat_2)
        skip_4 = F.interpolate(skip_4, size=E_1.shape[2:], mode='trilinear')
        
        skip_3 = self.out_proj_3(E_hat_1)
        skip_3 = F.interpolate(skip_3, size=E_2.shape[2:], mode='trilinear')
      
        # resizing encoder outputs for skip 2
        up_2_1 = F.interpolate(E_hat_1, size=E_3.shape[2:], mode='trilinear')
        up_2_2 = F.interpolate(E_hat_2, size=E_3.shape[2:], mode='trilinear')
        up_2_3 = F.interpolate(E_hat_3, size=E_3.shape[2:], mode='trilinear')
        up_2_4 = F.interpolate(E_hat_4, size=E_3.shape[2:], mode='trilinear')
        
        cat_2 = torch.cat([up_2_1, up_2_2, up_2_3, up_2_4], dim=1)
        logits_2 = self.controller_2(cat_2)
        weights_2 = F.softmax(logits_2, dim=1)
        
        skip_2 = weights_2[:, 0, :, :, :] * up_2_1 + \
                 weights_2[:, 1, :, :, :] * up_2_2 + \
                 weights_2[:, 2, :, :, :] * up_2_3 + \
                 weights_2[:, 3, :, :, :] * up_2_4
                 
        skip_2 = self.out_proj_2(skip_2)
        
        # resizing encoder outputs for skip 1
        up_1_1 = F.interpolate(E_hat_1, size=E_4.shape[2:], mode='trilinear')
        up_1_2 = F.interpolate(E_hat_2, size=E_4.shape[2:], mode='trilinear')
        up_1_3 = F.interpolate(E_hat_3, size=E_4.shape[2:], mode='trilinear')
        up_1_4 = F.interpolate(E_hat_4, size=E_4.shape[2:], mode='trilinear')
        
        cat_1 = torch.cat([up_1_1, up_1_2, up_1_3, up_1_4], dim=1)
        logits_1 = self.controller_1(cat_1)
        weights_1 = F.softmax(logits_1, dim=1)
        
        skip_1 = weights_1[:, 0, :, :, :] * up_1_1 + \
                 weights_1[:, 1, :, :, :] * up_1_2 + \
                 weights_1[:, 2, :, :, :] * up_1_3 + \
                 weights_1[:, 3, :, :, :] * up_1_4
                 
        skip_1 = self.out_proj_1(skip_1)
        
        return skip_1, skip_2, skip_3, skip_4
    
    
class LightMedSeg(Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        pass
    
 