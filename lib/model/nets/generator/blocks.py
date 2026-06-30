import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class Attention_block_3d(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block_3d,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
            )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x): # g:3D, x:3D
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, SeqLen, d_model]
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.embedding = nn.Linear(d_model, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=64)  # 4×4×4=64

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x):  # x: [B, 256, 4, 4, 4]
        x = rearrange(x, 'B C D H W -> B (D H W) C')  # [B, 64, 256]
        x = self.embedding(x)
        x = self.pos_encoder(x)

        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = residual + attn_output

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        # reshape back if needed: [B, 64, 256] → [B, 256, 4, 4, 4]
        x = rearrange(x, 'B (D H W) C -> B C D H W', D=4, H=4, W=4)
        return x

class Project2Dto3D(nn.Module):
    def __init__(self, c_in=704, c_out=256, d=4, h=4, w=4, nhead=8):
        super().__init__()
        self.d = d
        self.h = h
        self.w = w

        self.query_embed = nn.Parameter(torch.randn(1, d * h * w, c_out))  # 3D voxel queries

        self.attn = nn.MultiheadAttention(embed_dim=c_out, num_heads=nhead, batch_first=True, dropout=0.0)
        self.linear_proj = nn.Sequential(
            nn.Linear(c_in, c_out),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

    def forward(self, x):  # x: [B, c_in, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        x = self.linear_proj(x)           # [B, HW, c_out]

        query = self.query_embed.expand(B, -1, -1)  # [B, D*H*W, c_out]

        out, _ = self.attn(query, x, x)  # [B, D*H*W, c_out]

        out = out.transpose(1, 2).reshape(B, -1, self.d, self.h, self.w)  # [B, c_out, D, H, W]
        return out

class Project2Dto3D_FFN(nn.Module):
    def __init__(self, c_in=704, c_out=256, d=4, h=4, w=4, nhead=8):
        super().__init__()
        self.d = d
        self.h = h
        self.w = w
        self.pos_embed = nn.Parameter(torch.randn(1, h * w, c_out))  # [1, 16, 704]
        self.query_embed = nn.Parameter(torch.randn(1, d * h * w, c_out))  # 3D voxel queries

        self.attn = nn.MultiheadAttention(embed_dim=c_out, num_heads=nhead, batch_first=True, dropout=0.2)
        self.linear_proj = nn.Sequential(
            nn.Linear(c_in, c_out),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.norm1 = nn.LayerNorm(c_out)
        self.ffn = nn.Sequential(
            nn.Linear(c_out, c_out*4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(c_out*4, c_out)
        )
        self.norm2 = nn.LayerNorm(c_out)

    def forward(self, x):  # x: [B, c_in, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        x = self.linear_proj(x)           # [B, HW, c_out]
        x = x + self.pos_embed.expand(B, -1, -1)  # [B, HW, C] + [1, 16, 704] -> [B, HW, C]

        query = self.query_embed.expand(B, -1, -1)  # [B, D*H*W, c_out]

        out, _ = self.attn(query, x, x)  # [B, D*H*W, c_out]

        res = self.norm1(out + query) # add & norm
        out = self.ffn(res) # FFN
        out = self.norm2(out + res) # add & norm

        out = out.transpose(1, 2).reshape(B, -1, self.d, self.h, self.w)  # [B, c_out, D, H, W]
        return out
    
class TransformerBaseLink(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads=4, dropout=0.5, ffn_dim=None, num_layers=1):
        super().__init__()
        self.embed_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = 1  # single token

        if ffn_dim is None:
            ffn_dim = input_dim * 4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, C] → [B, 1, C]
        x = x.unsqueeze(1)
        x = self.transformer(x)  # [B, 1, C]
        x = self.out_proj(x.squeeze(1))  # [B, output_dim]
        return x

class DropBlock2D(nn.Module):
    def __init__(self, block_size=5, drop_prob=0.1):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        gamma = self.drop_prob * (x.numel() / x.shape[0]) / (
            (x.shape[2] - self.block_size + 1) * (x.shape[3] - self.block_size + 1)
        )
        mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) < gamma).float()
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1 - mask
        return x * mask * (mask.numel() / mask.sum())

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16, drop_p=0.1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False),
            nn.Dropout(p=drop_p)  # Dropout after attention weights
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, dropblock=True, drop_prob=0.1):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.use_dropblock = dropblock
        self.dropblock = DropBlock2D(drop_size=3, drop_prob=drop_prob) if dropblock else nn.Identity()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        out = x * attn
        out = self.dropblock(out)
        return out

class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, drop_p=0.1, dropblock=True):
        super().__init__()
        self.ca = ChannelAttention(in_planes, reduction, drop_p)
        self.sa = SpatialAttention(dropblock=dropblock, drop_prob=drop_p)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
    
class DropBlock3D(nn.Module):
    def __init__(self, block_size=1, drop_prob=0.1):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        gamma = self.drop_prob * (x.numel() / x.shape[0]) / (
            (x.shape[2] - self.block_size + 1) *
            (x.shape[3] - self.block_size + 1) *
            (x.shape[4] - self.block_size + 1)
        )
        mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4], device=x.device) < gamma).float()
        mask = F.max_pool3d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1 - mask

        # Prevent NaN
        eps = 1e-6
        return x * mask * (mask.numel() / (mask.sum() + eps))

class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, reduction=16, drop_p=0.1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // reduction, in_planes, 1, bias=False),
            nn.Dropout3d(p=drop_p)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention3D(nn.Module):
    def __init__(self, dropblock=True, drop_prob=0.1):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.use_dropblock = dropblock
        self.dropblock = DropBlock3D(block_size=3, drop_prob=drop_prob) if dropblock else nn.Identity()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        out = x * attn
        out = self.dropblock(out)
        return out

class CBAM3D(nn.Module):
    def __init__(self, in_planes, reduction=16, drop_p=0.1, dropblock=True):
        super().__init__()
        self.ca = ChannelAttention3D(in_planes, reduction, drop_p)
        self.sa = SpatialAttention3D(dropblock=dropblock, drop_prob=drop_p)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, reduction=8):  # 더 작은 reduction
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False)  # 커널 사이즈 3으로 축소
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn

class LightweightCBAM3D(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention3D(in_planes, reduction=8)
        self.sa = SpatialAttention3D()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class SEBlock3D(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class ChannelMLPBlock(nn.Module):
    def __init__(self, in_channels, hidden_ratio=0.5):
        super().__init__()
        hidden_dim = int(in_channels * hidden_ratio)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, 1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, in_channels, 1)
        )

    def forward(self, x):
        return x + self.fc(x)

class TinyTransformerBlock3D(nn.Module):
    def __init__(self, in_channels, drop_p=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.ff = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.ReLU(),
            nn.Linear(in_channels * 2, in_channels),
            nn.Dropout(drop_p),
        )

    def forward(self, x):
        b, c, d, h, w = x.shape
        x_flat = x.view(b, c, -1).transpose(1, 2)  # [B, D*H*W, C]
        x_out = self.ff(self.norm(x_flat))
        x_out = x_out.transpose(1, 2).view(b, c, d, h, w)
        return x + x_out

from mamba_ssm import Mamba

class MambaBlock3D(nn.Module):
    def __init__(self, in_channels, d_state=16, d_conv=4, expand=2, drop_p=0.3):
        super().__init__()
        
        # 1. Spatial Mixing (Mamba)
        # Mamba는 입력 시퀀스의 길이(L)에 선형적인 복잡도를 가짐
        self.norm1 = nn.LayerNorm(in_channels)
        self.mamba = Mamba(
            d_model=in_channels, # Model dimension d_model
            d_state=d_state,     # SSM state expansion factor
            d_conv=d_conv,       # Local convolution width
            expand=expand,       # Block expansion factor
        )
        
        # 2. Channel Mixing (기존 코드의 FF 부분)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ff = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.SiLU(), # ReLU보다 Mamba/LLM에서 많이 쓰이는 SiLU(Swish) 추천
            nn.Dropout(drop_p),
            nn.Linear(in_channels * 2, in_channels),
            nn.Dropout(drop_p),
        )

    def forward_mamba(self, x):
        """
        3D 이미지에 Mamba를 적용하기 위한 Forward 및 Bidirectional 처리
        x: [B, L, C] (Flattened)
        """
        # Mamba는 기본적으로 Causal(과거->미래)하므로, 이미지에서는 
        # 정방향 + 역방향을 합쳐주는 것이 성능에 매우 중요합니다.
        
        # 정방향 (Forward)
        out_fwd = self.mamba(x)
        
        # 역방향 (Backward): 시퀀스를 뒤집어서 넣고 다시 뒤집음
        # (Weight를 공유하거나 별도의 Mamba를 쓸 수 있는데, 여기서는 공유 방식 사용)
        out_bwd = self.mamba(x.flip(dims=[1])).flip(dims=[1])
        
        # 두 방향의 정보를 합침 (평균 또는 합)
        return out_fwd + out_bwd

    def forward(self, x):
        b, c, d, h, w = x.shape
        
        # [B, C, D, H, W] -> [B, D*H*W, C] 로 변환 (Sequence화)
        x_flat = x.view(b, c, -1).transpose(1, 2)
        
        # ---------------------------------------------------------
        # 1. Mamba Block (Spatial / Token Mixing)
        # ---------------------------------------------------------
        residual = x_flat
        x_norm = self.norm1(x_flat)
        
        # Bidirectional Mamba 적용
        x_mamba = self.forward_mamba(x_norm)
        
        x_flat = residual + x_mamba # Residual Connection

        # ---------------------------------------------------------
        # 2. Feed Forward Block (Channel Mixing) - 기존 로직 유지
        # ---------------------------------------------------------
        residual = x_flat
        x_norm = self.norm2(x_flat)
        
        x_ff = self.ff(x_norm)
        
        x_flat = residual + x_ff # Residual Connection
        
        # [B, D*H*W, C] -> [B, C, D, H, W] 로 복원
        x_out = x_flat.transpose(1, 2).view(b, c, d, h, w)
        
        return x_out
    
class ConvEnhancedTransformerBlock3D(nn.Module):
    def __init__(self, in_channels, drop_p=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.ff = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(in_channels * 2, in_channels)
        )

        # Local path (depthwise + pointwise conv)
        self.local_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # depthwise
            nn.Conv3d(in_channels, in_channels, kernel_size=1),  # pointwise
            nn.GELU()
        )

    def forward(self, x):
        b, c, d, h, w = x.shape

        # Transformer path
        x_flat = x.view(b, c, -1).transpose(1, 2)  # [B, D*H*W, C]
        x_trans = self.ff(self.norm(x_flat))
        x_trans = x_trans.transpose(1, 2).view(b, c, d, h, w)

        # Local convolution path
        x_local = self.local_conv(x)

        return x + x_trans + x_local

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast mask
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep_prob * random_tensor.floor()

class AdvancedTinyTransformer3D(nn.Module):
    def __init__(self, in_channels, mlp_ratio=2.0, drop_p=0.3, droppath_p=0.3):
        super().__init__()
        hidden_dim = int(in_channels * mlp_ratio)

        # Channel-wise normalization
        self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)

        # MLP path (token mixer)
        self.ff = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout3d(drop_p),
            nn.Conv3d(hidden_dim, in_channels, kernel_size=1)
        )

        # Local convolutional enhancement
        self.local_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # Depthwise
            nn.Conv3d(in_channels, in_channels, kernel_size=1),  # Pointwise
            nn.GELU()
        )

        # DropPath for regularization
        self.drop_path = DropPath(droppath_p)

    def forward(self, x):
        x_norm = self.norm(x)
        x_mlp = self.ff(x_norm)
        x_local = self.local_conv(x)

        return x + self.drop_path(x_mlp + x_local)

if __name__ == '__main__':  
    input_tensor = torch.randn(2, 256, 4, 4, 4).cuda()
    cbam3d = CBAM3D(in_planes=256, drop_p=0.5, dropblock=True).cuda()
    output = cbam3d(input_tensor)
    print(output.shape)
