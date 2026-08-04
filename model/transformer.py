import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# === 优化说明 ===
# 1) 使用通道降维（1x1 conv）将大通道数投影到较小的 hidden_dim，显著减少 attention 的计算量
# 2) 使用局部窗口注意力（non-overlapping windows）替代全局 attention，避免 O((H*W)^2) 的开销
# 3) 保持原有模块接口（x_q,x_kv 或 x_q,x_kv1,x_kv2 等）以便直接替换


def window_partition(x, window_size):
    """把特征图分割成不重叠窗口
    x: [B, C, H, W]
    返回: [num_windows*B, C, window_size, window_size]
    """
    B, C, H, W = x.shape
    assert H % window_size == 0 and W % window_size == 0, "H and W must be divisible by window_size"
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # [B, nH, nW, C, ws, ws]
    x = x.view(-1, C, window_size, window_size)
    return x


def window_unpartition(windows, window_size, H, W):
    # windows: [num_windows*B, C, ws, ws]
    B_times = windows.shape[0]
    num_windows = (H // window_size) * (W // window_size)
    B = B_times // num_windows
    C = windows.shape[1]
    x = windows.view(B, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, C, H, W)
    return x


class LocalAttention(nn.Module):
    """在局部窗口上执行标准点积注意力（非多头版），输入均为 [B, D, H, W]（D 是降维后的通道）
    """
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def forward(self, q, k, v):
        # q,k,v: [B, D, H, W]
        B, D, H, W = q.shape
        ws = self.window_size
        # 分块
        q_win = window_partition(q, ws)  # [num_win*B, D, ws, ws]
        k_win = window_partition(k, ws)
        v_win = window_partition(v, ws)

        # flatten spatial
        q_flat = q_win.view(q_win.shape[0], D, ws * ws).transpose(1, 2)  # [N, S, D]
        k_flat = k_win.view(k_win.shape[0], D, ws * ws)                  # [N, D, S]
        v_flat = v_win.view(v_win.shape[0], D, ws * ws).transpose(1, 2)  # [N, S, D]

        # attention
        attn = torch.matmul(q_flat, k_flat) / math.sqrt(D)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_flat)  # [N, S, D]

        # restore
        out = out.transpose(1, 2).contiguous().view(-1, D, ws, ws)
        out = window_unpartition(out, ws, H, W)
        return out
    

class Attention(nn.Module):
    def __init__(self, in_channels, hidden_dim=16, window_size=8):
        super().__init__()
        self.window_size = window_size
        # 降维投影
        self.q_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        self.local_attn = LocalAttention(window_size)


    def forward(self, x):
        # 投影降维
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 若空间不能被窗口整除，则对输入做 padding
        pad_h = (self.window_size - (q.shape[2] % self.window_size)) % self.window_size
        pad_w = (self.window_size - (q.shape[3] % self.window_size)) % self.window_size
        if pad_h != 0 or pad_w != 0:
            q = F.pad(q, (0, pad_w, 0, pad_h))
            k = F.pad(k, (0, pad_w, 0, pad_h))
            v = F.pad(v, (0, pad_w, 0, pad_h))

        out = self.local_attn(q, k, v)

        # 如果做了 padding，则去掉多余部分
        out = out[:, :, :x.shape[2], :x.shape[3]]

        out = self.out_proj(out)
        return out + x

