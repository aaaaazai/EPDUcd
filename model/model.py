import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DWConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=7, padding=3, groups=in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=1)
        )

    def forward(self, x):
        return self.dwconv(x)


class CBA1x1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.cba = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.cba(x)


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(FeatureExtractor, self).__init__()
        # 禁用自动下载，仅初始化网络结构
        resnet = models.resnet34(pretrained=False)
        # 加载本地预训练权重
        state_dict = torch.load('../weight/resnet34-333f7ec4.pth')
        resnet.load_state_dict(state_dict)

        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.mlfa = Multi_Level_Feature_Aggreagation()

    def forward(self, x):
        x = self.layer0(x)  # size:1/2      64*128*128
        x = self.maxpool(x)  # size:1/4      64*64*64
        x_low = self.layer1(x)  # size:1/4      64*64*64
        x1 = self.layer2(x_low)  # size:1/8      128*32*32
        x2 = self.layer3(x1)  # size:1/8     256*32*32
        x3 = self.layer4(x2)  # size:1/8     512*32*32
        x = self.mlfa(x1, x2, x3)  # 128*32*32
        return x


class Multi_Level_Feature_Aggreagation(nn.Module):

    def __init__(self, ):
        super(Multi_Level_Feature_Aggreagation, self).__init__()

        self.proj1 = DWConv(512, 128)
        self.proj2 = DWConv(256, 128)

        self.cat_conv = CBA1x1(384, 128)

    def forward(self, x1, x2, x3):
        x3 = self.proj1(x3)
        x2 = self.proj2(x2)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.cat_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=3, bilinear=True):
        super(UNet, self).__init__()

        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.down4 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.up1 = Up(128, 128 // factor, False)
        self.up2 = Up(64, 64 // factor, False)
        self.up3 = Up(32, 32 // factor, False)
        self.up4 = Up(16, 3, False)
        self.outc = nn.Conv2d(16, out_channels, kernel_size=1)  # 新增输出层

    def forward(self, x):
        x1 = self.inc(x)
        print('x1', x1.shape)
        x2 = self.down1(x1)
        print('x2', x2.shape)
        x3 = self.down2(x2)
        print('x3', x3.shape)
        x4 = self.down3(x3)
        print('x4', x4.shape)
        x5 = self.down4(x4)
        print('x5', x5.shape)

        x6 = self.up1(x5, x4)
        print('x6', x6.shape)
        x7 = self.up2(x6, x3)
        print('x7', x7.shape)
        x8 = self.up3(x7, x2)
        print('x8', x.shape)
        x = self.up4(x8, x1)
        print('x', x.shape)
        return x


class UCNet(nn.Module):
    def __init__(self, in_channels=5):
        super(UCNet, self).__init__()
        self.unet = UNet(in_channels=in_channels, out_channels=3)

    def forward(self, T, alpha, mu1):
        B, C, H, W = T.shape

        combined = torch.cat([T, alpha], dim=1)
        combined = torch.cat([combined, mu1], dim=1)

        return self.unet(combined)


class CENet(nn.Module):
    def __init__(self, in_channels=5):
        super(CENet, self).__init__()
        # 第一路分支 (Δ处理)
        self.delta_conv1 = nn.Conv2d(5, 64, 3, padding=1)
        self.delta_res1 = ResidualBlock(64)
        self.delta_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.delta_res2 = ResidualBlock(64)
        self.delta_conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.delta_final = nn.Conv2d(160, 1, 1)

        # 第二路分支 (▽Δ处理)
        self.grad_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.grad_res1 = ResidualBlock(64)
        self.grad_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.grad_res2 = ResidualBlock(64)
        self.grad_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.grad_final = nn.Conv2d(64, 1, 1)

    def forward(self, Delta, grad_Delta, beta, gamma, mu2, mu3):
        B, _, H, W = Delta.shape
        combined = torch.cat([Delta, beta], dim=1)
        combined = torch.cat([combined, mu2], dim=1)
        d_conv1 = self.delta_conv1(combined)
        d_res1 = self.delta_res1(d_conv1)
        d_conv2 = self.delta_conv2(d_res1)  # 跳跃连接1
        # print("combined", combined.shape)
        # print("d_conv1", d_conv1.shape)
        # print("d_res1",d_res1.shape)
        # print("d_conv2", d_conv2.shape)

        d_res2 = self.delta_res2(d_conv2 + d_conv1)
        d_conv3 = self.delta_conv3(d_res2)
        # 拼接操作 [conv1, conv2, conv3]
        delta_combined = torch.cat([d_conv1, d_conv2, d_conv3], dim=1)
        Delta_prime = self.delta_final(delta_combined)
        # print("d_res2", d_res2.shape)
        # print("d_conv3", d_conv3.shape)
        # print("delta_combined", delta_combined.shape)
        # print("conv_Delta_prime", Delta_prime.shape)

        # 第二路处理
        g_conv1 = self.grad_conv1(torch.cat([grad_Delta, gamma, mu3], dim=1))
        g_res1 = self.grad_res1(g_conv1)
        g_conv2 = self.grad_conv2(g_res1)  # 跳跃连接1
        g_res2 = self.grad_res2(g_conv2 + g_conv1)
        g_conv3 = self.grad_conv3(g_res2)
        # print("g_conv1", g_conv1.shape)
        # print("g_res1", g_res1.shape)
        # print("g_conv2", g_conv2.shape)
        # print("g_res2", g_res2.shape)
        # print("g_conv3", g_conv3.shape)
        # 添加Canny边缘特征
        canny_feat = canny_edge_detector(delta_combined.detach())  # 使用第一路特征
        # 拼接操作 [conv2, conv3, canny_feat]
        # grad_combined = torch.cat([g_conv2, g_conv3, canny_feat], dim=1)
        grad_combined = g_conv1 + g_conv2 + g_conv3 + canny_feat
        grad_Delta_prime = self.grad_final(grad_combined)
        # print("grad_combined", grad_combined.shape)
        # print("conv_grad_Delta_prime", grad_Delta_prime.shape)

        return Delta_prime, grad_Delta_prime


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x += residual
        return self.relu(x)


def canny_edge_detector(x):
    # 转换为灰度
    gray = x.mean(dim=1, keepdim=True)
    # 将Sobel算子权重转移到当前设备
    sobel_kernel_x = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]]).float().to(gray.device)
    sobel_kernel_y = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]]).float().to(gray.device)

    # Sobel算子边缘检测
    sobel_x = F.conv2d(gray, sobel_kernel_x, padding=1)
    sobel_y = F.conv2d(gray, sobel_kernel_y, padding=1)
    edge = torch.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return torch.sigmoid(edge)  # 归一化到0-1

class TBlock(nn.Module):
    def forward(self, T1, T2, Delta, T_prime, mu1):
        T = (T2 - Delta.expand_as(T2) + T1 + mu1.expand_as(T2) * T_prime) / (2 + mu1).expand_as(T2)
        return T

class DeltaBlock(nn.Module):
    def forward(self, T2, T, Delta_prime, mu2):
        Z = (T2 - T + mu2 * Delta_prime) / (1 + mu2)
        U, S, Vt = torch.linalg.svd(Z, full_matrices=False)
        Sigma_diag = torch.diag_embed(S)
        lambda_ = 1.0 / (2 * (1 + mu2))
        Sigma_thresh = torch.sign(Sigma_diag) * torch.clamp(torch.abs(Sigma_diag) - lambda_, min=0)
        Delta = U @ Sigma_thresh @ Vt
        return Delta

class GradDeltaBlock(nn.Module):
    def forward(self, grad_Delta_prime, mu3):
        norms = torch.norm(grad_Delta_prime, p=2, dim=3, keepdim=True)  # [B,C,1,W]

        shrinkage = torch.clamp(1.0 - 1.0 / (2 * mu3 * norms + 1e-8), min=0.0)
        v_opt = shrinkage * grad_Delta_prime
        return v_opt

class hyparaNet(nn.Module):
        return hypara


class Network(nn.Module):
        return T_prime, Delta_prime, grad_Delta


if __name__ == "__main__":

    T1 = torch.randn(2, 3, 256, 256)
    T2 = torch.randn(2, 3, 256, 256)

    B, C, H, W = 2, 128, 32, 32
    Delta = torch.randn(B, C, H, W)
    T_prime = torch.randn(B, C, H, W)
    mu1 = torch.randn(1)
    alpha = torch.randn(1)
    mu2 = torch.randn(1)
    mu3 = torch.randn(1)
    beta = torch.randn(1)
    gamma = torch.randn(1)
    grad_Delta = torch.randn(B, C, H, W)

    FeatureExtractor = FeatureExtractor()
    T1 = FeatureExtractor(T1)
    T2 = FeatureExtractor(T2)

    tb = TBlock()
    T = tb(T1, T2, Delta, T_prime, mu1)
    print("T", T.shape)

    ucnet = UCNet()
    T_prime = ucnet(T, alpha, mu1)
    print("T prime", T_prime.shape)

    db = DeltaBlock()
    Delta = db(T2, T, Delta, mu2)
    print("Delta", Delta.shape)

    cenet = CENet()
    Delta_prime, grad_Delta_prime = cenet(Delta, grad_Delta, beta, gamma, mu2, mu3)
    print("Delta prime", Delta_prime.shape)
    print("gradDelta prime", grad_Delta_prime.shape)

    gdb = GradDeltaBlock()
    grad_Delta = gdb(grad_Delta_prime, mu3)
    print("gradDelta", grad_Delta.shape)
    # 测试hyparaNet
    hypara_net = hyparaNet(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1)
    grad_Delta = torch.randn(B, 2, H, W)
    hypara = hypara_net(grad_Delta)
    print("hypara", hypara.shape)

'''
# 修改测试部分
if __name__ == "__main__":
    import cv2
    import numpy as np
    from torchvision import transforms

    B, C, H, W = 1, 3, 256, 256
    A_path = "../datasets/LEVIRCD_FC/train/A/train_12.jpg"
    B_path = "../datasets/LEVIRCD_FC/train/B/train_12.jpg"
    label_path = "../datasets/LEVIRCD_FC/train/label/train_12.jpg"
    grad_label_path = "../datasets/LEVIRCD_FC/train/grad_label/train_12.jpg"

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    label_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # 加载输入数据
    T1 = transform(Image.open(A_path).convert('RGB')).unsqueeze(0)
    T2 = transform(Image.open(B_path).convert('RGB')).unsqueeze(0)
    label = label_transform(Image.open(label_path)).unsqueeze(0)
    grad_label = label_transform(Image.open(grad_label_path)).unsqueeze(0)

    # 初始化网络
    net = Network()
    # 前向传播
    T_prime, Delta_prime, grad_Delta = net(T1, T2, label, grad_label)

    print("Delta prime", Delta_prime.shape)
    print("gradDelta prime", grad_Delta.shape)
    print("T prime", T_prime.shape)
    # 选择第一个样本进行可视化
    idx = 0


    # 转换张量为numpy格式
    def tensor_to_numpy(tensor):
        return tensor[idx].detach().cpu().numpy().transpose(1, 2, 0)


    # Delta_prime可视化（单通道）
    delta_vis = tensor_to_numpy(Delta_prime)[..., 0] * 255
    grad_vis = tensor_to_numpy(grad_Delta)[..., 0] * 255
    t_vis = tensor_to_numpy(T_prime)[..., 0] * 255  # 假设T_prime是单通道

    # 合并显示
    combined = np.hstack([delta_vis, grad_vis, t_vis]).astype(np.uint8)

    cv2.imshow("Delta_prime | GradDelta_prime | T_prime", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
