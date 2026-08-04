import torch
import torch.nn as nn
import torch.nn.functional as F

#常用多分类交叉熵
class CombinedLoss(nn.Module):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()

    def forward(self, input, target, weight=None, reduction='mean', ignore_index=255):
        target = target.long()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)
        if input.shape[-1] != target.shape[-1]:
            input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

        return F.cross_entropy(input=input, target=target, weight=weight,
                               ignore_index=ignore_index, reduction=reduction)


#单通道bce二分类损失函数
class BinaryCombinedLoss(nn.Module):
    """
    适用于单通道二分类的损失函数（带 sigmoid）
    :param input: torch.Tensor, N*1*H*W（模型原始输出，未激活）
    :param target: torch.Tensor, N*1*H*W（0/1 二值标签）
    :return: 二值交叉熵损失
    """
    def __init__(self):
        super(BinaryCombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()  # 内置 sigmoid + BCE

    def forward(self, input, target):
        # 尺寸对齐（与原逻辑一致）
        if input.shape[-2:] != target.shape[-2:]:
            input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)
        # 目标标签需为 float 类型（0.0/1.0）
        # print(target)
        return self.bce_loss(input, target.float())

#推拉
class MyEntropyLoss(nn.Module):
    def __init__(self):
        super(MyEntropyLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, outputs, labels):  # 自己写损失函数

        outputs = self.softmax(outputs)
        outputs = outputs[:, 1:2, :, :]  # 先切片得到第二个变化的图片

        nc = torch.sum((labels == 1).float())  # 加和就是1的数量   1就是变化，0就是没变化
        nu = torch.sum((labels == 0).float())

        loss1 = 0
        loss2 = 0
        if nc != 0:  # 写了这个就不用写加一个特别小的数字了
            loss1 = torch.sum(labels * torch.clamp(1 - outputs, min=0.0)) / nc

        if nu != 0:
            loss2 = torch.sum((0 - labels) * outputs) / nu

        loss = loss1 + loss2

        return loss


class TLoss(nn.Module):
    def __init__(self):
        super(TLoss, self).__init__()
        self.mse_loss = nn.MSELoss()  # 二范数损失

    def forward(self, image, recon_image):
        loss = self.mse_loss(image, recon_image)  # 使用MSE损失计算
        return loss


class grad_label_Loss(nn.Module):
    def __init__(self):
        super(grad_label_Loss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, image, recon_image):
        loss = self.l1_loss(image, recon_image)
        return loss

#摘自-边界检测和任务交互CD
class BinaryDiceLoss1(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1e-8, p=1, reduction='mean'):
        super(BinaryDiceLoss1, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        if torch.isnan(predict).any():
            print("Warning: predict contains NaN before sigmoid!")  # 调试时启用
            predict = torch.clamp(predict, -1e3, 1e3)  # 限制极端值
        predict = torch.sigmoid(predict)
        predict = torch.clamp(predict, 1e-7, 1 - 1e-7)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = 2*torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den
        return loss.sum()

#摘自-伪目标
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, predict, target):
        smooth = 1  # 避免除零
        p = 2  # 次方参数
        valid_mask = torch.ones_like(target)  # 有效区域掩码（此处默认全为1，表示所有区域都参与计算）

        # 展平预测和目标张量
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)

        # 计算Dice系数
        num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
        den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
        loss = 1 - num / den  # 1 - Dice系数（损失越小越好）

        return loss.mean()

#摘自-半监督双模态显著目标检测
class BCE_IOULoss(nn.Module):
    def __init__(self):
        super(BCE_IOULoss, self).__init__()
    def forward(self, predict, target):
        # 1. 计算权重图：通过平均池化与原始掩码的差异生成细节区域权重（边缘权重大）
        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        # 2. 计算目标区域的平均权重
        count = (weit > 1).sum().item()
        sum_of_values = torch.sum(weit[weit > 1])
        foreground_weight = sum_of_values / count
        weit = torch.where(target == 1, foreground_weight * weit, weit)
        # 3. 计算加权BCE损失（逐像素保留损失再与权重图乘）
        wbce = F.binary_cross_entropy_with_logits(predict, target, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-8)
        # 4. 计算加权IoU损失（）
        pred = torch.sigmoid(predict)
        inter = ((pred * target) * weit).sum(dim=(2, 3))
        union = ((pred + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1 + 1e-8)
        return (wbce + wiou).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super(BCEDiceLoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = BinaryDiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)

        return self.weight_bce * bce_loss + self.weight_dice * dice_loss

#降低大量易分类样本的权重，使模型更加关注那些难以正确分类的样本
#当αt=1时，当γ=0时，Focal Loss退化为标准的交叉熵损失
class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss