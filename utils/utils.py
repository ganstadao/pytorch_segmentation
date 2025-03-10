import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt

#计算损失,用的dice loss，但是要剔除255的部分
def criterion(outputs,target,loss_weight=None,num_classes:int=2,dice:bool=True,ignore_index:int=100):
    losses={}
    #[N,1,H,W],[N,1,H,W]
    for name, x in outputs.items():
        #输入x是[N,C,H,W],C=numclasses
        loss=F.cross_entropy(x,target,weight=loss_weight,ignore_index=ignore_index)
        if dice is True:
            #创建对应计算dice对象
            dice_target = build_target(target, num_classes, ignore_index)
            #将dice_loss加入损失中
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name]=loss

    #这里就只有一个主输出
    if len(losses) == 1:
        return losses['out']

    #如果有辅助输出的情况下一般损失函数如下计算
    return losses['out'] + 0.5 * losses['aux']
        
# 目标会有前景若干类(1...)+背景0+无效部分255
# build_target需要将[N,1,H,W]转化为每个类别上的独热编码形式，从而得到[N,C,H,W](其中C是类别数)
# 这样返回的对象才能和[N,C,H,W]的x softmax后计算dice loss损失
def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)

def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size

def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    return dice / x.shape[1]

#计算dice_loss损失
def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    # Dice loss (objective to minimize) between 0 and 1
    x = nn.functional.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)


def plot_loss_curve(loss,path:str):
    plt.plot(loss)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(path)
    plt.show()


def plot_img_and_mask(img, mask, pred_mask, output_path, epoch=None, index=0):
    """
    绘制原始图像、真实mask和预测mask的对比图
    """
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像（需要反标准化）
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
    ax[0].imshow(img)
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    # 真实mask
    if classes > 1:
        true_mask = torch.argmax(mask, dim=0).cpu().numpy()
    else:
        true_mask = mask.cpu().numpy().squeeze()

    # 剔除255像素值（忽略区域）
    true_mask = np.where(true_mask == 255, 0, true_mask)  # 将255替换为0（背景）
    
    ax[1].imshow(true_mask, cmap='gray')
    ax[1].set_title('Ground Truth')
    ax[1].axis('off')

    # 预测mask
    if classes > 1:
        pred_mask = torch.argmax(pred_mask, dim=0).cpu().numpy()
    else:
        pred_mask = torch.sigmoid(pred_mask).cpu().numpy().squeeze()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
    ax[2].imshow(pred_mask.squeeze(), cmap='gray')
    ax[2].set_title('Prediction')
    ax[2].axis('off')

    # 保存图像
    if epoch is not None:
        output_path = os.path.join(output_path, f'epoch_{epoch}_sample_{index}.png')
    else:
        output_path = os.path.join(output_path, f'sample_{index}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def confusion_matrix(y_pred, y_true, num_classes):
    """计算混淆矩阵"""
    mask = (y_true >= 0) & (y_true < num_classes)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    cm = torch.bincount(
        num_classes * y_true + y_pred,
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    return cm

def compute_miou(cm):
    """从混淆矩阵计算mIoU"""
    iou_per_class = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[i, :].sum() - tp
        fn = cm[:, i].sum() - tp
        denominator = tp + fp + fn
        if denominator == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append(tp / denominator)
    miou = np.nanmean(iou_per_class)
    return miou, iou_per_class



