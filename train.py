
import torch
from torch.utils.data import DataLoader
from data.data_process import *
from model.unet import *
from torch import optim
from torchsummary import summary
from utils.utils import *
import utils.transforms as T


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 480

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def main(args):

    batch_size=args.batch_size
    epochs=args.epochs
    learning_rate=args.lr
    num_classes=args.num_classes+1 # 还需加上背景的一类
    device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_path=args.data_path
    weight_path="./results/weights/unet_weight.pth"
    loss_curve_path="./results/loss/loss_curve.png"
    predict_path="./results/predict"

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    #暂时用的默认的transform，看dataprocess.py
    train_dataset=my_dataset(data_path,True,transforms=get_transform(train=True, mean=mean, std=std))
    val_dataset=my_dataset(data_path,False,transforms=get_transform(train=False, mean=mean, std=std))

    train_dataloader=DataLoader(train_dataset,batch_size,shuffle=True)
    val_dataloader=DataLoader(val_dataset,batch_size)

    model=Unet(in_channels=3,num_classes=num_classes,base_c=32).to(device)
    
    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if os.path.exists(weight_path):
        print("weight found,loading...")
        model.load_state_dict(torch.load(weight_path,weights_only=True))
        eval(model,val_dataloader,device,num_classes,save_dir=predict_path)
    else:
        print("weight not found,start training...")
        
        model.train()
        total_loss=[]
        for epoch in range(epochs):

            running_loss=0.0
            best_miou = 0.0
            for img,mask in train_dataloader:
                img,mask=img.to(device),mask.to(device)
                #torch.Size([4, 3, 480, 480]) torch.Size([4, 480, 480])

                outputs=model(img)

                optimizer.zero_grad()

                #这里criterion损失函数的计算需要剔除mask中255的部分
                loss=criterion(outputs,mask,loss_weight,num_classes,ignore_index=255)
                running_loss+=loss.item()
                loss.backward()

                optimizer.step()

            avg_loss = running_loss / len(train_dataloader)
            total_loss.append(avg_loss)
            if (epoch+1) % 10==0:
                print(f"epoch {epoch+1}, loss {running_loss/100:.4f}")
                plot_loss_curve(total_loss,loss_curve_path)

        torch.save(model.state_dict(),weight_path)
        print(f"Model saved as {weight_path}")

        # 每个epoch结束后评估
        if (epoch+1) % 10 == 0:
            current_miou = eval(
                model=model,
                dataloader=val_dataloader,
                device=device,
                num_classes=num_classes,
                epoch=epoch+1,
                save_dir="./results/predict/"
            )
            
            # 保存最佳模型
            if current_miou > best_miou:
                best_miou = current_miou
                torch.save(model.state_dict(), "./results/weights/best_model.pth")
                print(f"New best model saved with mIoU: {best_miou:.4f}")
            


def eval(model, dataloader, device, num_classes, epoch=None, save_dir='./results/predict/'):
    print("Evaluating model...")
    model.eval()
    total_cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)['out']
            
            # 转换预测结果
            if num_classes == 1:
                preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
            else:
                preds = torch.argmax(outputs, dim=1)
            
            # 更新混淆矩阵
            preds_flat = preds.view(-1).cpu()
            targets_flat = targets.view(-1).cpu()
            cm = confusion_matrix(preds_flat, targets_flat, num_classes)
            total_cm += cm
            
            # 每批次保存前两个样本的可视化结果
            if idx < 2:  # 每个batch保存前两个样本
                for i in range(min(2, images.size(0))):

                    plot_img_and_mask(
                        img=images[i],
                        mask=targets[i],
                        pred_mask=preds[i],
                        output_path=save_dir,
                        epoch=epoch,
                        index=idx*images.size(0)+i
                    )
    
    # 计算指标
    miou, iou_per_class = compute_miou(total_cm.numpy())
    
    print("\nEvaluation Results:")
    print(f"mIoU: {miou:.4f}")
    for i, iou in enumerate(iou_per_class):
        print(f"Class {i} IoU: {iou:.4f}")
    
    return miou


def parser_args():
    import argparse
    parser=argparse.ArgumentParser(description="pytorch unet training")
    
    #定义参数：学习率、设备、类别、数据路径、迭代次数、批次大小
    parser.add_argument('--data_path',default='./data/DRIVE/',help='DRIVE root')
    parser.add_argument('--device',default="cuda",help='training device')
    parser.add_argument('--num_classes',default=1,type=int)
    parser.add_argument('--epochs',default=200,type=int,metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument("-b", "--batch_size", default=4, type=int)#-b缩写
    parser.add_argument("--momentum",default=0.9,type=float,help='mementum')


    args=parser.parse_args()

    return args
    



if __name__ == '__main__':
    args=parser_args()

    main(args)


