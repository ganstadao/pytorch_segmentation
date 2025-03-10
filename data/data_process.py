#自定义数据集,实现如下功能：
#进行数据加载 和 数据处理

#观察数据集，会发现manual为手动划分的分割结果即标签，mask为蒙版，image为原始图像

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
import os
import numpy as np
np.set_printoptions(threshold=np.inf)


class my_dataset(Dataset):
    def __init__(self,root_dir,train:bool,transforms=None):
        super(my_dataset,self).__init__()
        
        #check
        assert os.path.exists(root_dir), f"path '{root_dir}' does not exists."
        
        train_path=os.path.join(root_dir,'training')
        test_path=os.path.join(root_dir,'test')
        
        img_path=os.path.join(train_path,'images') if train else os.path.join(test_path,'images')
        manual_path=os.path.join(train_path,'1st_manual') if train else os.path.join(test_path,'1st_manual')
        roi_mask_path=os.path.join(train_path,'mask') if train else os.path.join(test_path,'mask')
        
        img_name=os.listdir(img_path)
        manual_name=os.listdir(manual_path)
        roi_mask_name=os.listdir(roi_mask_path)

        self.flag=train
        self.img_list=[os.path.join(img_path,i) for i in img_name]
        self.manual_list=[os.path.join(manual_path,i) for i in manual_name]
        #right mask
        self.roi_mask_list=[os.path.join(roi_mask_path,i) for i in roi_mask_name]

        self.transforms=transforms


    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):

        # img mode=RGB size=565x584 TiffImageFile 0-255
        # mask mode=L size=565x584 GifImageFile 0-255
        # manual mode=L size=565x584 GifImageFile 0-255  

        img=Image.open(self.img_list[index])# RGB PILImage
        manual=Image.open(self.manual_list[index])# L PILImage
        img,manual=img.convert('RGB'),manual.convert('L')# 输入要求三通道，输出单通道灰度图像
        manual=np.array(manual) / 255 # 由于像素值范围在0-255之间，语义分割前景从1开始，背景为0
        roi_mask=Image.open(self.roi_mask_list[index]).convert('L')

        #对于roi_mask,其中0的是背景即无效部分，255的是前景即有效部分，我们需要仅关注有效部分，而在计算损失时忽略255的部分
        roi_mask=255-np.array(roi_mask)

        #目标标签
        mask=np.clip(roi_mask+manual,0,255)
        mask=Image.fromarray(mask) # 转回PILImage图像，便于transforms

        #进行转化
        img,mask=self.transforms(img,mask)# -> tensor

        #输出[N,3,H,W] [N,1,H,W]
        return img,mask


'''test_dir=r"E:\AI\data\DRIVE\training\images\21_training.tif"
test_img=Image.open(test_dir)
print(np.array(test_img).max(),np.array(test_img).min())
#img mode=RGB size=565x584 TiffImageFile 0-255
test_dir=r"E:\AI\data\DRIVE\training\mask\21_training_mask.gif"
test_mask=Image.open(test_dir)
print(np.array(test_mask))
#mask mode=L size=565x584 GifImageFile 0-255
test_dir=r"E:\AI\data\DRIVE\training\1st_manual\21_manual1.gif"
test_manual=Image.open(test_dir)
print(np.array(test_manual).max(),np.array(test_manual).min())
#manual mode=L size=565x584 GifImageFile 0-255'''




    


    