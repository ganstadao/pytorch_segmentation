import torch.nn as nn
import torch
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c:int =64):
        super(Unet,self).__init__()

        self.in_conv=DoubleConv(in_channels,base_c)

        #经过四层下采样和四层上采样
        self.down1=Down(base_c,base_c*2)
        self.down2=Down(base_c*2,base_c*4)
        self.down3=Down(base_c*4,base_c*8)
        factor = 2 if bilinear else 1
        self.down4=Down(base_c*8,base_c*16//factor)
        self.up1=Up(base_c*16,base_c*8//factor,bilinear)
        self.up2=Up(base_c*8,base_c*4//factor,bilinear)
        self.up3=Up(base_c*4,base_c*2//factor,bilinear)
        self.up4=Up(base_c*2,base_c,bilinear)
    
        self.outconv=OutConv(base_c,num_classes)
    

    def forward(self,x):
        x1=self.in_conv(x)
        x2=self.down1(x1)
        x3=self.down2(x2)
        x4=self.down3(x3)
        x5=self.down4(x4)
        x=self.up1(x5,x4)
        x=self.up2(x,x3)
        x=self.up3(x,x2)
        x=self.up4(x,x1)
        logits=self.outconv(x)

        # 以字典形式输出[N,C=num_classes,H,W]
        return {'out' : logits}



class DoubleConv(nn.Sequential):
    def __init__(self,in_channels,out_channels,mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv,self).__init__(
            nn.Conv2d(in_channels,mid_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Down(nn.Sequential):
    def __init__(self,in_channels,out_channels):
        super(Down,self).__init__(
            #池化+双层卷积为一个模块
            nn.MaxPool2d(2,stride=2),
            DoubleConv(in_channels,out_channels)
        )


class Up(nn.Sequential):
    def __init__(self,in_channels,out_channels,bilinear=True):#bilinear双线性插值可选
        #由于要进行拼接等操作，所以自定义前向传播
        super(Up,self).__init__()
        #上采样（双线性插值/转置卷积）+双层卷积为一个模块
        if bilinear:
            self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            #这里是因为双线性插值不会改变channel大小，所以在卷积时得减半两次
            self.conv=DoubleConv(in_channels,out_channels,out_channels//2)
        else:
            self.up=nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
            #转置卷积会减半channel大小，这是原论文的方法
            self.conv=DoubleConv(in_channels,out_channels)
    
    #x1为输入，x2为待拼接特征，此为一个上采样+双卷积操作
    def forward(self, x1,x2):
        x1=self.up(x1)
        #需要在上采样后进行拼接操作

        # [N, C, H, W]这步操作是为了确保特征能在维度上对齐，从而顺利拼接
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])


        x = torch.cat([x2, x1], dim=1)#在C维度上拼接
        x=self.conv(x)
        return x
    
class OutConv(nn.Sequential):
    def __init__(self,in_channels,num_classes):
        super(OutConv,self).__init__(
            #仅仅是为了
            nn.Conv2d(in_channels,num_classes,kernel_size=1)
        )

