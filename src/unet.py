from typing import Dict                         
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义的DoubleConv，这是因为我们的UNet中有很多这样的结构，所以定义一个类来简化代码
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):  
        # mid_channels是中间层的通道数，如果没有指定，就和out_channels一样
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            # 说明一下，原论文中没有对第一层的输入进行BN，但是在实际使用中，发现加上BN效果更好，而且没有对其进行padding
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),  # 这里使用的是2d的BN，因为是对图片进行处理
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):  # 这里定义的Down是进行下采样的操作
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),  # 这里的最大池化层的kernel_size和stride都是2
            DoubleConv(in_channels, out_channels)  # 这里直接使用定义的DoubleConv
        )


class Up(nn.Module):  # 这里的Up是进行上采样的操作
    def __init__(self, in_channels, out_channels, bilinear=True):   # 对参数进行说明，bilinear是指是否使用双线性插值
        super(Up, self).__init__()
        if bilinear:
            # 对于双线性插值，使用的是nn.Upsample，这里使用的是scale_factor=2，也就是说的图片的大小是原来的2倍
            # 对于align_corners，这个参数是指是否对齐角点，这里是True，也就是说的角点是对齐的
            # 对于conv，使用的是定义的DoubleConv,最后一个输出通道数是in_channels // 2
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:  
        # 这里的x1是上一层的输出，x2是的下采样层的输出
        x1 = self.up(x1)  # 先对上一层的输出进行上采样
        # x1对应的是shape是[N, C, H, W] 
        diff_y = x2.size()[2] - x1.size()[2]  # 这里计算出两个的高度差
        diff_x = x2.size()[3] - x1.size()[3]  # 这里计算出两个的宽度差

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        # 这里对x1进行padding，使得x1和x2的大小一样
        x = torch.cat([x2, x1], dim=1)  # 这里将x1和x2进行拼接
        x = self.conv(x)  # 最后使用DoubleConv进行处理
        return x


class OutConv(nn.Sequential):  # 这里定义的是最后的输出层
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)  # 使用1x1的卷积核
        )


class UNet(nn.Module):
    # 这里定义的是UNet的结构
    def __init__(self,
                 in_channels: int = 1,  # 这里的in_channels是指的输入的通道数
                 num_classes: int = 2,  # 这里的num_classes是指的分类的类别数
                 bilinear: bool = True,  # 是否使用双线性插值
                 base_c: int = 64):  # 这里的base_c是指的最开始的通道数
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)  # 对于最开始的输入，使用的是DoubleConv，输出通道数是base_c
        self.down1 = Down(base_c, base_c * 2)  # 这里的输出通道数是base_c * 2
        self.down2 = Down(base_c * 2, base_c * 4) # 这里的输出通道数是base_c * 4
        self.down3 = Down(base_c * 4, base_c * 8)  # 这里的输出通道数是base_c * 8
        factor = 2 if bilinear else 1  # 这里的factor是指的下采样和上采样的通道数的比例，因为看我们的架构就可以知道这样操作可以直接进行拼接，会得到原来的通道数
        self.down4 = Down(base_c * 8, base_c * 16 // factor)  # 这里的输出通道数是base_c * 16 // factor
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)  # 这里的输出通道数是base_c * 8 // factor
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)  # 这里的输出通道数是base_c * 4 // factor
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)  # 这里的输出通道数是base_c * 2 // factor
        self.up4 = Up(base_c * 2, base_c, bilinear)  #  这里的输出通道数是base_c
        self.out_conv = OutConv(base_c, num_classes)  # 最后的输出层，输出通道数是num_classes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 这里的x是输入的图片，shape是[N, C, H, W]
        x1 = self.in_conv(x) # 先对输入的图片进行处理
        x2 = self.down1(x1)  # 然后进行下采样，经过四次双倍下采样，会得到[N, base_c * 16, H // 16, W // 16]
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)  # 这里进行上采样，经过四次双倍上采样，会得到[N, base_c, H, W]
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)  #最后使用输出层，得到最后的输出，logits的shape是[N, num_classes, H, W]

        return {"out": logits}  # 这里返回的是字典，因为我们的loss函数需要的是字典，所以这里返回的是字典
