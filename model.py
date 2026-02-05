import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):  # 两个卷积层块
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):  # 输入1通道（灰度），输出1通道（二值mask）
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Downsampling（左边）
        self.down1 = DoubleConv(n_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)  # 池化减尺寸

        # Bottleneck（底部）
        self.bottleneck = DoubleConv(512, 1024)

        # Upsampling（右边）
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 转置卷积放大
        self.upconv1 = DoubleConv(1024, 512)  # 跳连接：1024=512(上采样)+512(左对应)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv4 = DoubleConv(128, 64)

        # 输出层
        self.outconv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Down
        d1 = self.down1(x)
        p1 = self.pool(d1)
        d2 = self.down2(p1)
        p2 = self.pool(d2)
        d3 = self.down3(p2)
        p3 = self.pool(d3)
        d4 = self.down4(p3)
        p4 = self.pool(d4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Up（注意跳连接：cat融合左边对应层）
        u1 = self.up1(b)
        cat1 = torch.cat((d4, u1), dim=1)  # 通道维cat
        uc1 = self.upconv1(cat1)
        u2 = self.up2(uc1)
        cat2 = torch.cat((d3, u2), dim=1)
        uc2 = self.upconv2(cat2)
        u3 = self.up3(uc2)
        cat3 = torch.cat((d2, u3), dim=1)
        uc3 = self.upconv3(cat3)
        u4 = self.up4(uc3)
        cat4 = torch.cat((d1, u4), dim=1)
        uc4 = self.upconv4(cat4)

        # 输出（sigmoid在loss中或后处理）
        out = self.outconv(uc4)
        return out

# 测试模型
if __name__ == '__main__':
    model = UNet()
    print(model)
    # 假输入：批次1，1通道，256x256
    dummy_input = torch.randn(1, 1, 256, 256)
    output = model(dummy_input)
    print(output.shape)  # 应为 [1, 1, 256, 256]