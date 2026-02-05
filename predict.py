import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import random

# ===== U-Net 模型定义（已修正 pooling 错误）=====
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(n_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder（已全部修正）
        d1 = self.down1(x)
        p1 = self.pool(d1)
        d2 = self.down2(p1)
        p2 = self.pool(d2)
        d3 = self.down3(p2)
        p3 = self.pool(d3)
        d4 = self.down4(p3)
        p4 = self.pool(d4)    # ← 这里修正了！原来错写成 self.pool(p4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        u1 = self.up1(b)
        cat1 = torch.cat([d4, u1], dim=1)
        c1 = self.conv1(cat1)

        u2 = self.up2(c1)
        cat2 = torch.cat([d3, u2], dim=1)
        c2 = self.conv2(cat2)

        u3 = self.up3(c2)
        cat3 = torch.cat([d2, u3], dim=1)
        c3 = self.conv3(cat3)

        u4 = self.up4(c3)
        cat4 = torch.cat([d1, u4], dim=1)
        c4 = self.conv4(cat4)

        out = self.out(c4)
        return out

# 预处理函数（和训练完全一致）
def bio_preprocess(img):
    blurred = cv2.medianBlur(img, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    return enhanced

# ===== 配置路径（请确认正确）=====
test_data_dir = '/home/junjieli/pytorch/CellSeg/stage1_test'  # 测试集路径
model_path = 'model/unet_nuclei_epoch_20.pth'  # 模型文件（当前目录或完整路径）

if not os.path.exists(test_data_dir):
    raise ValueError(f"测试集文件夹不存在: {test_data_dir}")
if not os.path.exists(model_path):
    raise ValueError(f"模型文件不存在: {model_path}")

# 加载模型
device = torch.device('cpu')
model = UNet(n_channels=1, n_classes=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# 获取测试样本
sample_ids = [sid for sid in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, sid))]
selected_ids = random.sample(sample_ids, min(8, len(sample_ids)))

# 输出文件夹
output_dir = 'test_prediction_results'
os.makedirs(output_dir, exist_ok=True)

print("开始在测试集上预测并保存结果...")

for i, sample_id in enumerate(selected_ids):
    img_path = os.path.join(test_data_dir, sample_id, 'images', sample_id + '.png')
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    processed = bio_preprocess(original)
    processed_resized = cv2.resize(processed, (256, 256))
    img_tensor = torch.from_numpy(processed_resized / 255.0).float().unsqueeze(0).unsqueeze(0)

    # 预测
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.sigmoid(output) > 0.5
        pred_mask = pred.squeeze().cpu().numpy() * 255
        pred_mask = pred_mask.astype(np.uint8)

    # 原图 resize
    original_resized = cv2.resize(original, (256, 256))

    # 拼接并保存
    combined = np.hstack([original_resized, pred_mask])
    save_path = os.path.join(output_dir, f'test_sample_{i+1}_{sample_id[:8]}.png')
    cv2.imwrite(save_path, combined)
    print(f"保存: {save_path}  （左：原图，右：预测细胞核）")

print("\n完成！所有测试集预测结果已保存到 test_prediction_results 文件夹。")