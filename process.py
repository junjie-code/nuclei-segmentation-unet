import cv2
import numpy as np
import os

def bio_preprocess(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化：简化，细胞核常在灰度上明显
    blurred = cv2.medianBlur(gray, 5)  # 中值滤波：去除盐椒噪声（显微镜常见）
    # CLAHE：增强局部对比，细胞核边缘更锐利
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    return enhanced

# 测试：处理一个图像并保存/显示
sample_id = 'ff599c7301daa1f783924ac8cbe3ce7b42878f15a39c2d19659189951f540f48'
data_dir = '/home/junjieli/pytorch/CellSeg/stage1_train'
image_path = os.path.join(data_dir, sample_id, 'images', sample_id + '.png')

processed = bio_preprocess(image_path)
cv2.imwrite('processed_example.png', processed)  # 保存到当前目录

# 显示对比（用matplotlib）
original_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.title('Original Gray')
plt.imshow(original_gray, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Processed')
plt.imshow(processed, cmap='gray')
plt.show()