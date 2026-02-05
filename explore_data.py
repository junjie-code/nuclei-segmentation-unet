import cv2
import os
import matplotlib.pyplot as plt

# 数据路径（修改成你的实际路径）
data_dir = '/home/junjieli/pytorch/CellSeg/stage1_train'  # 用你的用户名替换 'junjieli'

# 选一个样本文件夹（随便挑一个ID）
sample_id = 'ff3e512b5fb860e5855d0c05b6cf5a6bcc7792e4be1f0bdab5a00af0e18435c0'  # 从ls中挑一个
sample_path = os.path.join(data_dir, sample_id)

# 加载原图
image_path = os.path.join(sample_path, 'images', sample_id + '.png')
original_img = cv2.imread(image_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # 转RGB便于显示

# 加载所有masks（多个，因为一个图可能有多个细胞核）
mask_dir = os.path.join(sample_path, 'masks')
masks = []
for mask_file in os.listdir(mask_dir):
    mask_path = os.path.join(mask_dir, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 灰度读
    masks.append(mask)

# 合并所有masks成一个（叠加，白色区域是所有细胞核）
combined_mask = masks[0]
for m in masks[1:]:
    combined_mask = cv2.bitwise_or(combined_mask, m)  # 或操作合并

# 显示：原图、合并mask
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(original_img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Combined Mask')
plt.imshow(combined_mask, cmap='gray')
plt.axis('off')

plt.show()
# 统计：总样本数、平均图像大小
num_samples = len(os.listdir(data_dir))
print(f'Total samples: {num_samples}')

heights, widths = [], []
for sample_id in os.listdir(data_dir):
    img_path = os.path.join(data_dir, sample_id, 'images', sample_id + '.png')
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    heights.append(h)
    widths.append(w)

avg_h = sum(heights) / num_samples
avg_w = sum(widths) / num_samples
print(f'Average image size: {avg_h:.0f} x {avg_w:.0f}')