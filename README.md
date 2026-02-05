# Automated Nuclei Segmentation with U-Net

细胞核分割项目，使用 PyTorch 实现 U-Net，在 kaggle Data Science Bowl 数据集上训练，预测。

### 项目简介
- 数据：Kaggle Data Science Bowl 细胞核图像
- 预处理：OpenCV（灰度化 + CLAHE + 中值滤波）
- 模型： U-Net
- 训练：Google Colab + T4 GPU，20 epochs，Loss 降到 0.086
- 效果：在测试集上能较好地分割细胞核

### 数据获取
1. 去 Kaggle 下载数据集：https://www.kaggle.com/c/data-science-bowl-2018/data
2. 解压 `stage1_train.zip`（训练）和 `stage1_test.zip`（测试）

（数据体积大，不上传到仓库）

### 文件说明
- model.py：U-Net 模型定义
- predict.py：在测试集上预测
- train.ipynb：Colab 训练代码及结果记录
- test_prediction_results/：测试集预测结果示例（左：原图，右：预测细胞核）
<img width="512" height="256" alt="test_sample_1_1cdbfee1" src="https://github.com/user-attachments/assets/256b59d7-1aa6-47ec-a2ad-a683caa20730" />
<img width="512" height="256" alt="test_sample_2_472b1c5f" src="https://github.com/user-attachments/assets/0c1ab1cd-78f6-4e57-9ac9-607e707ce6b5" />
<img width="512" height="256" alt="test_sample_4_f5effed2" src="https://github.com/user-attachments/assets/5b4c6113-8c02-4cc9-ab98-c221f243300c" />
<img width="512" height="256" alt="test_sample_3_da6c5934" src="https://github.com/user-attachments/assets/60ff6a3c-a8b9-4919-ae40-188cf94dbf5c" />
