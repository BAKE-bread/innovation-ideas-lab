# CBAM U-Net 宠物图像分割项目

数据集名称: Oxford-IIIT Pet Dataset

内容: 包含 37 个品种的猫狗图像（共 7390 张）及其对应的 trimap 标注。

链接：https://www.kaggle.com/datasets/julinmaloof/the-oxfordiiit-pet-dataset

评价指标: Dice Score

> 注意：本模型的目标是二元分类，代码已将原始标签转换。

## 使用方法

1.  下载数据集，将它放到 `data` 文件夹中。
2.  运行 `train.py` ，获得 `.pth` 模型文件。
3.  如果需要用于对无标签的用户图像进行预测，请运行 `predict.py` ；如果需要用于评估模型在单张有标签的测试图像上的性能，请运行 `eval.py` 。

