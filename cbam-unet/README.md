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

## 项目文件夹结构
/cbam-unet/

├── data/

│   ├── images/

│   │   ├── Abyssinian_1.jpg

│   │   ├── ... (总共 7390 个图像)

│   └── annotations/

│   ...   ├── list.txt

│   ...   ├── trainval.txt   <-- 官方训练验证集 (3680 张)

│   ...   ├── test.txt       <-- 官方测试集 (3669 张)

│   ...   └── trimaps/

│   ...   ...   ├── Abyssinian_1.png

│   ...   ...   ├── ... (总共 7390 个标注)

│

├── best_cbam_unet_pets.pth.tar

├── eval.py

├── model.py

├── pet_dataset.py

├── predict.py

├── utils.py

└── train.py
