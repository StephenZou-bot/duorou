# 多肉植物分类项目

本项目旨在利用深度学习模型（如AlexNet、VGG、ResNet）对多肉植物进行分类。通过该项目，您将学习如何准备数据集、选择和训练模型、评估模型性能，并最终部署模型以实现自动化多肉植物分类。

## 项目背景

多肉植物种类繁多，形态各异，手动分类工作量大且易出错。通过计算机视觉和深度学习技术，我们可以自动化这一过程，提高分类效率和准确性。本项目选用常见的深度学习模型AlexNet、VGG和ResNet来进行多肉植物的分类任务。

## 数据集准备

1. **数据来源**：从多个公开的多肉植物图像数据集收集图像，包括不同种类和形态的多肉植物图片。
2. **数据预处理**：对图像进行统一尺寸调整、数据增强（如旋转、缩放、翻转）以及归一化处理。
3. **数据划分**：将数据集分为训练集、验证集和测试集。

## 模型选择

1. **AlexNet**：经典的卷积神经网络模型，结构简单，适合初学者和小型数据集。
2. **VGG**：深层卷积神经网络模型，通过增加网络深度来提高模型性能。
3. **ResNet**：残差网络，通过引入残差模块解决深层网络的退化问题，性能优异。

## 模型训练

1. **环境配置**：
    - Python 3.x
    - TensorFlow 或 PyTorch
    - CUDA（可选，用于GPU加速）
2. **训练过程**：
    - 数据加载与预处理
    - 模型构建
    - 选择损失函数和优化器
    - 模型训练（包括超参数调优，如学习率、批量大小等）
    - 训练过程中的监控与调整
