# **MBUNeXt: Multibranch Encoder Aggregation Network for Multimodal Brain Tumor Segmentation**

This repository contains the official implementation of **MBUNeXt** , a novel U-shaped network for multimodal brain tumor segmentation. The method introduces three key modules:

- **Multimodal Feature Attention (MFA)** : Emphasizes shared features across modalities and filters redundancy via Gaussian-modulated attention.
- **Multibranch Encoder Aggregation (MEA)** : Leverages Transformer-based global modeling to fuse multimodal features.
- **Large-kernel Convolution Skip-connection (LCS)** : Captures multi-scale lesion details through depthwise separable convolutions.

Our approach achieves state-of-the-art performance on BraTS2019, BraTS2021, and BraTS-Africa2024 datasets, with robustness in low-quality imaging scenarios.

## 🧰 Dependencies

```
# Python 3.8+, PyTorch 1.10+, CUDA 11.6+  
pip install torch torchvision numpy SimpleITK scikit-learn tqdm  
```

## 🚀 Training

```
# Train on BraTS-Africa2024 
python run_training_db.py
```

## 📝 Note

This codebase is **built upon nnUNetV2**([https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) ), leveraging its dynamic inference and modular design. We extend it with custom fusion strategies and Transformer integration. For nnUNetV2-specific workflows (e.g., data organization, export), refer to their [official documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/README.md) .
