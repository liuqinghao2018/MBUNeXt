import os

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

# A6000 paths
nnUNet_raw = "/data3/zoe/Code/MBUNeXt/nnUNetData/MBUNeXt_raw"
nnUNet_preprocessed = "/data3/zoe/Code/MBUNeXt/nMBUNeXtData/MBUNeXt_preprocessed"
nnUNet_results = "/data3/zoe/Code/MBUNeXt/MBUNeXtData/MBUNeXt_trained_models"


if nnUNet_raw is None:
    print("MBUNeXt_raw is not defined and nnU-Net can only be used on data for which preprocessed files are already present on your system.")

if nnUNet_preprocessed is None:
    print("MBUNeXt_preprocessed is not defined and nnU-Net can not be used for preprocessing or training.")

if nnUNet_results is None:
    print("MBUNeXt_results is not defined and MBUNeXt cannot be used for training or inference.")
