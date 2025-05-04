import os
import shutil
import SimpleITK as sitk


def check_and_copy_folders(source_folder, dest_folder):
    # 获取源文件夹中所有子文件夹的列表
    subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]

    for folder in subfolders:
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)

        # 检查SEG.nii.gz文件是否存在且包含值为1的像素
        seg_file_path = os.path.join(item_path, 'SEG.nii.gz')
        if os.path.exists(seg_file_path):
            seg_image = sitk.ReadImage(seg_file_path)
            seg_array = sitk.GetArrayFromImage(seg_image)
            if 1 in seg_array:
                # 如果存在值为1的像素，则将文件夹复制到目标文件夹中
                folder_name = os.path.basename(folder)
                dest_path = os.path.join(dest_folder, folder_name)
                shutil.copytree(folder, dest_path)
                print(f"Folder '{folder_name}' copied to '{dest_folder}'")


# 源文件夹和目标文件夹的路径
source_folder = '/data3/zoe/Data/autoPET/FDG-PET-CT-Lesions/'
dest_folder = '/data3/zoe/Data/autoPET/PET-CT/'

# 执行函数
check_and_copy_folders(source_folder, dest_folder)
