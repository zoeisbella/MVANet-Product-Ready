"""
MVANet 数据预处理和验证脚本
用于准备自定义数据集并验证模型在新数据上的表现
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import json


class CustomDataset(Dataset):
    """
    自定义数据集类，用于加载图像和对应的分割标签
    """
    def __init__(self, image_dir, mask_dir=None, transform=None, input_size=(256, 256)):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.input_size = input_size
        
        # 获取所有图像文件
        self.image_files = [
            f for f in self.image_dir.iterdir() 
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ]
        
        # 检查对应的mask文件
        if self.mask_dir:
            self.mask_files = [
                f for f in self.mask_dir.iterdir()
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ]
        else:
            self.mask_files = None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_dir / self.image_files[idx]
        image = Image.open(img_path)
        
        # 转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 预处理图像
        if self.transform:
            image = self.transform(image)
        else:
            # 默认预处理
            transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)
        
        # 如果有mask，也加载它
        mask = None
        if self.mask_files:
            # 找到对应的mask文件（同名）
            mask_filename = self.image_files[idx].stem + self.image_files[idx].suffix
            mask_path = self.mask_dir / mask_filename
            
            if mask_path.exists():
                mask = Image.open(mask_path).convert('L')  # 灰度图
                mask = mask.resize(self.input_size)
                mask = np.array(mask)
                mask = torch.from_numpy(mask).float() / 255.0  # 归一化到[0,1]
            else:
                print(f"Warning: No corresponding mask found for {self.image_files[idx]}")
        
        return image, mask


def create_custom_dataset(image_folder, mask_folder=None, output_dir="./custom_dataset"):
    """
    创建自定义数据集，验证数据格式
    """
    print(f"Creating custom dataset from {image_folder}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建数据集
    dataset = CustomDataset(
        image_dir=image_folder,
        mask_dir=mask_folder
    )
    
    print(f"Dataset contains {len(dataset)} samples")
    
    # 验证数据集
    sample_img, sample_mask = dataset[0]
    print(f"Sample image shape: {sample_img.shape}")
    if sample_mask is not None:
        print(f"Sample mask shape: {sample_mask.shape}")
    
    # 保存一些基本信息
    dataset_info = {
        "num_samples": len(dataset),
        "image_shape": list(sample_img.shape),
        "has_masks": sample_mask is not None,
        "image_folder": str(image_folder),
        "mask_folder": str(mask_folder) if mask_folder else None
    }
    
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Dataset info saved to {os.path.join(output_dir, 'dataset_info.json')}")
    
    return dataset


def validate_dataset_format(dataset_path):
    """
    验证数据集格式是否符合要求
    """
    print(f"Validating dataset format in {dataset_path}")
    
    # 检查目录结构
    image_dir = Path(dataset_path) / "images"
    mask_dir = Path(dataset_path) / "masks"
    
    if not image_dir.exists():
        print(f"Error: Images directory {image_dir} does not exist")
        return False
    
    if not mask_dir.exists():
        print(f"Warning: Masks directory {mask_dir} does not exist")
    
    # 统计图像数量
    image_count = len([f for f in image_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    mask_count = len([f for f in mask_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]) if mask_dir.exists() else 0
    
    print(f"Found {image_count} images and {mask_count} masks")
    
    # 检查文件名匹配（如果有mask的话）
    if mask_dir.exists() and mask_count > 0:
        image_names = {f.stem for f in image_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
        mask_names = {f.stem for f in mask_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
        
        unmatched_images = image_names - mask_names
        unmatched_masks = mask_names - image_names
        
        if unmatched_images:
            print(f"Warning: {len(unmatched_images)} images without corresponding masks")
            if len(unmatched_images) < 10:
                print(f"Unmatched image names: {unmatched_images}")
        
        if unmatched_masks:
            print(f"Warning: {len(unmatched_masks)} masks without corresponding images")
            if len(unmatched_masks) < 10:
                print(f"Unmatched mask names: {unmatched_masks}")
    
    return True


def prepare_sample_data():
    """
    准备一些示例数据用于测试
    """
    print("Preparing sample data...")
    
    # 创建示例图像
    sample_dir = Path("./sample_data/images")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建一些示例图像
    for i in range(5):
        # 创建随机图像用于测试
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(sample_dir / f"sample_{i}.jpg")
    
    print(f"Created {len(list(sample_dir.iterdir()))} sample images")
    print("Sample data ready for testing")


def main():
    """
    主函数 - 演示数据处理能力
    """
    print("=" * 60)
    print("MVANet 数据处理能力验证")
    print("=" * 60)
    
    # 1. 准备示例数据
    prepare_sample_data()
    
    # 2. 验证数据集格式
    validate_dataset_format("./sample_data")
    
    # 3. 创建自定义数据集
    dataset = create_custom_dataset(
        image_folder="./sample_data/images",
        output_dir="./custom_dataset"
    )
    
    print("\n数据处理能力验证完成！")
    print("- 可以处理自定义图像数据")
    print("- 支持带标签和不带标签的数据集")
    print("- 包含数据验证和格式检查")
    print("- 为模型训练和推理做好准备")


if __name__ == "__main__":
    main()