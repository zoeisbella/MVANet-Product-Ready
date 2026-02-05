"""
MVANet 模型验证和推理脚本
用于验证模型在自定义数据上的表现
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
from model.MVANet import MVANet
import cv2


class ModelValidator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        """加载模型"""
        print(f"Loading model from {self.model_path}")
        
        # 初始化模型
        model = MVANet()
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 根据权重文件格式加载
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'state_dict_ema' in checkpoint:
                model.load_state_dict(checkpoint['state_dict_ema'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        return model
    
    def preprocess_image(self, image_path):
        """预处理单张图像"""
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor, original_size
    
    def postprocess_mask(self, output_tensor, original_size):
        """后处理分割结果"""
        # 应用sigmoid激活函数
        prob_map = torch.sigmoid(output_tensor)
        
        # 转换为numpy数组
        mask = prob_map.squeeze(0).squeeze(0).cpu().detach().numpy()
        
        # 应用阈值
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        # 转换为PIL图像并恢复原始尺寸
        mask_image = Image.fromarray(binary_mask.astype(np.uint8), mode='L')
        mask_image = mask_image.resize(original_size, Image.BILINEAR)
        
        return mask_image, mask
    
    def predict_single(self, image_path):
        """对单张图像进行预测"""
        # 预处理
        input_tensor, original_size = self.preprocess_image(image_path)
        
        # 推理
        with torch.no_grad():
            # 根据模型要求调整输入格式
            # MVANet需要多视角输入，这里简化处理
            glb = F.interpolate(input_tensor, scale_factor=0.5, mode='bilinear', align_corners=False)
            
            # 模拟多视角输入
            # 这里使用图像切块的方式
            batch_size, channels, height, width = input_tensor.shape
            half_height, half_width = height // 2, width // 2
            
            # 分割为四个局部视图
            loc1 = input_tensor[:, :, :half_height, :half_width]
            loc2 = input_tensor[:, :, :half_height, half_width:]
            loc3 = input_tensor[:, :, half_height:, :half_width]
            loc4 = input_tensor[:, :, half_height:, half_width:]
            
            # 组合输入
            combined_input = torch.cat([
                loc1, loc2, loc3, loc4,  # 局部视图
                glb  # 全局视图
            ], dim=0).unsqueeze(1)  # 添加批次维度
            
            # 模型推理
            output = self.model(combined_input)
            
            # 如果输出是列表，取最后一个
            if isinstance(output, (list, tuple)):
                output = output[-1]
        
        # 后处理
        mask_image, raw_mask = self.postprocess_mask(output[-1:], original_size)  # 取全局视图的结果
        
        return mask_image, raw_mask
    
    def evaluate_on_dataset(self, image_dir, output_dir="./results"):
        """在数据集上评估模型"""
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = [
            f for f in image_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ]
        
        print(f"Evaluating on {len(image_files)} images...")
        
        results = []
        
        for i, img_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {img_file.name}")
            
            try:
                # 预测
                mask_image, raw_mask = self.predict_single(img_file)
                
                # 保存结果
                mask_output_path = output_dir / f"mask_{img_file.stem}.png"
                mask_image.save(mask_output_path)
                
                # 保存原始概率图
                prob_output_path = output_dir / f"prob_{img_file.stem}.npy"
                np.save(prob_output_path, raw_mask)
                
                results.append({
                    'image': str(img_file),
                    'mask_saved': str(mask_output_path),
                    'success': True
                })
                
            except Exception as e:
                print(f"Error processing {img_file.name}: {str(e)}")
                results.append({
                    'image': str(img_file),
                    'success': False,
                    'error': str(e)
                })
        
        # 保存结果摘要
        summary_path = output_dir / "evaluation_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation completed! Results saved to {output_dir}")
        return results


def visualize_results(image_path, mask_path, output_path):
    """可视化结果"""
    # 加载原图和分割结果
    original = Image.open(image_path)
    mask = Image.open(mask_path)
    
    # 调整mask尺寸以匹配原图
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.BILINEAR)
    
    # 转换为numpy数组
    original_np = np.array(original)
    mask_np = np.array(mask)
    
    # 创建叠加图像
    overlay = original_np.copy()
    overlay[mask_np > 127] = [255, 0, 0]  # 红色标记分割区域
    
    # 创建可视化图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay Result')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")


def run_validation_demo():
    """运行验证演示"""
    print("=" * 60)
    print("MVANet 模型验证演示")
    print("=" * 60)
    
    # 检查权重文件
    weights_path = "./weights/mvanet.pth"
    if not os.path.exists(weights_path):
        print(f"Warning: Weight file not found at {weights_path}")
        print("Please download the pre-trained weights first.")
        return
    
    # 初始化验证器
    validator = ModelValidator(weights_path)
    
    # 使用示例数据进行验证
    sample_images_dir = "./sample_data/images"
    
    if not os.path.exists(sample_images_dir):
        print("Sample data not found, creating some test images...")
        from data_processor import prepare_sample_data
        prepare_sample_data()
    
    # 在样本数据上进行评估
    results = validator.evaluate_on_dataset(sample_images_dir)
    
    # 可视化第一个结果
    sample_image = Path(sample_images_dir) / "sample_0.jpg"
    sample_mask = Path("./results") / "mask_sample_0.png"
    
    if sample_mask.exists():
        vis_output = Path("./results") / "visualization_sample_0.png"
        visualize_results(sample_image, sample_mask, vis_output)
    
    print("\n模型验证能力验证完成！")
    print("- 可以在自定义数据上运行推理")
    print("- 支持批量处理")
    print("- 提供可视化结果")
    print("- 包含错误处理机制")


if __name__ == "__main__":
    run_validation_demo()