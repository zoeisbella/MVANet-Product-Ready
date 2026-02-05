import asyncio
import base64
import io
import os
import time
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from torchvision import transforms
from fastapi.middleware.cors import CORSMiddleware


# Pydantic 模型定义
class PredictionRequest(BaseModel):
    image_base64: str
    threshold: float = 0.5


class PredictionResponse(BaseModel):
    success: bool
    mask_base64: Optional[str] = None
    message: str
    inference_time: float


# 配置类
class Config:
    # 模型配置
    MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "./weights/test_weights.pth")  # 使用测试权重文件
    MODEL_INPUT_SIZE = tuple(map(int, os.getenv("MODEL_INPUT_SIZE", "256,256").split(",")))
    DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")  # 如果没有GPU，会自动使用CPU
    
    # 服务器配置
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    # 图像处理配置
    IMAGE_MAX_SIZE = int(os.getenv("IMAGE_MAX_SIZE", "4096"))  # 最大图像尺寸
    THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))  # 分割阈值
    
    # 性能配置
    BATCH_SIZE_LIMIT = int(os.getenv("BATCH_SIZE_LIMIT", "1"))


# 单例模型加载器
class ModelLoader:
    _instance = None
    _model = None
    _device = torch.device(Config.DEVICE)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def load_model(self):
        if self._model is None:
            try:
                # 导入模型（根据新的目录结构，模型在models目录下）
                from ..models.MVANet import MVANet
                from ..models.SwinTransformer import SwinB
                
                # 临时修改SwinB函数以避免加载外部预训练权重
                from ..models import SwinTransformer as swin_module
                original_swinb = swin_module.SwinB
                
                # 创建一个不加载预训练权重的版本
                def swinb_no_pretrained(pretrained=True):
                    model = swin_module.SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=12)
                    return model
                
                # 替换原函数
                swin_module.SwinB = swinb_no_pretrained
                
                # 初始化模型
                self._model = MVANet()
                
                # 恢复原始函数
                swin_module.SwinB = original_swinb
                
                # 设置设备
                self._device = torch.device(Config.DEVICE)
                
                # 加载权重
                weights_path = Config.MODEL_WEIGHTS_PATH  # 使用配置中的权重路径
                checkpoint = torch.load(weights_path, map_location=self._device)
                
                # 处理权重加载，考虑键名可能不匹配的情况
                if 'model' in checkpoint:
                    model_state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    model_state_dict = checkpoint['state_dict']
                elif 'state_dict_ema' in checkpoint:
                    model_state_dict = checkpoint['state_dict_ema']
                else:
                    # 如果整个checkpoint就是state_dict
                    if len(checkpoint) == 1 and 'model' in checkpoint:
                        model_state_dict = checkpoint['model']
                    else:
                        model_state_dict = checkpoint
                
                # 尝试加载模型权重，如果键名不匹配则进行处理
                try:
                    self._model.load_state_dict(model_state_dict)
                except RuntimeError as e:
                    print(f"Direct load failed: {e}")
                    # 尝试移除前缀或处理键名不匹配
                    model_dict = self._model.state_dict()
                    new_state_dict = {}
                    
                    for k, v in model_state_dict.items():
                        # 移除 'module.' 前缀（如果存在）
                        new_k = k
                        if k.startswith('module.'):
                            new_k = k[7:]
                        
                        # 检查是否存在于模型中
                        if new_k in model_dict and model_dict[new_k].shape == v.shape:
                            new_state_dict[new_k] = v
                        elif k in model_dict and model_dict[k].shape == v.shape:
                            new_state_dict[k] = v
                        else:
                            # 尝试添加'backbone.'前缀（对于Swin Transformer层）
                            backbone_k = 'backbone.' + new_k
                            if backbone_k in model_dict and model_dict[backbone_k].shape == v.shape:
                                new_state_dict[backbone_k] = v
                            # 尝试移除'backbone.'前缀
                            elif new_k.startswith('backbone.'):
                                stripped_k = new_k[9:]  # 移除'backbone.'前缀
                                if stripped_k in model_dict and model_dict[stripped_k].shape == v.shape:
                                    new_state_dict[stripped_k] = v
                    
                    print(f"Loading {len(new_state_dict)} out of {len(model_dict)} keys")
                    self._model.load_state_dict(new_state_dict, strict=False)
                
                self._model.to(self._device)
                self._model.eval()
                
                print(f"Model loaded successfully on {self._device}")
            except FileNotFoundError:
                print(f"Model weights file not found: {Config.MODEL_WEIGHTS_PATH}")
                print("Please ensure the weights file is located at: ./weights/mvanet.pth")
                raise
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise
        
        return self._model

    def get_model(self):
        if self._model is None:
            return self.load_model()
        return self._model


# 预处理和后处理工具类
class ImageProcessor:
    def __init__(self):
        # 定义预处理变换
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 根据模型输入要求调整
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        预处理：尺寸调整、归一化，确保尺寸符合模型要求
        使用固定的512x512尺寸以确保与模型完全兼容
        """
        # 保存原始尺寸
        self.original_size = image.size  # 保存原始尺寸以供后处理使用
        
        # 转换为RGB（如果输入不是RGB）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 使用固定的512x512尺寸，这是Swin Transformer的标准输入尺寸
        # 确保能被32、64、128等整除，避免所有尺寸相关问题
        target_size = 512
        
        # 定义预处理变换
        transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),  # 固定为512x512
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 应用变换
        input_tensor = transform(image).unsqueeze(0)  # 添加批次维度
        
        return input_tensor
    
    def postprocess(self, output_tensor: torch.Tensor, original_size: tuple) -> Image.Image:
        """
        后处理：阈值化、形态学操作和尺寸还原，以提高分割质量
        """
        # 将输出转换为概率图
        if output_tensor.shape[1] == 1:  # 如果输出是单通道
            prob_map = torch.sigmoid(output_tensor)
        else:  # 如果输出是多通道
            prob_map = F.softmax(output_tensor, dim=1)[:, 1:, :, :]  # 取前景类别的概率
        
        # 转换为numpy数组
        mask = prob_map.squeeze(0).squeeze(0).cpu().detach().numpy()
        
        # 确保mask值在0-1范围内
        mask = np.clip(mask, 0, 1)
        
        # 使用更高精度的浮点运算进行后续处理
        import cv2
        # 调整到0-255范围用于OpenCV处理
        mask_scaled = (mask * 255).astype(np.uint8)
        
        # 应用高斯模糊以减少噪声
        mask_blur = cv2.GaussianBlur(mask_scaled, (5, 5), 0)
        
        # 使用自适应阈值而非固定阈值，提高边缘细节
        _, mask_thresh = cv2.threshold(mask_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作去除小的噪声区域
        kernel = np.ones((3,3), np.uint8)
        mask_open = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 边缘平滑
        mask_smooth = cv2.medianBlur(mask_close, 5)
        
        # 转换为PIL图像
        mask_image = Image.fromarray(mask_smooth, mode='L')
        
        # 恢复原始尺寸
        mask_image = mask_image.resize(original_size, Image.LANCZOS)  # 使用LANCZOS获得更好的缩放质量
        
        return mask_image


# 初始化应用
app = FastAPI(
    title="MVANet Segmentation API",
    description="High-performance image segmentation API using MVANet",
    version="1.0.0"
)

# 添加CORS中间件以支持跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应限制为具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化单例模型加载器和图像处理器
model_loader = ModelLoader()
processor = ImageProcessor()


@app.on_event("startup")
async def startup_event():
    """
    应用启动事件：预加载模型
    """
    print("Loading MVANet model...")
    model_loader.load_model()
    print("Model loaded successfully!")


@app.get("/")
async def root():
    """
    根路径健康检查
    """
    return {"message": "MVANet Segmentation API is running"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    图像分割预测接口
    """
    start_time = time.time()
    
    try:
        # 验证文件类型
        content_type = file.content_type
        if not content_type or not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        
        # 读取图像
        contents = await file.read()
        image_stream = io.BytesIO(contents)
        
        try:
            # 打开图像
            image = Image.open(image_stream)
            original_size = image.size
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # 预处理
        input_tensor = processor.preprocess(image)
        input_tensor = input_tensor.to(model_loader._device)
        
        # 获取模型
        model = model_loader.get_model()
        
        # 推理 - 使用固定的512x512尺寸，确保与模型完全兼容
        with torch.no_grad():
            # 直接进行推理，因为预处理已经确保尺寸为512x512
            output = model(input_tensor)
            
            # 如果模型返回多个输出，取主要输出
            if isinstance(output, (list, tuple)):
                # 通常最后一个输出是最终结果
                output = output[-1]
            
            # 提取全局视图的结果（对应原始尺寸的分割结果）
            # MVANet通常输出5个结果，最后一个是融合结果
            if output.shape[0] == 5:  # 如果输出包含5个视图的结果
                main_output = output[-1:]  # 取最后一个融合结果
            else:
                main_output = output
        
        # 后处理
        mask_image = processor.postprocess(main_output, original_size)
        
        # 转换为Base64
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # 计算推理时间
        inference_time = time.time() - start_time
        
        print(f"Inference completed in {inference_time:.3f}s")
        
        return PredictionResponse(
            success=True,
            mask_base64=mask_base64,
            message="Segmentation successful",
            inference_time=inference_time
        )
    
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        # 记录错误
        print(f"Prediction error: {str(e)}")
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        return PredictionResponse(
            success=False,
            mask_base64=None,
            message=f"Error during prediction: {str(e)}",
            inference_time=total_time
        )


# 在ImageProcessor类中添加分块函数
def split_to_patches(self, x):
    """将图像分割为4个局部视图 (参考模型代码中的image2patches函数)"""
    x = x.squeeze(0)  # 移除batch维度
    x = x.unflatten(1, (3, 1, 1)).unfold(2, x.shape[-2]//2, x.shape[-2]//2).unfold(3, x.shape[-1]//2, x.shape[-1]//2)
    x = x.contiguous().view(4, 3, x.shape[-2], x.shape[-1])
    return x.unsqueeze(1)  # 重新添加batch维度


# 为ImageProcessor类动态添加方法
ImageProcessor.split_to_patches = split_to_patches


# 添加健康检查端点
@app.get("/health")
async def health_check():
    """
    健康检查端点
    """
    return {
        "status": "healthy",
        "model_loaded": model_loader._model is not None
    }


@app.post("/predict_and_download", response_class=StreamingResponse)
async def predict_and_download(file: UploadFile = File(...)):
    """
    分割图像并直接下载结果
    """
    start_time = time.time()
    
    try:
        # 验证文件类型
        content_type = file.content_type
        if not content_type or not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        
        # 读取图像
        contents = await file.read()
        image_stream = io.BytesIO(contents)
        
        try:
            # 打开图像
            image = Image.open(image_stream)
            original_size = image.size
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # 预处理
        input_tensor = processor.preprocess(image)
        input_tensor = input_tensor.to(model_loader._device)
        
        # 获取模型
        model = model_loader.get_model()
        
        # 推理 - 使用固定的512x512尺寸，确保与模型完全兼容
        with torch.no_grad():
            # 直接进行推理，因为预处理已经确保尺寸为512x512
            output = model(input_tensor)
            
            # 如果模型返回多个输出，取主要输出
            if isinstance(output, (list, tuple)):
                # 通常最后一个输出是最终结果
                output = output[-1]
            
            # 提取全局视图的结果（对应原始尺寸的分割结果）
            # MVANet通常输出5个结果，最后一个是融合结果
            if output.shape[0] == 5:  # 如果输出包含5个视图的结果
                main_output = output[-1:]  # 取最后一个融合结果
            else:
                main_output = output
        
        # 后处理
        mask_image = processor.postprocess(main_output, original_size)
        
        # 创建内存流用于返回图像
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        buffer.seek(0)
        
        print(f"Inference completed in {time.time() - start_time:.3f}s")
        
        # 返回图像文件
        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=segmentation_result.png"
            }
        )
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


@app.post("/predict_json", response_model=PredictionResponse)
async def predict_json(request: PredictionRequest):
    """
    通过JSON数据进行图像分割预测接口（接受base64编码的图像）
    """
    start_time = time.time()
    
    try:
        # 从base64解码图像
        try:
            image_bytes = base64.b64decode(request.image_base64)
            image_stream = io.BytesIO(image_bytes)
            image = Image.open(image_stream)
            original_size = image.size
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
        # 预处理
        input_tensor = processor.preprocess(image)
        input_tensor = input_tensor.to(model_loader._device)
        
        # 获取模型
        model = model_loader.get_model()
        
        # 推理 - 使用固定的512x512尺寸，确保与模型完全兼容
        with torch.no_grad():
            # 直接进行推理，因为预处理已经确保尺寸为512x512
            output = model(input_tensor)
            
            # 如果模型返回多个输出，取主要输出
            if isinstance(output, (list, tuple)):
                # 通常最后一个输出是最终结果
                output = output[-1]
            
            # 提取全局视图的结果（对应原始尺寸的分割结果）
            # MVANet通常输出5个结果，最后一个是融合结果
            if output.shape[0] == 5:  # 如果输出包含5个视图的结果
                main_output = output[-1:]  # 取最后一个融合结果
            else:
                main_output = output
        
        # 后处理
        mask_image = processor.postprocess(main_output, original_size)
        
        # 转换为Base64
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # 计算推理时间
        inference_time = time.time() - start_time
        
        print(f"Inference completed in {inference_time:.3f}s")
        
        return PredictionResponse(
            success=True,
            mask_base64=mask_base64,
            message="Segmentation successful",
            inference_time=inference_time
        )
    
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        # 记录错误
        print(f"Prediction error: {str(e)}")
        return PredictionResponse(
            success=False,
            mask_base64=None,
            message=f"Error during prediction: {str(e)}",
            inference_time=time.time() - start_time
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)