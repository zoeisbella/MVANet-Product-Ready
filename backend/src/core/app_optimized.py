import asyncio
import base64
import io
import time
import os
from typing import Optional
from contextlib import contextmanager

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from torchvision import transforms
from loguru import logger
import gc


# 配置类
class Config:
    # 模型配置
    MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "./weights/mvanet.pth")
    MODEL_INPUT_SIZE = tuple(map(int, os.getenv("MODEL_INPUT_SIZE", "256,256").split(",")))
    DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    # 服务器配置
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    # 图像处理配置
    IMAGE_MAX_SIZE = int(os.getenv("IMAGE_MAX_SIZE", "4096"))  # 最大图像尺寸
    THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))  # 分割阈值
    
    # 性能配置
    BATCH_SIZE_LIMIT = int(os.getenv("BATCH_SIZE_LIMIT", "1"))


# Pydantic 模型定义
class PredictionResponse(BaseModel):
    success: bool
    mask_base64: Optional[str] = None
    message: str
    inference_time: float
    original_size: Optional[tuple] = None
    processed_size: Optional[tuple] = None


# 显存管理装饰器
@contextmanager
def gpu_memory_cleanup():
    """GPU显存清理上下文管理器"""
    try:
        yield
    finally:
        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 触发垃圾回收
        gc.collect()


# 单例模型加载器
class ModelLoader:
    _instance = None
    _model = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def load_model(self):
        if self._model is None:
            try:
                logger.info(f"Loading model from {Config.MODEL_WEIGHTS_PATH}")
                
                # 导入模型
                from model.MVANet import MVANet
                
                # 初始化模型
                self._model = MVANet()
                
                # 设置设备
                self._device = torch.device(Config.DEVICE)
                
                # 检查权重文件是否存在
                weights_path = Config.MODEL_WEIGHTS_PATH
                if not os.path.exists(weights_path):
                    # 如果正式权重文件不存在，尝试使用测试权重
                    test_weights_path = "./weights/test_weights.pth"
                    if os.path.exists(test_weights_path):
                        weights_path = test_weights_path
                        logger.warning(f"Formal weights not found, using test weights: {test_weights_path}")
                    else:
                        logger.error(f"Model weights file not found: {Config.MODEL_WEIGHTS_PATH}")
                        raise FileNotFoundError(f"Model weights file not found: {Config.MODEL_WEIGHTS_PATH}")
                
                # 加载权重
                checkpoint = torch.load(
                    weights_path, 
                    map_location=self._device
                )
                
                # 如果权重是以state_dict形式保存
                if 'model' in checkpoint:
                    self._model.load_state_dict(checkpoint['model'])
                elif 'state_dict' in checkpoint:
                    self._model.load_state_dict(checkpoint['state_dict'])
                elif 'state_dict_ema' in checkpoint:
                    self._model.load_state_dict(checkpoint['state_dict_ema'])
                else:
                    # 如果是空的测试权重文件，跳过加载
                    if checkpoint:  # 避免加载空字典导致错误
                        self._model.load_state_dict(checkpoint)
                
                self._model.to(self._device)
                self._model.eval()
                
                logger.success(f"Model loaded successfully on {self._device}")
                
                # 显存信息
                if torch.cuda.is_available():
                    logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                    
            except FileNotFoundError:
                logger.error(f"Model weights file not found: {Config.MODEL_WEIGHTS_PATH}")
                raise
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise
        
        return self._model

    def get_model(self):
        if self._model is None:
            return self.load_model()
        return self._model

    def clear_cache(self):
        """清理模型缓存"""
        if self._model is not None:
            del self._model
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Model cache cleared")


# 预处理和后处理工具类
class ImageProcessor:
    def __init__(self):
        # 定义预处理变换
        self.transform = transforms.Compose([
            transforms.Resize(Config.MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def validate_image_size(self, image: Image.Image) -> bool:
        """验证图像尺寸是否超出限制"""
        width, height = image.size
        max_dimension = max(width, height)
        return max_dimension <= Config.IMAGE_MAX_SIZE
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        预处理图像
        """
        # 验证图像尺寸
        if not self.validate_image_size(image):
            raise ValueError(f"Image size exceeds limit of {Config.IMAGE_MAX_SIZE}px")
        
        # 转换为RGB（如果输入不是RGB）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 应用预处理
        tensor = self.transform(image)
        
        # 添加batch维度
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def postprocess(self, output_tensor: torch.Tensor, original_size: tuple) -> Image.Image:
        """
        后处理：阈值化和尺寸还原
        """
        # 将输出转换为概率图
        if output_tensor.shape[1] == 1:  # 如果输出是单通道
            prob_map = torch.sigmoid(output_tensor)
        else:  # 如果输出是多通道
            prob_map = F.softmax(output_tensor, dim=1)[:, 1:, :, :]  # 取前景类别的概率
        
        # 转换为numpy数组
        mask = prob_map.squeeze(0).squeeze(0).cpu().detach().numpy()
        
        # 应用阈值
        mask = (mask > Config.THRESHOLD).astype(np.uint8) * 255
        
        # 转换为PIL图像
        mask_image = Image.fromarray(mask.astype(np.uint8), mode='L')
        
        # 恢复原始尺寸
        mask_image = mask_image.resize(original_size, Image.BILINEAR)
        
        return mask_image

    def split_to_patches(self, x):
        """将图像分割为4个局部视图 (参考模型代码中的image2patches函数)"""
        # 使用einops重排张量
        x = x.squeeze(0)  # 移除batch维度
        x = torch.stack(x.split(x.shape[-2]//2, dim=-2), dim=0)  # 分割高度
        x = torch.stack(x.split(x.shape[-1]//2, dim=-1), dim=0)  # 分割宽度
        x = x.permute(1, 2, 0, 3, 4).contiguous()  # 重新排列
        x = x.view(4, x.shape[-3], x.shape[-2], x.shape[-1])  # 重塑为4个块
        return x.unsqueeze(1)  # 重新添加batch维度


# 初始化应用
app = FastAPI(
    title="MVANet Segmentation API",
    description="High-performance image segmentation API using MVANet",
    version="2.0.0"
)

# 初始化单例模型加载器和图像处理器
model_loader = ModelLoader()
processor = ImageProcessor()


@app.on_event("startup")
async def startup_event():
    """
    应用启动事件：预加载模型
    """
    logger.info("Starting up MVANet API service...")
    logger.info(f"Configuration: {Config.__dict__}")
    
    try:
        model_loader.load_model()
        logger.success("MVANet API service started successfully")
    except Exception as e:
        logger.error(f"Failed to start service: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """
    应用关闭事件：清理资源
    """
    logger.info("Shutting down MVANet API service...")
    model_loader.clear_cache()
    logger.success("MVANet API service shutdown completed")


@app.get("/")
async def root():
    """
    根路径健康检查
    """
    return {
        "message": "MVANet Segmentation API is running",
        "version": "2.0.0",
        "device": Config.DEVICE,
        "model_loaded": model_loader._model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    图像分割预测接口
    """
    start_time = time.time()
    request_id = f"{int(start_time * 1000000)}"
    
    logger.info(f"[{request_id}] Received prediction request for file: {file.filename}")
    
    try:
        # 验证文件类型
        content_type = file.content_type
        if not content_type or not content_type.startswith("image/"):
            logger.warning(f"[{request_id}] Invalid content type: {content_type}")
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        
        # 读取图像
        contents = await file.read()
        image_stream = io.BytesIO(contents)
        
        try:
            # 打开图像
            image = Image.open(image_stream)
            original_size = image.size
            logger.info(f"[{request_id}] Image loaded: {original_size}, mode: {image.mode}")
        except Exception as e:
            logger.error(f"[{request_id}] Failed to open image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # 预处理
        try:
            input_tensor = processor.preprocess(image)
            input_tensor = input_tensor.to(model_loader._device)
            processed_size = input_tensor.shape[-2:]
            logger.info(f"[{request_id}] Image preprocessed: {processed_size}")
        except ValueError as e:
            logger.error(f"[{request_id}] Image validation failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"[{request_id}] Preprocessing failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Preprocessing error")
        
        # 获取模型
        model = model_loader.get_model()
        
        # 推理
        with gpu_memory_cleanup():  # 确保显存清理
            with torch.no_grad():
                try:
                    # 根据模型代码，需要创建多视角输入
                    glb = F.interpolate(
                        input_tensor, 
                        scale_factor=0.5, 
                        mode='bilinear', 
                        align_corners=False
                    )
                    loc = processor.split_to_patches(input_tensor)
                    
                    # 组合输入
                    combined_input = torch.cat((loc, glb), dim=0)
                    
                    # 模型推理
                    output = model(combined_input)
                    
                    # 如果模型返回多个输出，取最后一个
                    if isinstance(output, (list, tuple)):
                        output = output[-1]
                    
                    logger.success(f"[{request_id}] Model inference completed")
                except Exception as e:
                    logger.error(f"[{request_id}] Model inference failed: {str(e)}")
                    raise HTTPException(status_code=500, detail="Model inference error")
        
        # 后处理
        try:
            mask_image = processor.postprocess(output, original_size)
            logger.info(f"[{request_id}] Post-processing completed")
        except Exception as e:
            logger.error(f"[{request_id}] Post-processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Post-processing error")
        
        # 转换为Base64
        try:
            buffer = io.BytesIO()
            mask_image.save(buffer, format="PNG")
            mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            logger.info(f"[{request_id}] Mask encoded to Base64")
        except Exception as e:
            logger.error(f"[{request_id}] Base64 encoding failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Encoding error")
        
        # 计算推理时间
        inference_time = time.time() - start_time
        
        logger.info(f"[{request_id}] Prediction completed successfully in {inference_time:.3f}s")
        
        return PredictionResponse(
            success=True,
            mask_base64=mask_base64,
            message="Segmentation successful",
            inference_time=inference_time,
            original_size=original_size,
            processed_size=processed_size
        )
    
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        # 记录错误
        error_msg = f"[{request_id}] Prediction error: {str(e)}"
        logger.error(error_msg)
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        return PredictionResponse(
            success=False,
            mask_base64=None,
            message=f"Error during prediction: {str(e)}",
            inference_time=total_time,
            original_size=None,
            processed_size=None
        )


# 添加健康检查端点
@app.get("/health")
async def health_check():
    """
    健康检查端点
    """
    health_info = {
        "status": "healthy",
        "model_loaded": model_loader._model is not None,
        "device": Config.DEVICE,
        "timestamp": time.time()
    }
    
    # 如果是GPU，添加显存信息
    if torch.cuda.is_available():
        health_info.update({
            "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**2,
            "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**2
        })
    
    logger.debug(f"Health check: {health_info}")
    return health_info


# 添加配置信息端点
@app.get("/config")
async def config_info():
    """
    配置信息端点
    """
    return {
        "model_weights_path": Config.MODEL_WEIGHTS_PATH,
        "model_input_size": Config.MODEL_INPUT_SIZE,
        "device": Config.DEVICE,
        "image_max_size": Config.IMAGE_MAX_SIZE,
        "threshold": Config.THRESHOLD
    }


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {Config.HOST}:{Config.PORT}")
    uvicorn.run(
        app, 
        host=Config.HOST, 
        port=Config.PORT,
        reload=True
    )