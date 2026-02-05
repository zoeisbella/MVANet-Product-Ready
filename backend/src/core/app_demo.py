"""
简化的API服务，用于演示目的
这个版本绕过了模型加载问题，专注于展示API功能
"""

import asyncio
import base64
import io
import time
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from torchvision import transforms
from loguru import logger
import gc


# 配置类
class Config:
    # 模型配置 - 使用测试权重路径
    MODEL_WEIGHTS_PATH = "./weights/test_weights.pth"
    MODEL_INPUT_SIZE = (256, 256)
    DEVICE = "cpu"  # 使用CPU以避免GPU问题
    
    # 服务器配置
    HOST = "0.0.0.0"
    PORT = 8000
    
    # 图像处理配置
    IMAGE_MAX_SIZE = 4096  # 最大图像尺寸
    THRESHOLD = 0.5  # 分割阈值


# Pydantic 模型定义
class PredictionResponse(BaseModel):
    success: bool
    mask_base64: Optional[str] = None
    message: str
    inference_time: float
    original_size: Optional[tuple] = None
    processed_size: Optional[tuple] = None


# 模拟模型加载器（演示用途）
class ModelLoader:
    _instance = None
    _model = None
    _device = torch.device("cpu")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def load_model(self):
        if self._model is None:
            try:
                logger.info(f"Loading model from {Config.MODEL_WEIGHTS_PATH}")
                
                # 模拟模型加载（演示用途）
                # 实际部署时需要真实的预训练权重
                self._model = "dummy_model"
                logger.success(f"Dummy model loaded successfully on {self._device}")
                    
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
        # 为演示创建模拟输出
        # 在实际应用中，这是模型的输出结果
        mask = np.random.rand(original_size[1], original_size[0])  # 模拟分割结果
        
        # 应用阈值
        mask = (mask > Config.THRESHOLD).astype(np.uint8) * 255
        
        # 转换为PIL图像
        mask_image = Image.fromarray(mask.astype(np.uint8), mode='L')
        
        return mask_image


# 初始化应用
app = FastAPI(
    title="MVANet Segmentation API (Demo)",
    description="High-performance image segmentation API using MVANet (Demo Version)",
    version="2.0.0"
)

# 初始化模拟模型加载器和图像处理器
model_loader = ModelLoader()
processor = ImageProcessor()


@app.on_event("startup")
async def startup_event():
    """
    应用启动事件：预加载模型
    """
    logger.info("Starting up MVANet API service...")
    logger.info(f"Configuration: Device={Config.DEVICE}, Host={Config.HOST}, Port={Config.PORT}")
    
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
        "message": "MVANet Segmentation API (Demo) is running",
        "version": "2.0.0",
        "device": Config.DEVICE,
        "model_loaded": model_loader._model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    图像分割预测接口（演示版）
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
            processed_size = input_tensor.shape[-2:]
            logger.info(f"[{request_id}] Image preprocessed: {processed_size}")
        except ValueError as e:
            logger.error(f"[{request_id}] Image validation failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"[{request_id}] Preprocessing failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Preprocessing error")
        
        # 模拟模型推理（演示版）
        try:
            # 模拟推理时间
            await asyncio.sleep(0.1)  # 模拟处理时间
            
            # 模拟模型输出
            output = input_tensor  # 实际应用中这里应该是模型的输出
            logger.success(f"[{request_id}] Model inference completed (simulated)")
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
            message="Segmentation successful (demo)",
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