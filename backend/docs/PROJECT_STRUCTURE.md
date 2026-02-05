# MVANet 项目结构说明

## 项目概述
这是一个基于 MVANet 的图像分割 API 服务，从学术研究代码转换为生产级应用。

## 项目结构

```
MVANet/
├── model/                    # 模型定义文件
│   ├── MVANet.py            # 核心模型定义
│   └── SwinTransformer.py   # 骨干网络定义
├── utils/                    # 工具函数
│   ├── config.py            # 配置文件
│   ├── ...                  # 其他工具
├── weights/                  # 模型权重文件 (需要下载)
├── app_optimized.py         # 优化的FastAPI应用 (生产就绪)
├── app.py                   # 原始API应用
├── data_processor.py        # 数据处理脚本
├── model_validator.py       # 模型验证脚本
├── stress_test.py           # 压力测试脚本
├── interview_demo.py        # 面试演示脚本
├── requirements.txt         # 依赖包列表
├── .env.example            # 环境变量配置模板
├── .env                    # 环境变量配置文件
├── Dockerfile              # Docker配置文件
├── start_production.sh     # 生产环境启动脚本
├── README.md               # 项目说明文档
└── ...
```

## 核心功能模块

### 1. API服务 (app_optimized.py)
- 异步图像分割接口
- GPU显存管理
- 结构化日志记录
- 健康检查接口
- 配置外部化管理

### 2. 数据处理 (data_processor.py)
- 自定义数据集处理
- 图像预处理管道
- 数据格式验证

### 3. 模型验证 (model_validator.py)
- 模型加载验证
- 推理功能测试
- 结果可视化

### 4. 性能测试 (stress_test.py)
- 并发请求测试
- 性能指标统计
- QPS计算

### 5. 面试演示 (interview_demo.py)
- 完整能力验证
- 工程化最佳实践展示

## 面试重点

### 1. 环境配置能力
- 复杂深度学习环境搭建
- 依赖管理策略
- GPU环境配置

### 2. 数据处理能力
- 自定义数据集处理流程
- 数据预处理和验证
- 格式兼容性处理

### 3. 工程化能力
- 从研究代码到生产代码的转换
- 性能优化策略
- 错误处理和日志记录
- 容器化部署方案
- 监控和健康检查

## 使用说明

### 开发环境
```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件

# 启动服务
python app_optimized.py
```

### 生产环境
```bash
# 使用Docker
docker build -t mvanet-api .
docker run -p 8000:8000 mvanet-api

# 或使用启动脚本
bash start_production.sh
```

### 面试演示
```bash
python interview_demo.py
```

## 关键优化点

1. **显存管理**: GPU内存清理和监控
2. **异步处理**: 提高并发性能
3. **配置管理**: 环境变量外部化
4. **日志记录**: 结构化日志系统
5. **错误处理**: 全面的异常处理机制
6. **健康检查**: 符合K8s标准
7. **性能测试**: 压力测试和性能分析