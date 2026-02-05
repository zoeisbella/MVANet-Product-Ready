import asyncio
import time
import random
from typing import List, Dict
import httpx
import io
from PIL import Image
import numpy as np


async def generate_test_image(size=(256, 256)) -> bytes:
    """
    生成测试图像
    """
    # 创建随机图像用于测试
    image_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    # 将图像保存到字节流
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()


async def send_single_request(client: httpx.AsyncClient, image_data: bytes, url: str) -> Dict:
    """
    发送单个请求并返回结果
    """
    start_time = time.time()
    
    try:
        response = await client.post(
            url,
            files={'file': ('test_image.jpg', image_data, 'image/jpeg')},
            timeout=30.0  # 30秒超时
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            'success': response.status_code == 200,
            'status_code': response.status_code,
            'response_time': response_time,
            'error': None if response.status_code == 200 else response.text
        }
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        return {
            'success': False,
            'status_code': None,
            'response_time': response_time,
            'error': str(e)
        }


async def run_stress_test(
    url: str = "http://localhost:8000/predict",
    num_requests: int = 50,
    concurrency: int = 10
) -> Dict:
    """
    运行压力测试
    """
    print(f"开始压力测试...")
    print(f"目标URL: {url}")
    print(f"请求数量: {num_requests}")
    print(f"并发数: {concurrency}")
    print("-" * 50)
    
    # 生成测试图像
    print("生成测试图像...")
    test_images = []
    for _ in range(num_requests):
        image_data = await generate_test_image()
        test_images.append(image_data)
    
    # 创建HTTP客户端
    async with httpx.AsyncClient(timeout=30.0) as client:
        semaphore = asyncio.Semaphore(concurrency)  # 控制并发数
        
        async def limited_request(image_data):
            async with semaphore:
                return await send_single_request(client, image_data, url)
        
        # 发送并发请求
        print(f"发送 {num_requests} 个并发请求...")
        start_time = time.time()
        
        tasks = [
            asyncio.create_task(limited_request(image_data))
            for image_data in test_images
        ]
        
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
    
    # 统计结果
    successful_requests = [r for r in results if r['success']]
    failed_requests = [r for r in results if not r['success']]
    
    response_times = [r['response_time'] for r in results]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    max_response_time = max(response_times) if response_times else 0
    min_response_time = min(response_times) if response_times else 0
    
    success_rate = len(successful_requests) / num_requests * 100 if num_requests > 0 else 0
    
    # 打印结果
    print("-" * 50)
    print("压力测试结果:")
    print(f"总请求数: {num_requests}")
    print(f"成功请求数: {len(successful_requests)}")
    print(f"失败请求数: {len(failed_requests)}")
    print(f"成功率: {success_rate:.2f}%")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均响应时间: {avg_response_time:.3f}秒")
    print(f"最短响应时间: {min_response_time:.3f}秒")
    print(f"最长响应时间: {max_response_time:.3f}秒")
    print(f"QPS (每秒查询率): {num_requests / total_time:.2f}")
    
    if failed_requests:
        print(f"\n失败请求详情 (显示前5个):")
        for i, req in enumerate(failed_requests[:5]):
            print(f"  请求{i+1}: 状态码={req['status_code']}, 错误={req['error'][:100]}...")
    
    # 性能评估
    print("\n性能评估:")
    if success_rate >= 95:
        print("✅ 优秀: 成功率超过95%")
    elif success_rate >= 80:
        print("⚠️  良好: 成功率在80%-95%之间")
    else:
        print("❌ 需要改进: 成功率低于80%")
    
    if avg_response_time < 1.0:
        print("✅ 快速响应: 平均响应时间小于1秒")
    elif avg_response_time < 3.0:
        print("⚠️ 一般响应: 平均响应时间在1-3秒之间")
    else:
        print("❌ 需要优化: 平均响应时间超过3秒")
    
    return {
        'total_requests': num_requests,
        'successful_requests': len(successful_requests),
        'failed_requests': len(failed_requests),
        'success_rate': success_rate,
        'total_time': total_time,
        'avg_response_time': avg_response_time,
        'max_response_time': max_response_time,
        'min_response_time': min_response_time,
        'qps': num_requests / total_time,
        'results': results
    }


async def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='MVANet API 压力测试')
    parser.add_argument('--url', type=str, default='http://localhost:8000/predict',
                       help='API端点URL (默认: http://localhost:8000/predict)')
    parser.add_argument('--requests', type=int, default=50,
                       help='请求数量 (默认: 50)')
    parser.add_argument('--concurrency', type=int, default=10,
                       help='并发数 (默认: 10)')
    
    args = parser.parse_args()
    
    # 运行健康检查
    print("首先检查服务健康状态...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            health_resp = await client.get(f"{args.url.rsplit('/', 1)[0]}/health")
            if health_resp.status_code == 200:
                print("✅ 服务健康检查通过")
                health_data = health_resp.json()
                print(f"   服务状态: {health_data.get('status', 'unknown')}")
                print(f"   模型加载: {health_data.get('model_loaded', 'unknown')}")
                print(f"   设备: {health_data.get('device', 'unknown')}")
            else:
                print(f"⚠️  服务健康检查失败，状态码: {health_resp.status_code}")
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        return
    
    print()
    
    # 运行压力测试
    results = await run_stress_test(
        url=args.url,
        num_requests=args.requests,
        concurrency=args.concurrency
    )
    
    return results


if __name__ == "__main__":
    asyncio.run(main())