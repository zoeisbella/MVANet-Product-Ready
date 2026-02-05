import requests
import base64
from PIL import Image
import io

def test_api(image_path):
    """
    测试API的简单脚本
    """
    url = "http://localhost:8000/predict"
    
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    files = {'file': ('test_image.jpg', image_bytes, 'image/jpeg')}
    
    print("Sending request to API...")
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            print("Segmentation successful!")
            print(f"Inference time: {result['inference_time']:.3f}s")
            
            # 解码并保存结果图像
            mask_data = base64.b64decode(result['mask_base64'])
            mask_image = Image.open(io.BytesIO(mask_data))
            mask_image.save('segmentation_result.png')
            print("Segmentation result saved as 'segmentation_result.png'")
        else:
            print(f"Error: {result['message']}")
    else:
        print(f"Request failed with status {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python test_api.py <image_path>")
        sys.exit(1)
    
    test_api(sys.argv[1])