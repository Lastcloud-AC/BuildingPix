"""
测试图像编辑 API (Qwen qwen-image-2.0-pro)
使用 DashScope 的多模态生成接口
参考图通过 URL 传递
"""

import requests
import base64
import json
import os
import time
from pathlib import Path
from config import IMAGE_EDIT_API_URL, IMAGE_EDIT_API_KEY, IMAGE_EDIT_MODEL

# DashScope Qwen 图像编辑 API 配置（从 config.py 导入）
API_URL = IMAGE_EDIT_API_URL
API_KEY = IMAGE_EDIT_API_KEY
MODEL = IMAGE_EDIT_MODEL


def encode_local_image_to_data_uri(image_path: str) -> str:
    """将本地图片编码为 base64 data URI"""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(image_path).suffix.lower()
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp"
    }.get(ext, "image/png")
    return f"data:{media_type};base64,{b64}"


def get_image_as_content(image_path: str) -> dict:
    """
    将本地图片转为 API 所需的 content 格式。
    如果是 URL 则直接返回 dict，否则编码为 base64 data URI。
    """
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return {"image": image_path}
    else:
        data_uri = encode_local_image_to_data_uri(image_path)
        print(f"  已编码为 data URI ({len(data_uri)} 字符)")
        return {"image": data_uri}


def test_image_edit():
    # 测试图片路径（可改为本地路径或 URL）
    image_path = "output/phase2/SurfCG/zone_02_upper.png"
    output_path = "output/phase3_test/test_api_result.png"
    response_path = "output/phase3_test/test_api_response.json"

    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    prompt = """参考建筑立面图，生成2x2四宫格模块拆分正交视图。

【视角要求】
统一前左45度角：同时展示组件的正面和左侧面，无透视变形。

【构图要求】
- 等轴测正交投影
- 纯白背景
- 四个格子均匀分布，格子间有清晰分割线
- 组件居中展示，占格子85%以上空间

【四个组件】
1. 左上：玫瑰窗 - 哥特风格圆形花窗，展示正面圆形窗框与花瓣图案
2. 右上：老虎窗 - 带歇山屋顶的小窗，展示正面窗体与左侧屋顶结构
3. 左下：阳台 - 悬挑石栏阳台，展示正面栏板与左侧支撑结构
4. 右下：屋顶栏杆 - 镂空石栏杆段落，展示正面与左侧镂空结构

【风格要求】
- 游戏资产风格：干净边缘，扁平着色
- 组件结构完整，可独立作为模块使用
- 同一建筑风格，与参考图一致"""

    print(f"测试图像编辑 API (Qwen {MODEL})")
    print(f"  URL: {API_URL}")
    print(f"  测试图片: {image_path}")
    print()

    # 构建图片 content
    image_content = get_image_as_content(image_path)

    # 构建请求体 - DashScope 多模态生成格式
    payload = {
        "model": MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        image_content,
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        },
        "parameters": {
            "n": 1,
            "negative_prompt": " ",
            "prompt_extend": True,
            "watermark": False,
            "size": "1024*1024"
        }
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    print(f"\n发送请求...")
    try:
        resp = requests.post(
            API_URL,
            json=payload,
            headers=headers,
            timeout=300,
        )

        print(f"状态码: {resp.status_code}")
        print(f"响应内容: {resp.text[:3000]}")

        if resp.status_code == 200:
            result = resp.json()

            # 保存完整响应
            with open(response_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n完整响应已保存: {response_path}")

            # 解析 DashScope 多模态生成响应格式
            if isinstance(result, dict):
                output = result.get("output", {})
                choices = output.get("choices", [])

                if choices:
                    message = choices[0].get("message", {})
                    content_list = message.get("content", [])

                    for item in content_list:
                        # 图片 URL
                        img_url = item.get("image")
                        if img_url:
                            print(f"\n  图片URL: {img_url}")
                            img_resp = requests.get(img_url, timeout=60)
                            if img_resp.status_code == 200:
                                with open(output_path, "wb") as f:
                                    f.write(img_resp.content)
                                print(f"\n✅ 图片已保存: {output_path}")
                            else:
                                print(f"\n❌ 下载图片失败: {img_resp.status_code}")
                            return

                        # base64 图片
                        b64_img = item.get("image_url", {}).get("url", "")
                        if b64_img and b64_img.startswith("data:"):
                            _, b64_data = b64_img.split(",", 1)
                            with open(output_path, "wb") as f:
                                f.write(base64.b64decode(b64_data))
                            print(f"\n✅ 图片已保存 (base64): {output_path}")
                            return

                # 备用解析：直接检查 output 中的图片
                results = output.get("results", [])
                if results:
                    for item in results:
                        img_url = item.get("url") or item.get("image")
                        if img_url:
                            print(f"\n  图片URL: {img_url}")
                            img_resp = requests.get(img_url, timeout=60)
                            if img_resp.status_code == 200:
                                with open(output_path, "wb") as f:
                                    f.write(img_resp.content)
                                print(f"\n✅ 图片已保存: {output_path}")
                                return

                print(f"\n⚠️ 未能从响应中解析图片，请检查响应文件: {response_path}")
            elif isinstance(result, list):
                with open(response_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"\n⚠️ 响应为列表格式，请检查: {response_path}")
        else:
            print(f"\n❌ 请求失败，状态码: {resp.status_code}")

    except Exception as e:
        print(f"请求异常: {e}")


if __name__ == "__main__":
    test_image_edit()