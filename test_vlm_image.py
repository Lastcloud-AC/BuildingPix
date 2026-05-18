"""测试 VLM 是否支持图片输入"""
import requests
import base64
import json
from config import VLM_API_URL, VLM_API_KEY, VLM_MODEL


def test_text_only():
    """测试纯文本请求"""
    payload = {
        "model": VLM_MODEL,
        "messages": [{"role": "user", "content": "回复OK"}],
        "max_tokens": 10,
    }
    headers = {
        "Authorization": f"Bearer {VLM_API_KEY}",
        "Content-Type": "application/json",
    }
    print("[1] 测试纯文本请求...")
    resp = requests.post(VLM_API_URL, json=payload, headers=headers, timeout=30)
    print(f"  状态: {resp.status_code}")
    if resp.status_code == 200:
        print(f"  回复: {resp.json()['choices'][0]['message']['content']}")
        return True
    else:
        print(f"  错误: {resp.text[:300]}")
        return False


def test_with_image():
    """测试带图片的请求"""
    with open("input/SurfCG.jpg", "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "描述这张图片，用一句话"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 50,
    }
    headers = {
        "Authorization": f"Bearer {VLM_API_KEY}",
        "Content-Type": "application/json",
    }
    print("[2] 测试带图片请求...")
    try:
        resp = requests.post(VLM_API_URL, json=payload, headers=headers, timeout=60)
        print(f"  状态: {resp.status_code}")
        if resp.status_code == 200:
            print(f"  回复: {resp.json()['choices'][0]['message']['content']}")
            return True
        else:
            print(f"  错误: {resp.text[:500]}")
            return False
    except requests.exceptions.ReadTimeout:
        print("  ❌ 超时！该模型可能不支持图片输入")
        return False
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print(f"  模型: {VLM_MODEL}")
    print(f"  URL:  {VLM_API_URL}")
    print("=" * 50)

    ok1 = test_text_only()
    print()
    ok2 = test_with_image()

    print("\n" + "=" * 50)
    if ok1 and ok2:
        print("  ✅ 该模型支持图片输入，可以用于 Phase 2")
    elif ok1 and not ok2:
        print("  ❌ 该模型是纯文本模型，不支持图片输入！")
        print("  💡 请换用支持图片的模型，如: qwen-vl-max, qwen2.5-vl-72b-instruct")
    else:
        print("  ❌ 连接失败，请检查 API 配置")