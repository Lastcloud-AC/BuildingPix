"""
Phase 5 测试脚本：从裁剪结果中选取一张组件图，一次生成正视图 + 侧视图
===========================================================================
输入：phase4 裁剪后的单个组件图片
输出：2 张独立的正交视图（正视图 + 侧视图），纯白背景，无文字

使用方式：
  python test_phase5_views.py                    # 自动选取第一张可用图片
  python test_phase5_views.py path/to/image.png  # 指定图片路径

依赖：
  - config.py 中的 IMAGE_EDIT_API_URL, IMAGE_EDIT_API_KEY, IMAGE_EDIT_MODEL
  - PIL (Pillow)
"""

import os
import sys
import base64
import json
import random
import requests
import time
from pathlib import Path
from PIL import Image

# ─── 路径配置 ───
PHASE4_OUTPUT_DIR = Path(__file__).parent / "output" / "phase4_cropped"
TEST_OUTPUT_DIR = Path(__file__).parent / "output" / "phase5_test"

# ─── 输出尺寸 ───
OUTPUT_SIZE = "512*512"

# ─── 固定提示词模板 ───
# 要求一次生成两张独立图片：正视图和侧视图
PROMPT_FRONT = """参考这张建筑组件图片，生成该组件的正面正交视图。

【视角】
从正前方平视该组件，正交投影，无透视变形。

【画面要求】
- 纯白背景
- 无任何文字、标注、水印
- 组件居中，占画面80%以上
- 干净边缘，扁平着色
- 保持与原图一致的风格和细节"""

PROMPT_SIDE = """参考这张建筑组件图片，生成该组件的左侧正交视图。

【视角】
从该组件左侧90度方向平视，正交投影，无透视变形。看到组件的完整左侧面。

【画面要求】
- 纯白背景
- 无任何文字、标注、水印
- 组件居中，占画面80%以上
- 干净边缘，扁平着色
- 保持与原图一致的风格和细节"""


# ════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════

def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """根据扩展名返回 media type"""
    ext = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext, "image/png")


def download_image(url: str, output_path: str) -> bool:
    """下载图片"""
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return False


def find_first_available_image() -> Path:
    """从 phase4 输出中找到第一张可用的裁剪图片"""
    if not PHASE4_OUTPUT_DIR.exists():
        return None

    for module_dir in PHASE4_OUTPUT_DIR.iterdir():
        if not module_dir.is_dir():
            continue
        for zone_dir in module_dir.iterdir():
            if not zone_dir.is_dir():
                continue
            for img_file in sorted(zone_dir.glob("*.png")):
                return img_file

    return None


# ════════════════════════════════════════════
# 核心：单次 API 调用，生成指定视图
# ════════════════════════════════════════════

def generate_view(
    source_image_path: str,
    output_path: str,
    prompt: str,
    view_name: str,
    max_retries: int = 3,
) -> bool:
    """
    调用 DashScope Qwen 图像编辑 API 生成单个视图（带重试机制）

    Args:
        source_image_path: 源组件图片路径
        output_path: 输出图片路径
        prompt: 提示词
        view_name: 视图名称（用于日志）
        max_retries: 最大重试次数

    Returns:
        是否成功
    """
    from config import IMAGE_EDIT_API_URL, IMAGE_EDIT_API_KEY, IMAGE_EDIT_MODEL

    # 编码图片为 base64 data URI
    b64 = encode_image_to_base64(source_image_path)
    media_type = get_image_media_type(source_image_path)
    data_uri = f"data:{media_type};base64,{b64}"

    # 构建请求体
    payload = {
        "model": IMAGE_EDIT_MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": data_uri},
                        {"text": prompt}
                    ]
                }
            ]
        },
        "parameters": {
            "n": 1,
            "negative_prompt": "文字, 水印, 标注, 文字说明, 透视变形, 3D渲染, 写实风格, 真实照片, 模糊, 低质量, 杂乱背景",
            "prompt_extend": False,
            "watermark": False,
            "size": OUTPUT_SIZE,
            "seed": random.randint(1, 999999)
        }
    }

    headers = {
        "Authorization": f"Bearer {IMAGE_EDIT_API_KEY}",
        "Content-Type": "application/json"
    }

    for attempt in range(1, max_retries + 1):
        print(f"  🎨 生成{view_name}... (尝试 {attempt}/{max_retries})")
        try:
            resp = requests.post(
                IMAGE_EDIT_API_URL,
                json=payload,
                headers=headers,
                timeout=300,
            )

            if resp.status_code != 200:
                print(f"  ❌ 生成失败 (HTTP {resp.status_code})")
                print(f"  📄 错误: {resp.text[:500]}")
                if attempt < max_retries:
                    wait = 15 * attempt
                    print(f"     等待 {wait} 秒后重试...")
                    time.sleep(wait)
                    continue
                return False

            result = resp.json()

            # 保存响应用于调试
            response_path = Path(output_path).with_suffix(".json")
            with open(response_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            # 解析响应
            output = result.get("output", {})
            choices = output.get("choices", [])

            if not choices:
                print(f"  ❌ 响应中无 choices")
                if attempt < max_retries:
                    wait = 15 * attempt
                    print(f"     等待 {wait} 秒后重试...")
                    time.sleep(wait)
                    continue
                return False

            message = choices[0].get("message", {})
            content_list = message.get("content", [])

            for item in content_list:
                # 图片 URL
                img_url = item.get("image")
                if img_url:
                    return download_image(img_url, output_path)

                # base64 图片
                b64_img = item.get("image_url", {}).get("url", "")
                if b64_img and b64_img.startswith("data:"):
                    _, b64_data = b64_img.split(",", 1)
                    with open(output_path, "wb") as f:
                        f.write(base64.b64decode(b64_data))
                    return True

            print(f"  ❌ 响应中无图片数据")
            if attempt < max_retries:
                wait = 15 * attempt
                print(f"     等待 {wait} 秒后重试...")
                time.sleep(wait)
                continue
            return False

        except requests.exceptions.ReadTimeout:
            print(f"  ⚠️  请求超时 (第 {attempt} 次)")
            if attempt < max_retries:
                wait = 15 * attempt
                print(f"     等待 {wait} 秒后重试...")
                time.sleep(wait)
            else:
                print(f"  ❌ 已达最大重试次数")
                return False
        except requests.exceptions.ConnectionError as e:
            print(f"  ⚠️  连接错误 (第 {attempt} 次): {e}")
            if attempt < max_retries:
                wait = 15 * attempt
                print(f"     等待 {wait} 秒后重试...")
                time.sleep(wait)
            else:
                print(f"  ❌ 已达最大重试次数")
                return False

    return False


# ════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════

def main():
    """主入口"""
    # 确保输出目录存在
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 确定输入图片
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
        if not input_path.exists():
            print(f"❌ 指定的图片不存在: {input_path}")
            sys.exit(1)
    else:
        input_path = find_first_available_image()
        if not input_path:
            print(f"❌ Phase 4 输出目录中没有找到裁剪图片: {PHASE4_OUTPUT_DIR}")
            print(f"   请先运行 phase4_crop_quadrant.py，或手动指定图片路径")
            sys.exit(1)

    stem = input_path.stem  # 如 lower_下层区域_base_stone_lower
    front_output = TEST_OUTPUT_DIR / f"{stem}_front.png"
    side_output = TEST_OUTPUT_DIR / f"{stem}_side.png"

    print(f"{'='*60}")
    print(f"🖼️  Phase 5 测试：生成正视图 + 侧视图")
    print(f"{'='*60}")
    print(f"  输入图片: {input_path}")
    print(f"  输出尺寸: {OUTPUT_SIZE}")
    print(f"  正视图输出: {front_output.name}")
    print(f"  侧视图输出: {side_output.name}")
    print()

    # 显示源图片信息
    try:
        img = Image.open(str(input_path))
        print(f"  📐 源图片尺寸: {img.size[0]}x{img.size[1]}")
    except Exception:
        pass

    results = {}

    # ── 生成正视图 ──
    print(f"\n{'─'*40}")
    print(f"📐 生成正视图 (Front View)")
    print(f"{'─'*40}")
    front_ok = generate_view(
        source_image_path=str(input_path),
        output_path=str(front_output),
        prompt=PROMPT_FRONT,
        view_name="正视图",
    )
    results["正视图"] = front_ok

    if front_ok:
        print(f"  ✅ 正视图已保存: {front_output.name}")
    else:
        print(f"  ❌ 正视图生成失败")

    # 间隔等待，避免限流
    print(f"\n  ⏳ 等待 5 秒...")
    time.sleep(5)

    # ── 生成侧视图 ──
    print(f"\n{'─'*40}")
    print(f"📐 生成侧视图 (Side View)")
    print(f"{'─'*40}")
    side_ok = generate_view(
        source_image_path=str(input_path),
        output_path=str(side_output),
        prompt=PROMPT_SIDE,
        view_name="侧视图",
    )
    results["侧视图"] = side_ok

    if side_ok:
        print(f"  ✅ 侧视图已保存: {side_output.name}")
    else:
        print(f"  ❌ 侧视图生成失败")

    # ── 结果摘要 ──
    print(f"\n{'='*60}")
    print(f"📊 测试结果摘要")
    print(f"{'='*60}")
    print(f"  源组件: {input_path.name}")

    for view_name, ok in results.items():
        status = "✅ 成功" if ok else "❌ 失败"
        print(f"  {view_name}: {status}")

    # 显示生成图片信息
    for path in [front_output, side_output]:
        if path.exists():
            try:
                gen_img = Image.open(str(path))
                print(f"  📐 {path.name}: {gen_img.size[0]}x{gen_img.size[1]}")
            except Exception:
                pass


if __name__ == "__main__":
    main()