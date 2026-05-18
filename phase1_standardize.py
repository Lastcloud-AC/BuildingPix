"""
Phase 1: 立面标准化
===========================
输入：一张建筑截图（任意透视角度）
输出：纯白背景、无阴影的正面正交图

流程：
  1. 读取 input/ 中的图片
  2. 调用 VLM 分析图片：判断透视类型、提取建筑描述
  3. 调用图像生成模型：基于描述生成正交正视图
  4. 输出到 output/phase1/

使用方式：
  将建筑截图放入 input/ 文件夹，运行本脚本即可。
"""

import os
import sys
import base64
import json
import time
import requests
from pathlib import Path

# ─── 路径配置 ───
INPUT_DIR = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output" / "phase1"


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
    }.get(ext, "image/jpeg")


# ════════════════════════════════════════════
# Step 1: VLM 分析 - 判断透视类型 + 提取建筑描述
# ════════════════════════════════════════════

ANALYZE_PROMPT = """You are an expert architectural draftsman. Analyze this building screenshot and return a PRECISE geometric description in JSON format.

Your goal: describe the building so precisely that someone can recreate the EXACT same building from your text alone. Count everything. Measure proportions. No vague artistic language.

{
  "perspective_type": "one-point / two-point / three-point / front-view / bird-eye / unknown",
  "building_style": "architectural style in 2-3 words",
  "total_floors": "exact number of floors",
  "width_height_ratio": "approximate width:height ratio of the facade, e.g. 3:2",
  "roof": {
    "type": "flat / gable / hip / mansard / dome / pagoda / other",
    "color": "roof color",
    "details": "e.g. triangular gable with decorative cornice"
  },
  "facade": {
    "material": "brick / stone / concrete / wood / stucco / mixed",
    "main_color": "dominant facade color",
    "accent_color": "secondary color if any"
  },
  "windows": {
    "count_per_floor": "number of windows per floor",
    "arrangement": "symmetrical / asymmetrical / irregular",
    "style": "arched / rectangular / round / bay / other",
    "frame_color": "window frame color",
    "has_shutters": true,
    "shutter_color": "color if has shutters"
  },
  "door": {
    "position": "center / left / right",
    "style": "single / double / arched / glass / other",
    "color": "door color",
    "has_canopy": false,
    "canopy_description": ""
  },
  "decorative_elements": [
    "list every visible decorative element: cornices, columns, pilasters, balconies, railings, moldings, keystones, etc."
  ],
  "ground_floor": "description of ground floor features (shop fronts, arcade, base, steps)",
  "symmetry": "symmetrical / mostly symmetrical / asymmetrical",
  "unique_features": "any distinctive features that make this building recognizable"
}

CRITICAL RULES:
- Count exact numbers (windows, floors, columns, etc.)
- Use proportions (e.g. "windows are 1/3 the height of each floor")
- Describe LEFT-to-RIGHT and TOP-to-BOTTOM layout
- Ignore background/scenery, ONLY describe the building itself
- If uncertain about a detail, say "uncertain" rather than guessing
- Return ONLY the JSON, no extra text"""


def analyze_image(image_path: str) -> dict:
    """调用 VLM 分析建筑图片"""
    from config import VLM_API_URL, VLM_API_KEY, VLM_MODEL

    if not all([VLM_API_URL, VLM_API_KEY, VLM_MODEL]):
        print("❌ 错误：请先在 config.py 中填写 VLM 相关配置")
        sys.exit(1)

    b64 = encode_image_to_base64(image_path)
    media_type = get_image_media_type(image_path)

    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ANALYZE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0.3,
        "max_tokens": 1000,
    }

    headers = {
        "Authorization": f"Bearer {VLM_API_KEY}",
        "Content-Type": "application/json",
    }

    print("  🔍 调用 VLM 分析图片...")
    resp = requests.post(
        VLM_API_URL,
        json=payload,
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    # 尝试提取 JSON
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return json.loads(content)


# ════════════════════════════════════════════
# Step 2: 图像生成 - 生成正交正视图
# ════════════════════════════════════════════

def build_generate_prompt(analysis: dict) -> str:
    """基于 VLM 分析结果，构建精确的图像生成 prompt"""
    # 安全取值
    facade = analysis.get("facade", {})
    windows = analysis.get("windows", {})
    door = analysis.get("door", {})
    roof = analysis.get("roof", {})
    deco = analysis.get("decorative_elements", [])

    prompt = f"""Architectural front elevation drawing, orthographic projection, game asset reference sheet.
建筑正立面图，正交投影，游戏素材参考图。

=== BUILDING SPEC / 建筑规格 ===
- Style (建筑风格): {analysis.get('building_style', 'unknown')}
- Total floors (总楼层数): {analysis.get('total_floors', 'unknown')}
- Proportion (宽高比例): {analysis.get('width_height_ratio', 'unknown')}
- Symmetry (对称性): {analysis.get('symmetry', 'unknown')}
- Facade (外墙): {facade.get('material', 'stone')} walls, {facade.get('main_color', 'gray')}, accent {facade.get('accent_color', 'none')}
- Roof (屋顶): {roof.get('type', 'flat')}, {roof.get('color', 'gray')}, {roof.get('details', 'simple')}
- Windows (窗户): {windows.get('count_per_floor', '?')} per floor, {windows.get('style', 'rectangular')}, {windows.get('arrangement', 'symmetrical')}, frame {windows.get('frame_color', 'white')}, shutters={windows.get('has_shutters', False)}
- Door (门): {door.get('position', 'center')} position, {door.get('style', 'single')}, {door.get('color', 'brown')}
- Ground floor (底层): {analysis.get('ground_floor', 'standard')}
- Decorative (装饰元素): {', '.join(deco) if deco else 'minimal'}
- Unique (独特特征): {analysis.get('unique_features', 'none')}

=== VISUAL RULES / 视觉规则 ===
- Strict FRONT orthographic view, zero perspective distortion (严格正面正交视图，无透视变形)
- Pure WHITE background (#FFFFFF), no environment, no sky, no ground shadow (纯白色背景，无环境、天空、阴影)
- Technical illustration style, clean lines, clear details (技术插画风格，线条清晰)
- All windows/doors/elements must match the exact count specified above (所有元素数量必须匹配)
- Show every decorative element described (展示所有装饰元素)
- 1024x1024 landscape (1024x1024 横版)"""

    return prompt


def generate_ortho_image(prompt: str, output_path: str) -> str:
    """调用图像生成模型，生成正交图，支持超时重试"""
    from config import IMAGE_GEN_API_URL, IMAGE_GEN_API_KEY, IMAGE_GEN_MODEL

    if not all([IMAGE_GEN_API_URL, IMAGE_GEN_API_KEY, IMAGE_GEN_MODEL]):
        print("❌ 错误：请先在 config.py 中填写图像生成模型相关配置")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {IMAGE_GEN_API_KEY}",
        "Content-Type": "application/json",
    }

    # DALL-E 3 仅支持 256x256 / 512x512 / 1024x1024
    payload = {
        "model": IMAGE_GEN_MODEL,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
    }

    max_retries = 3
    timeout = 300  # 5分钟超时

    for attempt in range(max_retries):
        try:
            print(f"  🎨 调用图像生成模型... (尝试 {attempt + 1}/{max_retries})")
            resp = requests.post(
                IMAGE_GEN_API_URL,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code != 200:
                print(f"  ❌ 图像生成失败 (HTTP {resp.status_code})")
                print(f"  📄 错误详情: {resp.text}")
                if attempt < max_retries - 1:
                    print(f"  ⏳ 等待 {5 * (attempt + 1)} 秒后重试...")
                    time.sleep(5 * (attempt + 1))
                    continue
                sys.exit(1)
            result = resp.json()
            break
        except requests.exceptions.Timeout:
            print(f"  ⏰ 请求超时 (timeout={timeout}s)")
            if attempt < max_retries - 1:
                print(f"  ⏳ 等待 {5 * (attempt + 1)} 秒后重试...")
                time.sleep(5 * (attempt + 1))
            else:
                print("  ❌ 多次重试后仍超时，请检查网络或API服务状态")
                sys.exit(1)
        except requests.exceptions.RequestException as e:
            print(f"  ❌ 请求失败: {e}")
            if attempt < max_retries - 1:
                print(f"  ⏳ 等待 {5 * (attempt + 1)} 秒后重试...")
                time.sleep(5 * (attempt + 1))
            else:
                print("  ❌ 多次重试后仍失败")
                sys.exit(1)

    image_url = result["data"][0]["url"]

    # 下载图片保存到本地
    print("  💾 下载生成的图片...")
    img_resp = requests.get(image_url, timeout=60)
    img_resp.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(img_resp.content)

    return output_path


# ════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════

def process_single_image(image_path: str):
    """处理单张图片：分析 → 生成正交图"""
    filename = Path(image_path).stem
    analysis_path = OUTPUT_DIR / f"{filename}_analysis.json"

    print(f"\n{'='*50}")
    print(f"📐 Phase 1: 立面标准化 - {Path(image_path).name}")
    print(f"{'='*50}")

    # ── Step 1: VLM 分析（检查是否已有结果）──
    analysis = None
    if analysis_path.exists():
        print(f"\n  ⚠️  已存在分析结果: {analysis_path}")
        with open(analysis_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"  📋 现有分析摘要:")
        print(f"     透视类型: {existing.get('perspective_type', '未知')}")
        print(f"     建筑风格: {existing.get('building_style', '未知')}")
        print(f"     楼层数:   {existing.get('total_floors', '未知')}")
        choice = input("  ❓ 是否替换？(y=重新分析 / n=直接使用现有结果): ").strip().lower()
        if choice == "y":
            print("  🔄 重新分析...")
            analysis = analyze_image(image_path)
        else:
            print("  ✅ 使用现有分析结果")
            analysis = existing
    else:
        print("\n[Step 1/2] 分析建筑图片...")
        analysis = analyze_image(image_path)

    # 保存分析结果
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"  📄 分析结果: {analysis_path}")

    # 打印关键信息
    print(f"  ✅ 透视类型: {analysis.get('perspective_type', '未知')}")
    print(f"  ✅ 建筑风格: {analysis.get('building_style', '未知')}")
    print(f"  ✅ 楼层数:   {analysis.get('total_floors', '未知')}")
    print(f"  ✅ 对称性:   {analysis.get('symmetry', '未知')}")

    # ── Step 2: 生成正交图 ──
    print("\n[Step 2/2] 生成正交正视图...")
    prompt = build_generate_prompt(analysis)
    print(f"  📝 生成 Prompt:\n{prompt}")

    output_path = str(OUTPUT_DIR / f"{filename}_ortho.png")
    generate_ortho_image(prompt, output_path)

    print(f"\n  ✅ 正交图已保存: {output_path}")
    return output_path


def main():
    # 确保目录存在
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 扫描 input 目录中的图片
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = [
        f for f in INPUT_DIR.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not images:
        print(f"⚠️  input/ 目录中没有找到图片文件")
        print(f"   请将建筑截图放入: {INPUT_DIR}")
        print(f"   支持格式: {', '.join(image_extensions)}")
        sys.exit(0)

    print(f"📁 找到 {len(images)} 张图片，开始处理...")

    results = []
    for img_path in images:
        output = process_single_image(str(img_path))
        results.append({
            "input": str(img_path),
            "output": output,
        })

    # 汇总
    print(f"\n{'='*50}")
    print(f"📊 Phase 1 处理完成")
    print(f"{'='*50}")
    print(f"  输入目录: {INPUT_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  处理数量: {len(results)}")
    for r in results:
        print(f"    {Path(r['input']).name} → {Path(r['output']).name}")


if __name__ == "__main__":
    main()
