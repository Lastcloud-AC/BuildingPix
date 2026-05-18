"""
Phase 3 单楼层测试脚本
======================
直接使用 Phase2 预生成的提示词，调用 DashScope 图生图API生成四宫格。

使用方式：
  python test_phase3_single_zone.py                    # 默认测试 zone_02_upper
  python test_phase3_single_zone.py zone_01_roof       # 指定区域
  python test_phase3_single_zone.py all                # 测试所有区域
"""

import sys
import base64
import json
import requests
from pathlib import Path
from PIL import Image

# ─── 路径配置 ───
PHASE2_DIR = Path(__file__).parent / "output" / "phase2" / "SurfCG"
OUTPUT_DIR = Path(__file__).parent / "output" / "phase3_test"


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def download_image(url: str, output_path: str) -> bool:
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"    ❌ 下载失败: {e}")
        return False


def generate_quadrant_image(source_image_path: str, prompt: str, output_path: str) -> bool:
    """调用 DashScope Qwen 图像编辑 API 生成四宫格"""
    from config import IMAGE_EDIT_API_URL, IMAGE_EDIT_API_KEY, IMAGE_EDIT_MODEL

    with open(source_image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(source_image_path).suffix.lower()
    media_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/png")
    data_uri = f"data:{media_type};base64,{b64}"

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
            "negative_prompt": "右视图, 右侧视角, 镜像, 翻转, 右朝向, 透视变形, 3D渲染, 写实风格, 真实照片",
            "prompt_extend": False,
            "watermark": False,
            "size": "1024*1024",
            "seed": 3238445,
        }
    }

    headers = {
        "Authorization": f"Bearer {IMAGE_EDIT_API_KEY}",
        "Content-Type": "application/json"
    }

    print(f"  🎨 调用 DashScope 图像编辑...")
    resp = requests.post(IMAGE_EDIT_API_URL, json=payload, headers=headers, timeout=300)

    if resp.status_code != 200:
        print(f"  ❌ 生成失败 (HTTP {resp.status_code}): {resp.text[:300]}")
        return False

    result = resp.json()
    output = result.get("output", {})
    choices = output.get("choices", [])

    if not choices:
        print(f"  ❌ 响应中无 choices")
        return False

    message = choices[0].get("message", {})
    content_list = message.get("content", [])

    for item in content_list:
        img_url = item.get("image")
        if img_url:
            return download_image(img_url, output_path)
        b64_img = item.get("image_url", {}).get("url", "")
        if b64_img and b64_img.startswith("data:"):
            _, b64_data = b64_img.split(",", 1)
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(b64_data))
            return True

    print(f"  ❌ 响应中无图片数据")
    return False


def test_zone(zone_name: str, manifest: dict, floor_zones: list) -> dict:
    """测试单个zone，按可复用类型分批生成四宫格"""
    zone_info = None
    for zone in floor_zones:
        cropped = zone.get("cropped_file", "")
        if cropped.replace(".png", "") == zone_name:
            zone_info = zone
            break

    if not zone_info:
        print(f"❌ 未找到区域: {zone_name}")
        return None

    zone_image = PHASE2_DIR / f"{zone_name}.png"
    if not zone_image.exists():
        print(f"❌ 区域图不存在: {zone_image}")
        return None

    components = zone_info.get("components", [])
    reusable_groups = manifest.get("reusable_groups", [])

    print(f"\n{'='*60}")
    print(f"🎨 测试 {zone_info.get('name', zone_name)}")
    print(f"{'='*60}")
    print(f"  组件数: {zone_info.get('component_count', 0)}")
    print(f"  组件类型: {zone_info.get('component_types', '?')}")
    for c in components:
        print(f"    📦 {c.get('name', '?')} ({c.get('type', '?')}) - {c.get('chinese_description', '')}")

    # ── 按可复用类型去重，每种类型取1个代表 ──
    # 建立组件名 -> 可复用类型的映射
    comp_name_to_group = {}
    for group in reusable_groups:
        group_type = group.get("type", "")
        for inst in group.get("all_instances", []):
            comp_name_to_group[inst.get("name", "")] = group_type

    # 为每个 zone component 找到可复用类型，去重
    unique_by_group = {}
    for comp in components:
        comp_name = comp.get("name", "")
        group_type = comp_name_to_group.get(comp_name, comp.get("type", ""))
        if group_type not in unique_by_group:
            unique_by_group[group_type] = comp

    unique_components = list(unique_by_group.values())
    total_unique = len(unique_components)

    print(f"\n  📋 可复用类型（{total_unique} 种）:")
    for c in unique_components:
        print(f"    📦 {c.get('name', '?')} ({c.get('type', '?')}) - {c.get('chinese_description', '')}")

    # ── 按4种类型一批分批 ──
    BATCH_SIZE = 4
    batches = []
    for i in range(0, total_unique, BATCH_SIZE):
        batch = unique_components[i:i + BATCH_SIZE]
        batches.append(batch)

    print(f"\n  📊 分为 {len(batches)} 批（每批最多 {BATCH_SIZE} 种类型）:")
    for i, batch in enumerate(batches):
        names = [c.get('name', '?') for c in batch]
        print(f"    批次 {i+1}: {', '.join(names)}")

    # ── 为每批生成四宫格 ──
    from phase2_module_recognition import generate_zone_prompt
    all_results = []

    for batch_idx, batch in enumerate(batches):
        batch_num = batch_idx + 1
        print(f"\n{'='*60}")
        print(f"🎨 批次 {batch_num}/{len(batches)}：{len(batch)} 种组件")
        print(f"{'='*60}")

        batch_prompt = generate_zone_prompt(
            zone_components=batch,
            zone_type=zone_info.get("type", "unknown"),
            zone_name=zone_info.get("name", "区域"),
            reusable_groups=reusable_groups,
        )

        print(f"\n  💬 Prompt ({len(batch_prompt)} 字符):")
        print(f"     {batch_prompt}")

        output_path = OUTPUT_DIR / f"quadrant_{zone_name}_batch{batch_num}.png"

        if output_path.exists():
            print(f"\n  ⏭️ 已存在，跳过: {output_path}")
            all_results.append({
                "batch": batch_num,
                "status": "skipped",
                "output": str(output_path),
                "components": [c.get('name', '?') for c in batch],
                "prompt": batch_prompt,
            })
            continue

        success = generate_quadrant_image(str(zone_image), batch_prompt, str(output_path))
        if success:
            print(f"  ✅ 生成成功: {output_path}")
            all_results.append({
                "batch": batch_num,
                "status": "generated",
                "output": str(output_path),
                "components": [c.get('name', '?') for c in batch],
                "prompt": batch_prompt,
            })
        else:
            print(f"  ❌ 生成失败")
            all_results.append({
                "batch": batch_num,
                "status": "failed",
                "components": [c.get('name', '?') for c in batch],
            })

    # ── 摘要 ──
    print(f"\n{'='*60}")
    print(f"📊 测试完成")
    print(f"{'='*60}")
    print(f"  区域: {zone_info.get('name', zone_name)}")
    print(f"  可复用类型数: {total_unique}")
    print(f"  批次数: {len(batches)}")
    for r in all_results:
        print(f"    批次 {r['batch']}: {r['status']} → {r.get('output', 'N/A')}")

    return {"zone": zone_name, "batches": all_results}


def main():
    zone_name = "zone_02_upper"
    test_all = False
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "all":
            test_all = True
        else:
            zone_name = arg if arg.startswith("zone_") else f"zone_{arg}"

    manifest_path = PHASE2_DIR / "manifest.json"
    if not manifest_path.exists():
        print(f"❌ manifest.json 不存在，请先运行 Phase2")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    floor_zones = manifest.get("floor_zones", [])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if test_all:
        results = []
        for zone in floor_zones:
            name = zone.get("cropped_file", "").replace(".png", "")
            if name:
                r = test_zone(name, manifest, floor_zones)
                if r:
                    results.append(r)
    else:
        test_zone(zone_name, manifest, floor_zones)


if __name__ == "__main__":
    main()