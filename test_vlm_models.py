"""
测试不同VLM模型的组件列表生成能力
==================================
对比多个模型在Phase 2.5增强版任务上的表现

使用方式：
  python test_vlm_models.py
"""

import os
import sys
import base64
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional

# ─── 路径配置 ───
PHASE2_OUTPUT_DIR = Path(__file__).parent / "output" / "phase2"
OUTPUT_DIR = Path(__file__).parent / "output" / "vlm_model_test"

# ─── 测试模型列表 ───
TEST_MODELS = [
    {
        "name": "gemini-3.1-flash-lite-preview",
        "api_url": "https://api.302.ai/v1/chat/completions",
        "api_key": "sk-W2Pie30RIDljwrF2PRLRQKQNzGf4a0topxoWOl2CIUaZ7v1K",
        "provider": "302.ai"
    },
    {
        "name": "qwen3.5-omni-plus",
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "api_key": "sk-24797c4574c64471b1b8b9914dfeff95",
        "provider": "DashScope"
    },
    {
        "name": "qwen3.6-plus",
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "api_key": "sk-24797c4574c64471b1b8b9914dfeff95",
        "provider": "DashScope"
    },
]


# ════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(ext, "image/jpeg")


# ════════════════════════════════════════════
# 组件清单摘要构造
# ════════════════════════════════════════════

def build_component_summary(manifest: dict, max_desc_chars: int = 80) -> str:
    """从 manifest 构造给质检 VLM 看的组件清单文本"""
    components = manifest.get("components", [])
    reusable_groups = manifest.get("reusable_groups", [])
    floor_zones = manifest.get("floor_zones", [])

    lines = []
    lines.append(f"总组件数：{len(components)}")
    lines.append(f"可复用组数：{len(reusable_groups)}")
    lines.append(f"楼层区域数：{len(floor_zones)}")

    fw = manifest.get("facade_width", 0)
    fh = manifest.get("facade_height", 0)
    if fw and fh:
        lines.append(f"图片尺寸：{fw}×{fh} 像素")
    lines.append("")

    if floor_zones:
        lines.append("── 楼层区域 ──")
        for z in floor_zones:
            zname = z.get("name", "?")
            zcount = z.get("component_count", 0)
            ztypes = z.get("component_types", "")
            lines.append(f"  {zname}: {zcount}个组件 ({ztypes})")
        lines.append("")

    lines.append("── 组件逐项清单 ──")
    lines.append("  格式: id | type | group | bbox[x1,y1,x2,y2] | 像素尺寸 | 描述")
    for c in components:
        cid = c.get("id", "?")
        ctype = c.get("type", "?")
        group = c.get("reusable_group", "?")
        bbox = c.get("bbox", [])
        desc = c.get("chinese_description", "")
        if len(desc) > max_desc_chars:
            desc = desc[:max_desc_chars] + "..."

        bbox_str = f"[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]" if len(bbox) == 4 else "?"
        if len(bbox) == 4:
            pw = bbox[2] - bbox[0]
            ph = bbox[3] - bbox[1]
            size_str = f"{pw}×{ph}px"
        else:
            size_str = "?"

        lines.append(f"  {cid} | {ctype} | {group} | {bbox_str} | {size_str} | {desc}")

    lines.append("")

    lines.append("── 可复用组摘要 ──")
    for g in reusable_groups:
        gtype = g.get("type", "?")
        base_type = g.get("base_type", "?")
        count = g.get("count", 0)
        desc = g.get("chinese_description", "")
        if len(desc) > max_desc_chars:
            desc = desc[:max_desc_chars] + "..."
        instances = g.get("all_instances", [])
        instance_ids = [i.get("id", "?") for i in instances]
        id_list = ",".join(instance_ids)
        lines.append(f"  {gtype} (base={base_type}) | {count}实例 [{id_list}] | {desc}")

    return "\n".join(lines)


def build_building_context(manifest: dict) -> str:
    """构造建筑上下文信息"""
    lines = []

    fw = manifest.get("facade_width", 0)
    fh = manifest.get("facade_height", 0)
    if fw and fh:
        lines.append(f"图片尺寸：{fw}×{fh} 像素")

    model = manifest.get("recognition_model", "unknown")
    lines.append(f"识别模型：{model}")

    total = manifest.get("total_components", 0)
    groups = manifest.get("total_reusable_types", 0)
    zones = manifest.get("total_floor_zones", 0)
    lines.append(f"识别结果：{total}个组件，{groups}个可复用组，{zones}个楼层区域")

    return "\n".join(lines)


# ════════════════════════════════════════════
# 简化的组件列表生成 Prompt
# ════════════════════════════════════════════

COMPONENT_LIST_PROMPT = """你是建筑组件识别专家。请仔细观察这张建筑立面图，识别出图中所有可见的建筑组件。

【任务要求】
1. 逐层扫描：从屋顶开始，到上层，最后到底层
2. 列出每个独立的建筑组件，包括：
   - 大面积组件：墙面(wall/facade)、屋顶(roof)、地面(base_wall)
   - 开口组件：窗户(window)、门(door)、阳台(balcony)
   - 装饰组件：遮阳篷(awning)、花箱(flower_box)、栏杆(railing)、装饰(decoration)
   - 功能组件：烟囱(chimney)、楼梯(staircase)、老虎窗(dormer)
3. 外观相同的多个组件（如一排窗户）要逐个列出
4. 组件总数通常在20-50个之间

【输出格式】
严格JSON格式，只输出这个对象：

{{
  "components": [
    {{
      "id": "comp_001",
      "type": "roof",
      "subtype": "slope",
      "name": "roof_left",
      "chinese_description": "左侧绿色瓦片屋顶",
      "bbox": [100, 50, 400, 200],
      "confidence": 0.95
    }},
    {{
      "id": "comp_002",
      "type": "wall",
      "subtype": "facade",
      "name": "wall_main",
      "chinese_description": "米色灰泥外墙",
      "bbox": [100, 200, 600, 800],
      "confidence": 0.9
    }}
  ],
  "total_count": 2,
  "type_distribution": "roof×1, wall×1"
}}

【重要】
- 必须列出所有可见组件，不能遗漏
- bbox 是像素坐标 [x1, y1, x2, y2]
- 每个组件都要有唯一的 id
- 输出完成后，检查一遍：窗户数量、门数量、屋顶数量是否与图片一致"""


# ════════════════════════════════════════════
# VLM 调用
# ════════════════════════════════════════════

def call_vlm(api_url: str, api_key: str, model: str, image_b64: str, 
             media_type: str, prompt: str, max_retries: int = 3) -> dict:
    """调用 VLM，返回 JSON 结果"""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 16000,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"  🔍 模型: {model}")

    import time
    for attempt in range(1, max_retries + 1):
        print(f"  🔍 调用 VLM... (尝试 {attempt}/{max_retries})")
        try:
            resp = requests.post(api_url, json=payload, headers=headers, timeout=300)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            # 提取 JSON
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            # 尝试解析
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # 尝试修复截断的 JSON
                for trim_pos in range(len(content) - 1, -1, -1):
                    if content[trim_pos] == '}':
                        trial = content[:trim_pos + 1].rstrip().rstrip(',')
                        open_brackets = trial.count('[') - trial.count(']')
                        open_braces = trial.count('{') - trial.count('}')
                        trial += ']' * max(open_brackets, 0) + '}' * max(open_braces, 0)
                        try:
                            return json.loads(trial)
                        except json.JSONDecodeError:
                            continue
                raise

        except requests.exceptions.ReadTimeout:
            print(f"  ⚠️  请求超时 (第 {attempt} 次)")
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"     等待 {wait} 秒后重试...")
                time.sleep(wait)
            else:
                raise
        except requests.exceptions.ConnectionError as e:
            print(f"  ⚠️  连接错误 (第 {attempt} 次): {e}")
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"     等待 {wait} 秒后重试...")
                time.sleep(wait)
            else:
                raise


# ════════════════════════════════════════════
# 测试流程
# ════════════════════════════════════════════

def test_model(model_config: dict, ortho_path: str, manifest: dict) -> dict:
    """测试单个模型"""
    model_name = model_config["name"]
    api_url = model_config["api_url"]
    api_key = model_config["api_key"]

    print(f"\n{'='*60}")
    print(f"🧪 测试模型: {model_name}")
    print(f"{'='*60}")

    # 编码图片
    print(f"  📷 编码图片...")
    ortho_b64 = encode_image_to_base64(ortho_path)
    media_type = get_image_media_type(ortho_path)
    print(f"  🖼️  base64 长度: {len(ortho_b64)} 字符")

    # 调用 VLM
    print(f"\n  📡 调用 VLM...")
    try:
        result = call_vlm(api_url, api_key, model_name, ortho_b64, media_type, COMPONENT_LIST_PROMPT)
    except Exception as e:
        print(f"  ❌ 调用失败: {e}")
        return {
            "model": model_name,
            "success": False,
            "error": str(e),
            "component_count": 0
        }

    # 分析结果
    components = result.get("components", [])
    component_count = len(components)

    # 类型分布
    type_counts = {}
    for c in components:
        t = c.get("type", "?")
        type_counts[t] = type_counts.get(t, 0) + 1
    type_dist = ", ".join(f"{t}×{n}" for t, n in sorted(type_counts.items()))

    print(f"\n  ✅ 测试成功!")
    print(f"  📦 组件数量: {component_count}")
    print(f"  📊 类型分布: {type_dist}")

    # 保存结果
    output_path = OUTPUT_DIR / f"test_{model_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  💾 结果已保存: {output_path.name}")

    return {
        "model": model_name,
        "success": True,
        "component_count": component_count,
        "type_distribution": type_dist,
        "components": components
    }


def main():
    """主入口"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 选择测试模块
    module_dirs = sorted(
        [d for d in PHASE2_OUTPUT_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name
    )

    if not module_dirs:
        print(f"⚠️  Phase 2 输出目录为空")
        return

    print(f"📁 找到 {len(module_dirs)} 个 Phase 2 模块：\n")
    for i, d in enumerate(module_dirs, 1):
        mf = d / "manifest.json"
        if mf.exists():
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                total = data.get("total_components", "?")
                print(f"  [{i}] {d.name}  组件={total}")
            except Exception:
                print(f"  [{i}] {d.name}  (manifest读取失败)")
        else:
            print(f"  [{i}] {d.name}  (无manifest)")

    # 用户选择
    try:
        choice = input(f"\n请选择要测试的模块编号 (默认=1): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消")
        return

    if not choice:
        choice = "1"

    try:
        idx = int(choice)
        if idx < 1 or idx > len(module_dirs):
            print(f"❌ 无效编号: {idx}")
            return
        selected_dir = module_dirs[idx - 1]
    except ValueError:
        print(f"❌ 无效输入: {choice}")
        return

    # 加载 manifest
    manifest_path = selected_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"❌ manifest.json 不存在")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 获取正交原图路径
    ortho_path = manifest.get("source", "")
    if not ortho_path or not Path(ortho_path).exists():
        print(f"❌ 正交原图不存在: {ortho_path}")
        return

    print(f"\n📷 正交原图: {Path(ortho_path).name}")
    print(f"📦 组件数: {manifest.get('total_components', 0)}")

    # 测试所有模型
    results = []
    for model_config in TEST_MODELS:
        result = test_model(model_config, ortho_path, manifest)
        results.append(result)

    # 汇总对比
    print(f"\n{'='*60}")
    print(f"📊 模型对比汇总")
    print(f"{'='*60}")

    print(f"\n  {'模型':<35} {'组件数':<10} {'状态':<10}")
    print(f"  {'-'*55}")
    
    phase2_count = manifest.get('total_components', 0)
    for r in results:
        model = r["model"]
        count = r["component_count"]
        status = "✅" if r["success"] else "❌"
        ratio = f"{count}/{phase2_count}" if r["success"] else "N/A"
        print(f"  {model:<35} {ratio:<10} {status}")

    print(f"\n  Phase 2 原始组件数: {phase2_count}")

    # 找出最佳模型
    successful = [r for r in results if r["success"]]
    if successful:
        best = max(successful, key=lambda x: x["component_count"])
        print(f"\n  🏆 最佳模型: {best['model']}")
        print(f"     组件数: {best['component_count']}")
        print(f"     类型分布: {best['type_distribution']}")


if __name__ == "__main__":
    main()
