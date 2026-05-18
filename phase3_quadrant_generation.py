"""
Phase 3: 四宫格生成
===========================
输入：phase2_module_recognition.py 输出的 manifest（含 reusable_groups 和 floor_zones）
输出：2x2 四宫格 45 度角正交图

流程：
  1. 读取 Phase 2 的 manifest.json
  2. 对每个楼层区域，按 reusable_group 跨 zone 去重
     - zone.components 包含所有实例，先按 reusable_group 合并为唯一类型列表
     - 跨 zone：如果某 reusable_group 已在上方 zone 生成过，则在下方 zone 中跳过
     - 结果：所有 21 种 reusable_group 每种恰好生成一次四宫格
  3. 超过4种类型时分批，每批最多4种（四宫格只有4格）
  4. 为每批使用 Phase2 的 generation_prompt 生成提示词，调用图生图 API 生成四宫格
  5. 将生成结果保存到 Phase3 自己的 manifest.json

使用方式：
  python phase3_quadrant_generation.py

依赖：
  - config.py 中的 IMAGE_EDIT_API_URL, IMAGE_EDIT_API_KEY, IMAGE_EDIT_MODEL
  - phase2_module_recognition.py 输出的 manifest.json
    components 含 generation_prompt，reusable_groups 含可复用类型分组
"""

import os
import sys
import base64
import json
import random
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
from datetime import datetime

# --- 路径配置 ---
PHASE2_OUTPUT_DIR = Path(__file__).parent / "output" / "phase2"
OUTPUT_DIR = Path(__file__).parent / "output" / "phase3_quadrant"

# --- 常量 ---
BATCH_SIZE = 4  # 每批最多4种可复用类型

# --- 固定提示词模板 ---
PROMPT_TEMPLATE = """参考建筑立面图，生成2x2四宫格模块拆分正交视图。

【视角要求】
固定视角：每个组件都必须从"正面偏左侧45度"观察，看到正面和左侧两个面。
所有四个格子的视角方向完全一致，禁止出现朝右的组件。
等轴测投影，无透视变形。

【构图要求】
- 纯白背景
- 四个格子均匀分布，格子间有清晰分割线
- 组件居中展示，占格子85%以上空间

【四个组件】
{components}

【风格要求】
- 哥特风格游戏资产：干净边缘，扁平着色
- 组件结构完整，可独立作为模块使用
- 同一建筑风格，与参考图一致"""


# 工具函数

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
        print(f"    X 下载失败: {e}")
        return False


# 去重和分批逻辑

def deduplicate_components_by_reusable_type(
    components: List[Dict],
    reusable_groups: List[Dict]
) -> List[Dict]:
    """
    按可复用类型对组件去重，每种类型取1个代表
    
    Args:
        components: 当前区域的组件列表
        reusable_groups: 可复用分组列表
    
    Returns:
        去重后的组件列表
    """
    # 建立组件名 -> 可复用类型的映射
    comp_name_to_group = {}
    for group in reusable_groups:
        group_type = group.get("type", "")
        for inst in group.get("all_instances", []):
            comp_name_to_group[inst.get("name", "")] = group_type

    # 按可复用类型去重
    unique_by_group = {}
    for comp in components:
        comp_name = comp.get("name", "")
        group_type = comp_name_to_group.get(comp_name, comp.get("type", ""))
        if group_type not in unique_by_group:
            unique_by_group[group_type] = comp

    return list(unique_by_group.values())


def split_into_batches(components: List[Dict], batch_size: int = BATCH_SIZE) -> List[List[Dict]]:
    """
    将组件列表按 batch_size 分批
    
    Args:
        components: 组件列表
        batch_size: 每批数量
    
    Returns:
        分批后的列表
    """
    if len(components) <= batch_size:
        return [components]
    return [components[i:i + batch_size] for i in range(0, len(components), batch_size)]


# 图生图 - 生成四宫格

def generate_quadrant_image(
    source_image_path: str,
    prompt: str,
    output_path: str,
    max_retries: int = 3,
) -> bool:
    """调用 DashScope Qwen 图像编辑 API 生成四宫格（带重试机制）"""
    from config import IMAGE_EDIT_API_URL, IMAGE_EDIT_API_KEY, IMAGE_EDIT_MODEL
    import time

    # 编码图片为 base64 data URI
    b64 = encode_image_to_base64(source_image_path)
    media_type = get_image_media_type(source_image_path)
    data_uri = f"data:{media_type};base64,{b64}"

    # 构建 DashScope 多模态生成请求体
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
            "seed": random.randint(1, 999999)
        }
    }

    headers = {
        "Authorization": f"Bearer {IMAGE_EDIT_API_KEY}",
        "Content-Type": "application/json"
    }

    for attempt in range(1, max_retries + 1):
        print(f"  调用 DashScope Qwen 图像编辑... (尝试 {attempt}/{max_retries})")
        try:
            resp = requests.post(
                IMAGE_EDIT_API_URL,
                json=payload,
                headers=headers,
                timeout=300,
            )

            if resp.status_code != 200:
                print(f"  生成失败 (HTTP {resp.status_code})")
                print(f"  错误: {resp.text[:500]}")
                if attempt < max_retries:
                    wait = 15 * attempt
                    print(f"     等待 {wait} 秒后重试...")
                    time.sleep(wait)
                    continue
                return False

            result = resp.json()

            # 解析 DashScope 多模态生成响应格式
            output = result.get("output", {})
            choices = output.get("choices", [])

            if not choices:
                print(f"  响应中无 choices: {resp.text[:300]}")
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

            print(f"  响应中无图片数据")
            if attempt < max_retries:
                wait = 15 * attempt
                print(f"     等待 {wait} 秒后重试...")
                time.sleep(wait)
                continue
            return False

        except requests.exceptions.ReadTimeout:
            print(f"  请求超时 (第 {attempt} 次)")
            if attempt < max_retries:
                wait = 15 * attempt
                print(f"     等待 {wait} 秒后重试...")
                time.sleep(wait)
            else:
                print(f"  已达最大重试次数")
                return False
        except requests.exceptions.ConnectionError as e:
            print(f"  连接错误 (第 {attempt} 次): {e}")
            if attempt < max_retries:
                wait = 15 * attempt
                print(f"     等待 {wait} 秒后重试...")
                time.sleep(wait)
            else:
                print(f"  已达最大重试次数")
                return False

    return False


# 主流程

def process_zone(
    module_dir: Path,
    zone: Dict,
    zone_index: int,
    manifest: Dict,
    overwrite: bool = False,
    skip_groups: set = None,
) -> List[Dict]:
    """
    处理单个楼层区域，按可复用类型去重后分批生成四宫格
    
    Args:
        module_dir: Phase2 输出的模块目录
        zone: 楼层区域数据（来自 manifest.floor_zones）
        zone_index: 当前区域索引（0-based）
        manifest: 完整的 manifest 数据（用于读取 reusable_groups）
        overwrite: 是否覆盖已有结果
        skip_groups: 跨 zone 去重：已在此前 zone 中生成过的 reusable_group 集合
    
    Returns:
        该区域所有批次的处理结果列表
    """
    if skip_groups is None:
        skip_groups = set()
    
    zone_type = zone.get("type", "unknown")
    zone_name = zone.get("name", f"区域{zone_index+1}")
    zone_file = zone.get("cropped_file", f"zone_{zone_index+1:02d}_{zone_type}.png")
    components = zone.get("components", [])
    reusable_groups = manifest.get("reusable_groups", [])
    
    print(f"\n{'='*60}")
    print(f"处理 {zone_name} ({zone_type})")
    print(f"{'='*60}")
    print(f"  组件数: {len(components)}")
    for c in components:
        print(f"    {c.get('name', '?')} ({c.get('type', '?')}) - {c.get('chinese_description', '')}")
    
    # -- 按可复用类型去重 --
    unique_components = deduplicate_components_by_reusable_type(components, reusable_groups)
    
    # -- 跨 zone 去重：移除已在上方 zone 中生成过的类型 --
    # 建立组件名 -> 可复用类型的映射
    comp_name_to_group = {}
    for group in reusable_groups:
        group_type = group.get("type", "")
        for inst in group.get("all_instances", []):
            comp_name_to_group[inst.get("name", "")] = group_type
    
    remaining_components = []
    skipped_groups = []
    for comp in unique_components:
        comp_name = comp.get("name", "")
        group_type = comp_name_to_group.get(comp_name, comp.get("type", ""))
        if group_type in skip_groups:
            skipped_groups.append(group_type)
        else:
            remaining_components.append(comp)
    
    if skipped_groups:
        print(f"\n  跨 zone 跳过 {len(skipped_groups)} 种已生成的类型: {', '.join(skipped_groups)}")
    
    unique_components = remaining_components
    total_unique = len(unique_components)
    
    if total_unique == 0:
        print(f"\n  该区域无新类型需要生成，跳过")
        return [{
            "zone_type": zone_type,
            "zone_name": zone_name,
            "batches": [{
                "batch_index": 0,
                "status": "skipped",
                "reason": "all_types_already_generated_in_previous_zones",
                "skipped_groups": skipped_groups,
            }]
        }]
    
    print(f"\n  本区域新类型（{total_unique} 种）:")
    for c in unique_components:
        print(f"    {c.get('name', '?')} ({c.get('type', '?')}) - {c.get('chinese_description', '')}")
    
    # -- 检查区域图是否存在 --
    zone_path = module_dir / zone_file
    if not zone_path.exists():
        print(f"  区域图不存在: {zone_file}")
        return [{
            "zone_type": zone_type,
            "zone_name": zone_name,
            "batches": [{
                "batch_index": 1,
                "status": "skipped",
                "reason": "zone_image_missing",
                "components_count": len(components),
            }]
        }]
    
    # -- 分批 --
    batches = split_into_batches(unique_components, BATCH_SIZE)
    num_batches = len(batches)
    
    print(f"\n  分为 {num_batches} 批（每批最多 {BATCH_SIZE} 种类型）:")
    for i, batch in enumerate(batches):
        names = [c.get('name', '?') for c in batch]
        print(f"    批次 {i+1}: {', '.join(names)}")
    
    # -- 为每批生成四宫格 --
    output_subdir = OUTPUT_DIR / module_dir.name
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    batch_results = []
    
    for batch_idx, batch in enumerate(batches):
        batch_num = batch_idx + 1
        
        # 输出文件名
        if num_batches == 1:
            output_filename = f"quadrant_{zone_index+1:02d}_{zone_type}.png"
        else:
            output_filename = f"quadrant_{zone_index+1:02d}_{zone_type}_batch{batch_num}.png"
        
        output_path = output_subdir / output_filename
        
        # -- 文件存在检查，避免重复生成 --
        if output_path.exists() and not overwrite:
            print(f"\n  批次 {batch_num}/{num_batches}：已存在，跳过: {output_filename}")
            batch_results.append({
                "batch_index": batch_num,
                "status": "skipped",
                "output_file": output_filename,
                "components": [c.get('name', '?') for c in batch],
            })
            continue
        
        # 批次间延迟，避免 API 限流
        if batch_idx > 0:
            import time
            print(f"\n  等待 10 秒避免限流...")
            time.sleep(10)
        
        print(f"\n{'='*60}")
        print(f"批次 {batch_num}/{num_batches}：{len(batch)} 种组件")
        print(f"{'='*60}")
        
        # -- 生成提示词（固定模板） --
        positions = ["左上", "右上", "左下", "右下"]
        comp_lines = []
        for j, comp in enumerate(batch[:4]):
            label = comp.get("generation_prompt", comp.get("chinese_description", comp.get("type", "组件")))
            comp_lines.append(f"{j+1}. {positions[j]}：{label}")
        # 不足4个用区域整体视角补齐
        while len(comp_lines) < 4:
            idx = len(comp_lines)
            comp_lines.append(f"{idx+1}. {positions[idx]}：{zone_name}整体视角 - 展示该区域建筑构件的完整侧面结构")
        batch_prompt = PROMPT_TEMPLATE.format(components="\n".join(comp_lines))
        
        print(f"\n  Prompt ({len(batch_prompt)} 字符):")
        print(f"     {batch_prompt}")
        
        # -- 调用图生图 API --
        success = generate_quadrant_image(
            str(zone_path),
            batch_prompt,
            str(output_path)
        )
        
        if success:
            print(f"  生成成功: {output_filename}")
            
            batch_results.append({
                "batch_index": batch_num,
                "status": "generated",
                "output_file": output_filename,
                "components": [c.get('name', '?') for c in batch],
                "prompt": batch_prompt,
            })
        else:
            print(f"  生成失败")
            batch_results.append({
                "batch_index": batch_num,
                "status": "failed",
                "components": [c.get('name', '?') for c in batch],
            })
    
    return [{
        "zone_type": zone_type,
        "zone_name": zone_name,
        "batches": batch_results,
    }]


def process_module(module_dir: Path, overwrite: bool = False) -> Dict:
    """
    处理单个模块目录
    
    Args:
        module_dir: Phase2 输出的模块目录
        overwrite: 是否覆盖已有结果
    
    Returns:
        处理结果
    """
    manifest_path = module_dir / "manifest.json"
    
    if not manifest_path.exists():
        print(f"  manifest.json 不存在，跳过: {module_dir.name}")
        return None
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    # 获取楼层区域信息
    floor_zones = manifest.get("floor_zones", [])
    reusable_groups = manifest.get("reusable_groups", [])
    
    print(f"\n{'='*60}")
    print(f"Phase 3: 四宫格生成 - {module_dir.name}")
    print(f"{'='*60}")
    print(f"  楼层区域: {len(floor_zones)} 个")
    for zone in floor_zones:
        print(f"    {zone.get('name', '?')}  含 {zone.get('component_count', 0)} 个组件")
    print(f"  可复用类型: {len(reusable_groups)} 种")
    
    # 创建输出目录
    output_subdir = OUTPUT_DIR / module_dir.name
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个楼层区域（跨 zone 去重）
    all_zone_results = []
    processed_groups = set()  # 跨 zone 跟踪已处理的 reusable_group
    
    # 建立组件名 -> 可复用类型的映射（用于跨 zone 跟踪）
    comp_name_to_group = {}
    for group in reusable_groups:
        group_type = group.get("type", "")
        for inst in group.get("all_instances", []):
            comp_name_to_group[inst.get("name", "")] = group_type
    
    for i, zone in enumerate(floor_zones):
        # 区域间延迟
        if i > 0:
            import time
            print(f"\n  区域间等待 5 秒...")
            time.sleep(5)
        
        zone_results = process_zone(
            module_dir=module_dir,
            zone=zone,
            zone_index=i,
            manifest=manifest,
            overwrite=overwrite,
            skip_groups=processed_groups,
        )
        all_zone_results.extend(zone_results)
        
        # 更新已处理的 reusable_group 集合
        # 从该 zone 的结果中提取实际生成的类型
        for zone_result in zone_results:
            for batch in zone_result.get("batches", []):
                if batch.get("status") == "generated":
                    for comp_name in batch.get("components", []):
                        group_type = comp_name_to_group.get(comp_name, "")
                        if group_type:
                            processed_groups.add(group_type)
    
    print(f"\n  跨 zone 去重：共处理 {len(processed_groups)} 种 unique reusable_group")
    print(f"  总可复用类型: {len(reusable_groups)} 种")
    if len(processed_groups) < len(reusable_groups):
        missing = set(g.get("type") for g in reusable_groups) - processed_groups
        print(f"  未覆盖的类型（可能因生成失败）: {', '.join(missing)}")
    
    # 统计结果
    total_batches = 0
    generated_count = 0
    failed_count = 0
    skipped_count = 0
    
    for zone_result in all_zone_results:
        for batch in zone_result.get("batches", []):
            total_batches += 1
            status = batch.get("status")
            if status == "generated":
                generated_count += 1
            elif status == "failed":
                failed_count += 1
            elif status == "skipped":
                skipped_count += 1
    
    # 保存 phase3 输出 manifest
    output_manifest = {
        "source_module": str(module_dir),
        "total_zones": len(floor_zones),
        "total_batches": total_batches,
        "generated_count": generated_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "zone_results": all_zone_results,
        "next_phase": "phase4_filter" if not failed_count else "needs_rework",
    }
    
    output_manifest_path = output_subdir / "manifest.json"
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(output_manifest, f, ensure_ascii=False, indent=2)
    
    # 打印摘要
    print(f"\n{'='*60}")
    print(f"Phase 3 处理完成")
    print(f"{'='*60}")
    print(f"  模块: {module_dir.name}")
    print(f"  楼层区域数: {len(floor_zones)}")
    print(f"  总批次数: {total_batches}")
    print(f"  成功生成: {generated_count}")
    print(f"  失败: {failed_count}")
    print(f"  跳过: {skipped_count}")
    
    return output_manifest


def main():
    """主入口"""
    # 确保目录存在
    PHASE2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 查找 Phase 2 的输出目录
    module_dirs = [d for d in PHASE2_OUTPUT_DIR.iterdir() if d.is_dir()]
    
    if not module_dirs:
        print(f"Phase 2 输出目录为空: {PHASE2_OUTPUT_DIR}")
        print(f"   请先运行 phase2_module_recognition.py")
        sys.exit(0)
    
    print(f"找到 {len(module_dirs)} 个模块目录")
    
    # 处理每个模块
    results = []
    for module_dir in module_dirs:
        result = process_module(module_dir)
        if result:
            results.append(result)
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"Phase 3 四宫格生成全部完成")
    print(f"{'='*60}")
    print(f"  处理模块数: {len(results)}")
    print(f"  输出目录: {OUTPUT_DIR}")
    
    total_generated = sum(r.get("generated_count", 0) for r in results)
    total_failed = sum(r.get("failed_count", 0) for r in results)
    print(f"  总生成数: {total_generated}")
    print(f"  总失败数: {total_failed}")


if __name__ == "__main__":
    main()