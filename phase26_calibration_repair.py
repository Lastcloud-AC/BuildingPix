"""
Phase 2.6: 校准+代码修复
===========================
输入：Phase 2 的 manifest.json + Phase 2.5 的 quality_report
输出：修复后的 manifest.json（回写覆盖 Phase 2 目录）

核心能力：
  - merged_components → 纯代码规则拆分（零 VLM 成本）
  - missing_components → 标记警告（代码无法凭空生成）
  - grouping_issues → 标记警告（后续向量模型处理）
  - type_errors → 标记警告（后续向量模型处理）

使用方式：
  python phase26_calibration_repair.py

依赖：
  - phase2_module_recognition.py 的 crop_component() / build_reusable_groups()
  - 无需 API 调用
"""

import os
import sys
import json
import copy
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

# ════════════════════════════════════════════
# 常量
# ════════════════════════════════════════════

PHASE2_OUTPUT_DIR = Path(__file__).parent / "output" / "phase2"
PHASE25_OUTPUT_DIR = Path(__file__).parent / "output" / "phase25"

# ── 拆分规则 ──
# (上方组件type, 下方组件type): {"split": "horizontal", "ratio": 上方占比}
MERGE_SPLIT_RULES = {
    ("canopy", "shopfront"):  {"split": "horizontal", "ratio": 0.35},
    ("window", "balcony"):    {"split": "horizontal", "ratio": 0.75},
    ("window", "flower_box"): {"split": "horizontal", "ratio": 0.85},
}

# ── 关键词 → 组件类型映射 ──
KEYWORD_TO_TYPE = {
    "遮阳篷": "canopy", "雨棚": "canopy", "awning": "canopy",
    "遮阳棚": "canopy", "遮阳板": "canopy",
    "橱窗": "shopfront", "商铺": "shopfront", "展示窗": "shopfront",
    "阳台": "balcony", "栏杆": "balcony",
    "花箱": "flower_box", "花台": "flower_box", "花盆": "flower_box",
    "窗台": "windowsill", "窗框": "window_frame",
    "门廊": "porch", "门斗": "porch",
    "石质装饰": "molding", "线脚": "molding", "装饰线": "molding",
}

# ── 子组件描述模板 ──
SUB_COMPONENT_TEMPLATES = {
    "canopy": {
        "cn": "遮阳篷",
        "prompt_template": "单个{color}{material}遮阳篷，45度等轴游戏建筑资产",
    },
    "shopfront": {
        "cn": "商铺橱窗",
        "prompt_template": "单个{color}{material}商铺橱窗，展示窗，45度等轴游戏建筑资产",
    },
    "balcony": {
        "cn": "阳台",
        "prompt_template": "单个{color}{material}阳台，带栏杆，45度等轴游戏建筑资产",
    },
    "flower_box": {
        "cn": "花箱",
        "prompt_template": "单个{color}{material}花箱，45度等轴游戏建筑资产",
    },
    "window": {
        "cn": "窗户",
        "prompt_template": "单个{color}{material}窗户，45度等轴游戏建筑资产",
    },
    "molding": {
        "cn": "装饰线脚",
        "prompt_template": "单个{color}{material}装饰线脚，45度等轴游戏建筑资产",
    },
}

# ── 颜色/材质提取关键词 ──
COLOR_KEYWORDS = [
    "红色", "蓝色", "绿色", "白色", "米色", "灰色", "棕色", "黑色",
    "金色", "银色", "深色", "浅色", "暖色", "冷色", "条纹",
    "红", "蓝", "绿", "白", "米", "灰", "棕", "黑", "金", "银",
]
MATERIAL_KEYWORDS = [
    "木质", "石质", "金属", "铁艺", "玻璃", "砖", "瓦", "陶土",
    "木", "石", "铁", "铜", "钢", "花岗岩", "大理石",
    "砖石", "灰泥", "石膏",
]


# ════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════

def load_inputs(phase2_dir: Path, phase25_report_path: Path) -> Tuple[dict, dict]:
    """加载 manifest.json 和 quality_report.json"""
    manifest_path = phase2_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json 不存在: {manifest_path}")
    if not phase25_report_path.exists():
        raise FileNotFoundError(f"质检报告不存在: {phase25_report_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(phase25_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    return manifest, report


def extract_color_from_desc(description: str) -> str:
    """从中文描述中提取颜色词"""
    for kw in COLOR_KEYWORDS:
        if kw in description:
            return kw
    return ""


def extract_material_from_desc(description: str) -> str:
    """从中文描述中提取材质词"""
    for kw in MATERIAL_KEYWORDS:
        if kw in description:
            return kw
    return ""


# ════════════════════════════════════════════
# Step 1: 规则匹配
# ════════════════════════════════════════════

def match_merge_rule(
    merged_issue: dict,
    manifest: dict,
) -> Optional[Tuple[str, str, float]]:
    """
    从 merged_issue 匹配 MERGE_SPLIT_RULES

    Args:
        merged_issue: 质检报告中的 merged_components 条目
        manifest: Phase 2 manifest

    Returns:
        (upper_type, lower_type, ratio) 或 None
    """
    component_id = merged_issue.get("component_id", "")
    description = merged_issue.get("description", "")

    # ── 找到原始组件的 type ──
    original_type = None
    for comp in manifest.get("components", []):
        if comp.get("id") == component_id:
            original_type = comp.get("type", "")
            break

    if not original_type:
        print(f"    ⚠️  组件 {component_id} 在 manifest 中找不到")
        return None

    # ── 从 description 中提取被合并的类型 ──
    detected_types = []
    for keyword, comp_type in KEYWORD_TO_TYPE.items():
        if keyword in description:
            detected_types.append(comp_type)

    if not detected_types:
        print(f"    ⚠️  无法从描述中识别被合并类型: {description[:60]}")
        return None

    # ── 在 MERGE_SPLIT_RULES 中查找匹配 ──
    for detected_type in detected_types:
        # 尝试两种排列：(original, detected) 和 (detected, original)
        key_a = (original_type, detected_type)
        key_b = (detected_type, original_type)

        if key_a in MERGE_SPLIT_RULES:
            rule = MERGE_SPLIT_RULES[key_a]
            return (original_type, detected_type, rule["ratio"])

        if key_b in MERGE_SPLIT_RULES:
            rule = MERGE_SPLIT_RULES[key_b]
            return (detected_type, original_type, rule["ratio"])

    # ── 兜底：用 original_type 作为被包含的类型 ──
    # 例如 comp_017 type=shopfront，描述含"遮阳篷"→ 查 (canopy, shopfront)
    for detected_type in detected_types:
        for (t_a, t_b), rule in MERGE_SPLIT_RULES.items():
            if original_type in (t_a, t_b) and detected_type in (t_a, t_b):
                return (t_a, t_b, rule["ratio"])

    print(f"    ⚠️  规则表中无匹配: original={original_type}, detected={detected_types}")
    return None


# ════════════════════════════════════════════
# Step 2: bbox 切分 + 裁剪
# ════════════════════════════════════════════

def split_bbox_horizontal(bbox: List[float], ratio: float) -> Tuple[List[float], List[float]]:
    """水平切分 bbox，ratio 是上方组件的高度占比"""
    x1, y1, x2, y2 = bbox
    split_y = y1 + (y2 - y1) * ratio
    upper_bbox = [x1, y1, x2, split_y]
    lower_bbox = [x1, split_y, x2, y2]
    return upper_bbox, lower_bbox


def crop_from_image(
    image_path: str,
    bbox: List[float],
    output_path: str,
    min_size: int = 20,
) -> bool:
    """
    从原图裁剪指定 bbox 区域（绝对像素坐标版本）

    Args:
        image_path: 原图路径
        bbox: [x1, y1, x2, y2] 绝对像素坐标
        output_path: 输出 PNG 路径
        min_size: 最小尺寸阈值

    Returns:
        是否成功
    """
    try:
        with Image.open(image_path) as img:
            left = max(0, int(bbox[0]))
            top = max(0, int(bbox[1]))
            right = min(img.width, int(bbox[2]))
            bottom = min(img.height, int(bbox[3]))

            if right - left < min_size or bottom - top < min_size:
                print(f"    ⚠️  裁剪区域太小 ({right-left}x{bottom-top}px)，跳过")
                return False

            cropped = img.crop((left, top, right, bottom))
            cropped.save(output_path, "PNG")
            return True
    except Exception as e:
        print(f"    ❌ 裁剪失败: {e}")
        return False


def split_component(
    ortho_image_path: str,
    component: dict,
    upper_type: str,
    lower_type: str,
    ratio: float,
    output_dir: Path,
    new_id_prefix: str,
) -> Tuple[Optional[dict], Optional[dict]]:
    """
    拆分单个 merged 组件为上下两个子组件

    Args:
        ortho_image_path: 正交原图路径
        component: 原始组件 dict
        upper_type: 上方组件类型
        lower_type: 下方组件类型
        ratio: 上方高度占比
        output_dir: 输出目录
        new_id_prefix: 新组件 ID 前缀（如 "017"）

    Returns:
        (upper_component, lower_component) 或 (None, None)
    """
    bbox = component.get("bbox", [])
    if len(bbox) != 4:
        print(f"    ⚠️  组件 bbox 格式异常: {bbox}")
        return None, None

    upper_bbox, lower_bbox = split_bbox_horizontal(bbox, ratio)

    # ── 生成子组件裁剪图 ──
    upper_file = f"{new_id_prefix}_upper_{upper_type}.png"
    lower_file = f"{new_id_prefix}_lower_{lower_type}.png"
    upper_path = output_dir / upper_file
    lower_path = output_dir / lower_file

    upper_ok = crop_from_image(ortho_image_path, upper_bbox, str(upper_path))
    lower_ok = crop_from_image(ortho_image_path, lower_bbox, str(lower_path))

    if not upper_ok and not lower_ok:
        print(f"    ❌ 两个子组件裁剪均失败")
        return None, None

    # ── 生成子组件 metadata ──
    upper_comp = None
    lower_comp = None

    if upper_ok:
        upper_comp = generate_sub_component(
            original=component,
            split_type=upper_type,
            new_bbox=upper_bbox,
            new_cropped_file=upper_file,
            position="upper",
            new_id=f"{new_id_prefix}_upper",
        )

    if lower_ok:
        lower_comp = generate_sub_component(
            original=component,
            split_type=lower_type,
            new_bbox=lower_bbox,
            new_cropped_file=lower_file,
            position="lower",
            new_id=f"{new_id_prefix}_lower",
        )

    return upper_comp, lower_comp


# ════════════════════════════════════════════
# Step 3: 子组件 metadata 生成
# ════════════════════════════════════════════

def generate_sub_component(
    original: dict,
    split_type: str,
    new_bbox: List[float],
    new_cropped_file: str,
    position: str,
    new_id: str,
) -> dict:
    """
    为切分后的子组件生成完整 metadata

    Args:
        original: 原始组件 dict
        split_type: 子组件类型 (canopy/shopfront/balcony/flower_box)
        new_bbox: 子组件 bbox
        new_cropped_file: 子组件裁剪文件名
        position: "upper" 或 "lower"
        new_id: 新组件 ID

    Returns:
        完整的子组件 dict
    """
    # ── 提取原组件的颜色和材质 ──
    orig_desc = original.get("chinese_description", "")
    color = extract_color_from_desc(orig_desc)
    material = extract_material_from_desc(orig_desc)

    # ── 用模板生成描述和 prompt ──
    template = SUB_COMPONENT_TEMPLATES.get(split_type, {})
    prompt_tmpl = template.get("prompt_template", f"单个{split_type}，45度等轴游戏建筑资产")

    generation_prompt = prompt_tmpl.format(
        color=f"{color}" if color else "",
        material=f"{material}" if material else "",
    )
    # 清理多余的空格
    generation_prompt = generation_prompt.replace("  ", " ").strip()

    cn_name = template.get("cn", split_type)
    chinese_description = f"{color}{material}{cn_name}" if (color or material) else cn_name

    # ── 构建 subtype 和 name ──
    orig_subtype = original.get("subtype", "")
    subtype = orig_subtype if orig_subtype else split_type
    orig_name = original.get("name", "")
    # 从原名提取位置信息
    position_suffix = ""
    for suffix in ["left", "right", "center", "main"]:
        if suffix in orig_name:
            position_suffix = f"_{suffix}"
            break
    name = f"{split_type}_{subtype}_{position}_{original.get('id', '')}"

    # ── 继承原组件的部分字段 ──
    new_comp = {
        "id": new_id,
        "type": split_type,
        "subtype": subtype,
        "name": name,
        "reusable_group": f"{split_type}_{subtype}",
        "chinese_description": chinese_description,
        "generation_prompt": generation_prompt,
        "material": original.get("material", ""),
        "color": original.get("color", ""),
        "bbox": [int(b) for b in new_bbox],
        "confidence": original.get("confidence", 0.8),
        "cropped_file": new_cropped_file,
        "original_bbox": [int(b) for b in new_bbox],
        "_phase26_split_from": original.get("id", ""),
        "_phase26_position": position,
    }

    return new_comp


# ════════════════════════════════════════════
# Step 4: manifest 更新
# ════════════════════════════════════════════

def update_manifest(
    manifest: dict,
    repairs: List[dict],
    warnings: List[dict],
) -> dict:
    """
    更新 manifest：替换被拆分的组件，重建分组

    Args:
        manifest: 原始 manifest
        repairs: 修复记录列表
        warnings: 警告记录列表

    Returns:
        更新后的 manifest
    """
    # ── 建立被拆分组件的 ID 集合 ──
    split_ids = set()
    new_components = []
    for repair in repairs:
        original_id = repair.get("original_id", "")
        split_ids.add(original_id)
        for sub in repair.get("sub_components", []):
            if sub is not None:
                new_components.append(sub)

    # ── 替换组件列表 ──
    updated_components = []
    for comp in manifest.get("components", []):
        if comp.get("id") in split_ids:
            # 跳过被拆分的原组件
            continue
        updated_components.append(comp)

    # 插入新子组件（在原组件位置）
    for repair in repairs:
        original_id = repair.get("original_id", "")
        # 找到原组件在列表中的位置
        insert_idx = None
        for i, comp in enumerate(updated_components):
            # 在原组件后面插入
            pass
        # 简单追加到末尾
        for sub in repair.get("sub_components", []):
            if sub is not None:
                updated_components.append(sub)

    # ── 重建 reusable_groups ──
    from phase2_module_recognition import build_reusable_groups
    component_dir = str(
        PHASE2_OUTPUT_DIR / manifest.get("_source_module_dir", "")
        if manifest.get("_source_module_dir")
        else Path(manifest.get("source", "")).parent
    )
    # 确定裁剪图目录
    source_path = manifest.get("source", "")
    # 从 components 中找一个 cropped_file 推断目录
    for comp in updated_components:
        cf = comp.get("cropped_file", "")
        if cf:
            # cropped_file 是文件名，目录就是 manifest 所在目录
            break

    fw = manifest.get("facade_width", 1024)
    fh = manifest.get("facade_height", 1024)

    # 直接用 Phase 2 的 build_reusable_groups
    try:
        reusable_groups = build_reusable_groups(
            updated_components, component_dir, fw, fh
        )
    except Exception as e:
        print(f"  ⚠️  重建 reusable_groups 失败: {e}")
        # 兜底：手动按 reusable_group 分组
        from collections import defaultdict
        groups = defaultdict(list)
        for comp in updated_components:
            rg = comp.get("reusable_group", comp.get("type", "unknown"))
            groups[rg].append(comp)
        reusable_groups = []
        for rg, instances in groups.items():
            reusable_groups.append({
                "type": rg,
                "base_type": rg.split("_")[0],
                "count": len(instances),
                "all_instances": [
                    {"id": i.get("id"), "name": i.get("name"), "file": i.get("cropped_file")}
                    for i in instances
                ],
            })

    # ── 更新 floor_zones 中的组件归属 ──
    # 遍历 floor_zones，替换被拆分组件的引用
    updated_zones = []
    for zone in manifest.get("floor_zones", []):
        zone_copy = copy.deepcopy(zone)
        zone_components = zone_copy.get("components", [])
        new_zone_comps = []
        for zc in zone_components:
            zc_id = zc.get("id", "")
            if zc_id in split_ids:
                # 替换为子组件
                for sub in new_components:
                    if sub and sub.get("_phase26_split_from") == zc_id:
                        new_zone_comps.append(sub)
            else:
                new_zone_comps.append(zc)
        zone_copy["components"] = new_zone_comps
        zone_copy["component_count"] = len(new_zone_comps)
        # 更新 component_types
        types = set(zc.get("type", "?") for zc in new_zone_comps)
        zone_copy["component_types"] = ", ".join(sorted(types))
        updated_zones.append(zone_copy)

    # ── 构建更新后的 manifest ──
    updated_manifest = copy.deepcopy(manifest)
    updated_manifest["components"] = updated_components
    updated_manifest["reusable_groups"] = reusable_groups
    updated_manifest["floor_zones"] = updated_zones
    updated_manifest["total_components"] = len(updated_components)
    updated_manifest["total_reusable_types"] = len(reusable_groups)

    # ── 记录修复和警告 ──
    updated_manifest["phase26_repairs"] = repairs
    updated_manifest["phase26_warnings"] = warnings
    updated_manifest["phase26_repair_count"] = len([r for r in repairs if r.get("status") == "repaired"])
    updated_manifest["phase26_warning_count"] = len(warnings)

    return updated_manifest


# ════════════════════════════════════════════
# Step 5: 主流程
# ════════════════════════════════════════════

def find_latest_report(phase25_dir: Path) -> Optional[Path]:
    """找到最新的质检报告"""
    if not phase25_dir.exists():
        return None

    reports = sorted(phase25_dir.glob("quality_report_*.json"), reverse=True)
    return reports[0] if reports else None


def process_repair(
    phase2_dir: Path,
    phase25_report_path: Optional[Path] = None,
) -> Optional[dict]:
    """
    主流程：读取质检报告，修复 merged 组件

    Args:
        phase2_dir: Phase 2 输出目录
        phase25_report_path: 质检报告路径（None 则自动查找最新）

    Returns:
        修复结果 dict，失败返回 None
    """
    print(f"\n{'='*60}")
    print(f"🔧 Phase 2.6: 校准+代码修复")
    print(f"{'='*60}")
    print(f"  Phase 2 目录: {phase2_dir.name}")

    # ── 查找质检报告 ──
    if phase25_report_path is None:
        module_name = phase2_dir.name
        # 尝试在 phase25 目录下找对应子目录
        phase25_module_dir = PHASE25_OUTPUT_DIR / module_name
        if phase25_module_dir.exists():
            phase25_report_path = find_latest_report(phase25_module_dir)
        if phase25_report_path is None:
            # 尝试直接在 phase25 根目录找
            phase25_report_path = find_latest_report(PHASE25_OUTPUT_DIR)

    if phase25_report_path is None or not phase25_report_path.exists():
        print(f"  ❌ 找不到质检报告")
        return None

    print(f"  质检报告: {phase25_report_path.name}")

    # ── 加载输入 ──
    manifest, report = load_inputs(phase2_dir, phase25_report_path)
    print(f"  组件数: {manifest.get('total_components', 0)}")
    print(f"  可复用组: {manifest.get('total_reusable_types', 0)}")

    # ── 统计问题 ──
    merged = report.get("merged_components", [])
    missing = report.get("missing_components", [])
    grouping = report.get("grouping_issues", [])
    type_errors = report.get("type_errors", [])

    print(f"\n  质检问题统计:")
    print(f"    merged_components:  {len(merged)} 处（将尝试修复）")
    print(f"    missing_components: {len(missing)} 个（仅标记警告）")
    print(f"    grouping_issues:    {len(grouping)} 个（仅标记警告）")
    print(f"    type_errors:        {len(type_errors)} 个（仅标记警告）")

    if not merged:
        print(f"\n  ✅ 无 merged 问题，无需修复")
        # 仍然记录 warnings
        warnings = []
        for m in missing:
            warnings.append({"type": "missing", "issue": m})
        for g in grouping:
            warnings.append({"type": "grouping", "issue": g})
        for t in type_errors:
            warnings.append({"type": "type_error", "issue": t})

        manifest["phase26_repairs"] = []
        manifest["phase26_warnings"] = warnings
        manifest["phase26_repair_count"] = 0
        manifest["phase26_warning_count"] = len(warnings)

        # 回写（只添加 phase26 字段，不改动组件）
        manifest_path = phase2_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"  💾 已更新 manifest（仅添加警告标记）")
        return manifest

    # ── 获取正交原图路径 ──
    ortho_path = manifest.get("source", "")
    if not ortho_path or not Path(ortho_path).exists():
        print(f"  ❌ 正交原图不存在: {ortho_path}")
        return None

    print(f"  正交原图: {Path(ortho_path).name}")

    # ── 逐个处理 merged 组件 ──
    repairs = []
    warnings = []

    for issue in merged:
        comp_id = issue.get("component_id", "")
        desc = issue.get("description", "")
        print(f"\n  ── 处理 {comp_id}: {desc[:50]}...")

        # 匹配规则
        rule_result = match_merge_rule(issue, manifest)

        if rule_result is None:
            print(f"    ⏭️  规则不匹配，跳过")
            repairs.append({
                "original_id": comp_id,
                "status": "skipped",
                "reason": "rule_not_matched",
                "description": desc,
                "sub_components": [],
            })
            warnings.append({
                "type": "merged_unfixable",
                "component_id": comp_id,
                "description": desc,
                "reason": "规则表中无匹配的拆分规则",
            })
            continue

        upper_type, lower_type, ratio = rule_result
        print(f"    ✅ 匹配规则: ({upper_type}, {lower_type}) ratio={ratio}")

        # 找到原始组件
        original_comp = None
        for comp in manifest.get("components", []):
            if comp.get("id") == comp_id:
                original_comp = comp
                break

        if original_comp is None:
            print(f"    ❌ 组件不存在于 manifest")
            repairs.append({
                "original_id": comp_id,
                "status": "skipped",
                "reason": "component_not_found",
                "sub_components": [],
            })
            continue

        # 拆分
        # 生成编号前缀
        num_part = comp_id.replace("comp_", "")
        upper_comp, lower_comp = split_component(
            ortho_image_path=ortho_path,
            component=original_comp,
            upper_type=upper_type,
            lower_type=lower_type,
            ratio=ratio,
            output_dir=phase2_dir,
            new_id_prefix=num_part,
        )

        status = "repaired" if (upper_comp or lower_comp) else "failed"
        repairs.append({
            "original_id": comp_id,
            "original_type": original_comp.get("type", ""),
            "status": status,
            "split_rule": {
                "upper_type": upper_type,
                "lower_type": lower_type,
                "ratio": ratio,
            },
            "sub_components": [upper_comp, lower_comp],
        })

        if upper_comp:
            print(f"    ✂️  上方: {upper_comp['id']} ({upper_comp['type']})")
        if lower_comp:
            print(f"    ✂️  下方: {lower_comp['id']} ({lower_comp['type']})")

    # ── 记录其他类型警告 ──
    for m in missing:
        warnings.append({
            "type": "missing",
            "severity": m.get("severity", "medium"),
            "description": m.get("description", ""),
            "position": m.get("approximate_position", ""),
        })

    for g in grouping:
        warnings.append({
            "type": "grouping",
            "severity": g.get("severity", "medium"),
            "group": g.get("reusable_group", ""),
            "issue": g.get("issue", ""),
        })

    for t in type_errors:
        warnings.append({
            "type": "type_error",
            "severity": t.get("severity", "medium"),
            "component_id": t.get("component_id", ""),
            "wrong_type": t.get("wrong_type", ""),
            "correct_type": t.get("correct_type", ""),
            "reason": t.get("reason", ""),
        })

    # ── 更新 manifest ──
    print(f"\n  ── 更新 manifest ──")
    # 设置组件目录路径供 build_reusable_groups 使用
    manifest["_source_module_dir"] = str(phase2_dir)
    updated_manifest = update_manifest(manifest, repairs, warnings)

    # ── 保存 ──
    manifest_path = phase2_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(updated_manifest, f, ensure_ascii=False, indent=2)

    # 保存修复日志
    repair_log_path = phase2_dir / "phase26_repair_log.json"
    log = {
        "phase2_dir": str(phase2_dir),
        "phase25_report": str(phase25_report_path),
        "total_merged_issues": len(merged),
        "repaired_count": len([r for r in repairs if r["status"] == "repaired"]),
        "skipped_count": len([r for r in repairs if r["status"] == "skipped"]),
        "warning_count": len(warnings),
        "repairs": [
            {
                "original_id": r["original_id"],
                "status": r["status"],
                "split_rule": r.get("split_rule"),
                "sub_ids": [s["id"] for s in r.get("sub_components", []) if s],
            }
            for r in repairs
        ],
        "warnings": warnings,
    }
    with open(repair_log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    # ── 打印摘要 ──
    repaired = len([r for r in repairs if r["status"] == "repaired"])
    skipped = len([r for r in repairs if r["status"] == "skipped"])

    print(f"\n{'='*60}")
    print(f"✅ Phase 2.6 修复完成")
    print(f"{'='*60}")
    print(f"  merged 修复: {repaired} 处")
    print(f"  merged 跳过: {skipped} 处（规则不匹配）")
    print(f"  警告:        {len(warnings)} 条")
    print(f"  新组件数:    {updated_manifest.get('total_components', 0)}")
    print(f"  新可复用组:  {updated_manifest.get('total_reusable_types', 0)}")
    print(f"  💾 manifest: {manifest_path}")
    print(f"  📋 修复日志: {repair_log_path}")

    return updated_manifest


# ════════════════════════════════════════════
# CLI 入口
# ════════════════════════════════════════════

def main():
    """命令行入口：列出 Phase 2 模块，选择修复"""
    PHASE2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PHASE25_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 列出 Phase 2 模块 ──
    module_dirs = sorted(
        [d for d in PHASE2_OUTPUT_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name
    )

    if not module_dirs:
        print(f"⚠️  Phase 2 输出目录为空: {PHASE2_OUTPUT_DIR}")
        print(f"   请先运行 phase2_module_recognition.py")
        sys.exit(0)

    print(f"📁 找到 {len(module_dirs)} 个 Phase 2 模块：\n")
    for i, d in enumerate(module_dirs, 1):
        has_manifest = "✅" if (d / "manifest.json").exists() else "❌"
        print(f"  [{i}] {d.name}  {has_manifest}")
    print(f"  [0] 全部修复")

    # ── 收集所有可用的质检报告 ──
    available_reports = []  # [(phase2_dir, report_path, report_name)]
    for d in module_dirs:
        module_name = d.name
        phase25_dir = PHASE25_OUTPUT_DIR / module_name
        if phase25_dir.exists():
            reports = sorted(phase25_dir.glob("quality_report_*.json"))
            for rp in reports:
                available_reports.append((d, rp, f"{module_name} / {rp.name}"))

    if not available_reports:
        print(f"\n⚠️  没有找到任何质检报告，请先运行 phase25_quality_check.py")
        sys.exit(0)

    # ── 用户选择 Phase 2.5 质检报告 ──
    print(f"\n📋 可用的质检报告：\n")
    for i, (_, _, name) in enumerate(available_reports, 1):
        print(f"  [{i}] {name}")
    print(f"  [0] 全部修复（每个模块用最新报告）")

    try:
        report_choice = input(f"\n请选择质检报告编号 (默认=全部): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消")
        sys.exit(0)

    if not report_choice or report_choice == "0":
        # 全部修复：每个模块用最新报告
        selected_dirs = module_dirs
        selected_report = None
    else:
        try:
            idx = int(report_choice)
            if idx < 1 or idx > len(available_reports):
                print(f"❌ 无效编号: {idx}")
                sys.exit(1)
            sel_phase2_dir, sel_report, _ = available_reports[idx - 1]
            selected_dirs = [sel_phase2_dir]
            selected_report = sel_report
        except ValueError:
            print(f"❌ 无效输入: {report_choice}")
            sys.exit(1)

    # ── 执行修复 ──
    results = []
    for module_dir in selected_dirs:
        if selected_report:
            result = process_repair(module_dir, selected_report)
        else:
            result = process_repair(module_dir)
        if result:
            results.append(result)

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"🎉 Phase 2.6 全部完成")
    print(f"{'='*60}")
    print(f"  处理模块数: {len(results)}")
    for r in results:
        repairs = r.get("phase26_repairs", [])
        warnings = r.get("phase26_warnings", [])
        repaired = len([x for x in repairs if x.get("status") == "repaired"])
        print(f"    组件数={r.get('total_components', '?')}  "
              f"修复={repaired}  警告={len(warnings)}")


if __name__ == "__main__":
    main()
