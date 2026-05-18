# Phase 2 模组识别 - 自检方案

> 目标：确保 Phase 2 生成的可复用部件**数量足够、描述正确、结构完整**，为 Phase 3 四宫格生成提供可靠的数据源。

---

## 一、自检概览

自检分为 **7 个维度**，每个维度包含具体检查项和判定标准：

| 维度 | 检查项数 | 严重级别 |
|------|----------|----------|
| 1. 组件数量完整性 | 5 | P0 |
| 2. reusable_group 一致性 | 4 | P0 |
| 3. 描述准确性 | 5 | P1 |
| 4. bbox 坐标合法性 | 4 | P0 |
| 5. 裁剪质量 | 4 | P1 |
| 6. 楼层区域覆盖 | 3 | P1 |
| 7. 下游可用性 | 3 | P0 |

---

## 二、详细检查项

### 1. 组件数量完整性 [P0]

> 确保 VLM 识别出的组件数量满足工作流最低要求。

| 检查项 | 判定标准 | 说明 |
|--------|----------|------|
| 1.1 最小组件数 | `total_components >= 10` | 低于 10 个组件通常意味着漏检严重 |
| 1.2 必须存在的类型 | 必须包含 `roof` 或 `roof_slope`，且必须包含 `facade` | 缺少屋顶或墙面说明识别失败 |
| 1.3 门的数量 | `door` 类型 >= 1 | 每栋建筑至少应有一扇门 |
| 1.4 窗户数量上限 | `window` 类型 <= 20 | 超过 20 个窗户可能是重复识别 |
| 1.5 可复用类型数 | `total_reusable_types >= 5` | 低于 5 种可复用类型说明分组过粗 |

**自检代码逻辑**：
```python
def check_component_count(manifest):
    errors = []
    
    total = manifest.get("total_components", 0)
    if total < 10:
        errors.append(f"[P0] 组件总数过少: {total} < 10")
    
    types = [c["type"] for c in manifest["components"]]
    if "roof" not in types and "roof_slope" not in types:
        errors.append("[P0] 缺少屋顶组件 (roof/roof_slope)")
    if "facade" not in types:
        errors.append("[P0] 缺少墙面组件 (facade)")
    if "door" not in types:
        errors.append("[P0] 缺少门组件 (door)")
    
    window_count = types.count("window")
    if window_count > 20:
        errors.append(f"[P1] 窗户数量偏多: {window_count} > 20")
    
    reusable = manifest.get("total_reusable_types", 0)
    if reusable < 5:
        errors.append(f"[P0] 可复用类型过少: {reusable} < 5")
    
    return errors
```

---

### 2. reusable_group 一致性 [P0]

> 确保相同视觉外观的组件被正确归入同一可复用组，不同外观不会被错误合并。

| 检查项 | 判定标准 | 说明 |
|--------|----------|------|
| 2.1 组内实例数校验 | 每个 reusable_group 的 `count` == 该组 `all_instances` 的实际数量 | 防止计数错误 |
| 2.2 组内描述一致性 | 同一组内所有实例的 `type` 和 `subtype` 必须相同 | 不同类型不能混入同组 |
| 2.3 组间命名不重复 | 所有 reusable_group 的 `type` 字段唯一 | 不能有两个同名组 |
| 2.4 位置词检查 | reusable_group 的 `type` 字段不包含位置词 | `left/right/top/bottom/center/upper/lower` |

**自检代码逻辑**：
```python
def check_reusable_group_consistency(manifest):
    errors = []
    groups = manifest.get("reusable_groups", [])
    components = manifest.get("components", [])
    
    # 2.1 组内实例数校验
    for group in groups:
        expected = group.get("count", 0)
        actual = len(group.get("all_instances", []))
        if expected != actual:
            errors.append(
                f"[P0] 组 {group['type']} 实例数不一致: "
                f"count={expected}, 实际={actual}"
            )
    
    # 2.2 组内类型一致性
    for group in groups:
        instance_types = set()
        for inst in group.get("all_instances", []):
            instance_types.add(inst.get("type", ""))
        if len(instance_types) > 1:
            errors.append(
                f"[P0] 组 {group['type']} 包含多种类型: {instance_types}"
            )
    
    # 2.3 组间命名不重复
    group_names = [g["type"] for g in groups]
    duplicates = [n for n in group_names if group_names.count(n) > 1]
    if duplicates:
        errors.append(f"[P0] 存在重复的可复用组名: {set(duplicates)}")
    
    # 2.4 位置词检查
    position_words = {"left", "right", "top", "bottom", "center", 
                      "upper", "lower", "1f", "2f", "3f"}
    for group in groups:
        words = set(group["type"].split("_"))
        found = words & position_words
        if found:
            errors.append(
                f"[P0] 组 {group['type']} 包含位置词: {found}"
            )
    
    return errors
```

---

### 3. 描述准确性 [P1]

> 确保每个组件的中文描述和生成提示词准确、完整、可用于下游图像生成。

| 检查项 | 判定标准 | 说明 |
|--------|----------|------|
| 3.1 chinese_description 非空 | 每个组件的 `chinese_description` 长度 >= 10 字符 | 空描述无法生成图像 |
| 3.2 generation_prompt 非空 | 每个组件的 `generation_prompt` 长度 >= 15 字符 | 空提示词无法生成图像 |
| 3.3 描述包含视觉特征 | 描述中应包含**颜色**或**材质**关键词 | 如 "绿色"、"木质"、"石质" 等 |
| 3.4 描述不含位置词 | `chinese_description` 中不应出现 "左侧"、"右侧" 等位置词 | 位置信息属于实例名，不属于资产描述 |
| 3.5 generation_prompt 包含 "45度" | 提示词应包含 "45度" 或 "等轴" 关键词 | 确保生成的是游戏资产风格 |

**自检代码逻辑**：
```python
def check_description_accuracy(manifest):
    errors = []
    components = manifest.get("components", [])
    
    color_keywords = ["色", "红", "蓝", "绿", "黄", "白", "黑", "灰", 
                      "棕", "粉", "紫", "金", "银", "木", "石", "铁",
                      "铜", "砖", "瓦", "陶", "泥", "玻", "布"]
    position_words = ["左侧", "右侧", "上方", "下方", "左边", "右边",
                      "上边", "下边", "左面", "右面"]
    
    for comp in components:
        cid = comp.get("id", "?")
        desc = comp.get("chinese_description", "")
        prompt = comp.get("generation_prompt", "")
        
        # 3.1 描述非空
        if len(desc) < 10:
            errors.append(f"[P1] {cid}: chinese_description 过短 ({len(desc)}字符)")
        
        # 3.2 提示词非空
        if len(prompt) < 15:
            errors.append(f"[P1] {cid}: generation_prompt 过短 ({len(prompt)}字符)")
        
        # 3.3 描述包含视觉特征
        has_visual = any(kw in desc for kw in color_keywords)
        if not has_visual:
            errors.append(f"[P1] {cid}: chinese_description 缺少颜色/材质描述")
        
        # 3.4 描述不含位置词
        for pw in position_words:
            if pw in desc:
                errors.append(f"[P1] {cid}: chinese_description 包含位置词 '{pw}'")
                break
        
        # 3.5 提示词包含游戏资产关键词
        game_keywords = ["45度", "等轴", "游戏", "资产"]
        has_game = any(kw in prompt for kw in game_keywords)
        if not has_game:
            errors.append(f"[P1] {cid}: generation_prompt 缺少游戏资产关键词")
    
    return errors
```

---

### 4. bbox 坐标合法性 [P0]

> 确保边界框坐标有效、合理，裁剪结果正确。

| 检查项 | 判定标准 | 说明 |
|--------|----------|------|
| 4.1 坐标范围 | 所有坐标在 `[0, image_width]` × `[0, image_height]` 内 | 不能超出图片边界 |
| 4.2 坐标顺序 | `x1 < x2` 且 `y1 < y2` | 左上角必须在右下角左上方 |
| 4.3 最小尺寸 | `width >= 20px` 且 `height >= 20px` | 太小的组件无法生成有效图像 |
| 4.4 重叠率 | 任意两个组件的 IoU < 30% | 高重叠说明识别有误 |

**自检代码逻辑**：
```python
def check_bbox_validity(manifest):
    errors = []
    components = manifest.get("components", [])
    w = manifest.get("facade_width", 1024)
    h = manifest.get("facade_height", 1024)
    
    bboxes = []
    for comp in components:
        cid = comp.get("id", "?")
        bbox = comp.get("bbox", [])
        
        if len(bbox) != 4:
            errors.append(f"[P0] {cid}: bbox 格式错误，应为4个值")
            continue
        
        x1, y1, x2, y2 = bbox
        
        # 4.1 坐标范围
        if not (0 <= x1 <= w and 0 <= x2 <= w and 
                0 <= y1 <= h and 0 <= y2 <= h):
            errors.append(f"[P0] {cid}: bbox 坐标超出图片范围 {bbox}")
        
        # 4.2 坐标顺序
        if x1 >= x2 or y1 >= y2:
            errors.append(f"[P0] {cid}: bbox 坐标顺序错误 x1={x1}>=x2={x2} 或 y1={y1}>=y2={y2}")
        
        # 4.3 最小尺寸
        bw, bh = x2 - x1, y2 - y1
        if bw < 20 or bh < 20:
            errors.append(f"[P0] {cid}: bbox 尺寸过小 {bw}x{bh} < 20x20")
        
        bboxes.append((cid, bbox))
    
    # 4.4 重叠率检查（简化版，完整版需计算 IoU）
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            cid1, bb1 = bboxes[i]
            cid2, bb2 = bboxes[j]
            iou = calculate_iou(bb1, bb2)
            if iou > 0.3:
                errors.append(
                    f"[P1] {cid1} 和 {cid2} 重叠率过高: {iou:.1%}"
                )
    
    return errors

def calculate_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0
```

---

### 5. 裁剪质量 [P1]

> 确保组件图片被正确裁剪并保存。

| 检查项 | 判定标准 | 说明 |
|--------|----------|------|
| 5.1 裁剪文件存在 | 每个组件的 `cropped_file` 指向的文件必须存在 | 文件丢失会导致 Phase 3 失败 |
| 5.2 裁剪尺寸合理 | 裁剪图宽高在 `[20, 1024]` 像素范围内 | 太小无法生成，太大说明裁剪有误 |
| 5.3 裁剪成功率 | 裁剪成功数 / 总组件数 >= 90% | 低于 90% 说明系统性问题 |
| 5.4 代表实例有效 | 每个 reusable_group 的 `representative_file` 必须存在 | 代表实例用于 Phase 3 生成 |

**自检代码逻辑**：
```python
def check_crop_quality(manifest, phase2_dir):
    errors = []
    components = manifest.get("components", [])
    groups = manifest.get("reusable_groups", [])
    
    crop_success = 0
    crop_total = len(components)
    
    for comp in components:
        cid = comp.get("id", "?")
        cropped_file = comp.get("cropped_file")
        
        if not cropped_file:
            errors.append(f"[P1] {cid}: 缺少 cropped_file")
            continue
        
        cropped_path = os.path.join(phase2_dir, cropped_file)
        if not os.path.exists(cropped_path):
            errors.append(f"[P0] {cid}: 裁剪文件不存在 {cropped_file}")
            continue
        
        # 检查图片尺寸
        try:
            from PIL import Image
            with Image.open(cropped_path) as img:
                iw, ih = img.size
                if iw < 20 or ih < 20:
                    errors.append(f"[P1] {cid}: 裁剪图尺寸过小 {iw}x{ih}")
                elif iw > 1024 or ih > 1024:
                    errors.append(f"[P1] {cid}: 裁剪图尺寸异常 {iw}x{ih}")
                else:
                    crop_success += 1
        except Exception as e:
            errors.append(f"[P0] {cid}: 无法打开裁剪图 {e}")
    
    # 5.3 裁剪成功率
    success_rate = crop_success / crop_total if crop_total > 0 else 0
    if success_rate < 0.9:
        errors.append(
            f"[P0] 裁剪成功率过低: {crop_success}/{crop_total} = {success_rate:.1%}"
        )
    
    # 5.4 代表实例有效
    for group in groups:
        rep_file = group.get("representative_file")
        if not rep_file:
            errors.append(f"[P0] 组 {group['type']} 缺少代表实例文件")
            continue
        rep_path = os.path.join(phase2_dir, rep_file)
        if not os.path.exists(rep_path):
            errors.append(f"[P0] 组 {group['type']} 代表实例文件不存在: {rep_file}")
    
    return errors
```

---

### 6. 楼层区域覆盖 [P1]

> 确保组件被正确分配到楼层区域，且每个区域有足够组件。

| 检查项 | 判定标准 | 说明 |
|--------|----------|------|
| 6.1 区域数量 | `total_floor_zones` 在 `[2, 5]` 范围内 | 太少说明分区失败，太多说明过细 |
| 6.2 区域覆盖完整 | 所有区域的 y 范围之和应覆盖 `[0, 1]` | 不能有遗漏区域 |
| 6.3 区域组件数 | 每个区域至少包含 2 个组件 | 空区域说明分配有误 |

**自检代码逻辑**：
```python
def check_floor_zone_coverage(manifest):
    errors = []
    zones = manifest.get("floor_zones", [])
    total_zones = manifest.get("total_floor_zones", 0)
    
    # 6.1 区域数量
    if total_zones < 2:
        errors.append(f"[P1] 楼层区域过少: {total_zones} < 2")
    elif total_zones > 5:
        errors.append(f"[P1] 楼层区域过多: {total_zones} > 5")
    
    # 6.2 区域覆盖完整
    y_ranges = []
    for zone in zones:
        bbox = zone.get("bbox", [0, 0, 1, 1])
        if len(bbox) == 4:
            y_ranges.append((bbox[1], bbox[3]))  # (y_start, y_end)
    
    y_ranges.sort()
    if y_ranges:
        # 检查是否从 0 开始
        if y_ranges[0][0] > 0.01:
            errors.append(f"[P1] 楼层区域未从顶部开始: y_start={y_ranges[0][0]}")
        # 检查是否到 1 结束
        if y_ranges[-1][1] < 0.99:
            errors.append(f"[P1] 楼层区域未覆盖到底部: y_end={y_ranges[-1][1]}")
    
    # 6.3 区域组件数
    for zone in zones:
        zone_name = zone.get("name", "?")
        comp_count = zone.get("component_count", 0)
        if comp_count < 2:
            errors.append(f"[P1] 区域 '{zone_name}' 组件数过少: {comp_count} < 2")
    
    return errors
```

---

### 7. 下游可用性 [P0]

> 确保 Phase 2 的输出可以直接被 Phase 3 使用。

| 检查项 | 判定标准 | 说明 |
|--------|----------|------|
| 7.1 manifest 必须字段 | 必须包含 `source`, `components`, `reusable_groups`, `floor_zones` | 缺少任何字段会导致 Phase 3 失败 |
| 7.2 组件 ID 唯一 | 所有组件的 `id` 不重复 | 重复 ID 会导致映射混乱 |
| 7.3 组件 name 唯一 | 所有组件的 `name` 不重复 | 重复 name 会导致 Phase 3 去重失败 |

**自检代码逻辑**：
```python
def check_downstream_usability(manifest):
    errors = []
    
    # 7.1 必须字段
    required_fields = ["source", "components", "reusable_groups", "floor_zones"]
    for field in required_fields:
        if field not in manifest:
            errors.append(f"[P0] manifest 缺少必须字段: {field}")
    
    components = manifest.get("components", [])
    
    # 7.2 组件 ID 唯一
    ids = [c.get("id") for c in components if c.get("id")]
    duplicate_ids = [x for x in ids if ids.count(x) > 1]
    if duplicate_ids:
        errors.append(f"[P0] 存在重复的组件 ID: {set(duplicate_ids)}")
    
    # 7.3 组件 name 唯一
    names = [c.get("name") for c in components if c.get("name")]
    duplicate_names = [x for x in names if names.count(x) > 1]
    if duplicate_names:
        errors.append(f"[P0] 存在重复的组件 name: {set(duplicate_names)}")
    
    return errors
```

---

## 三、完整自检脚本

将以上所有检查整合为一个可执行脚本：

```python
#!/usr/bin/env python3
"""
Phase 2 模组识别 - 自检脚本
用法: python phase2_self_check.py <manifest.json路径> [phase2输出目录]
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter


def run_self_check(manifest_path: str, phase2_dir: str = None):
    """运行全部自检项，返回 (errors, warnings) 列表"""
    
    # 读取 manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    if phase2_dir is None:
        phase2_dir = os.path.dirname(manifest_path)
    
    errors = []    # P0 必须修复
    warnings = []  # P1 建议修复
    
    # ========== 1. 组件数量完整性 ==========
    errors.extend(check_component_count(manifest))
    
    # ========== 2. reusable_group 一致性 ==========
    errors.extend(check_reusable_group_consistency(manifest))
    
    # ========== 3. 描述准确性 ==========
    warnings.extend(check_description_accuracy(manifest))
    
    # ========== 4. bbox 坐标合法性 ==========
    errors.extend(check_bbox_validity(manifest))
    
    # ========== 5. 裁剪质量 ==========
    crop_errors = check_crop_quality(manifest, phase2_dir)
    # 区分 P0 和 P1
    for e in crop_errors:
        if "[P0]" in e:
            errors.append(e)
        else:
            warnings.append(e)
    
    # ========== 6. 楼层区域覆盖 ==========
    warnings.extend(check_floor_zone_coverage(manifest))
    
    # ========== 7. 下游可用性 ==========
    errors.extend(check_downstream_usability(manifest))
    
    return errors, warnings


def print_report(errors, warnings):
    """打印自检报告"""
    print("=" * 60)
    print("Phase 2 自检报告")
    print("=" * 60)
    
    if not errors and not warnings:
        print("✅ 全部检查通过！")
        return True
    
    if errors:
        print(f"\n❌ 发现 {len(errors)} 个必须修复的问题 (P0):")
        for i, e in enumerate(errors, 1):
            print(f"  {i}. {e}")
    
    if warnings:
        print(f"\n⚠️ 发现 {len(warnings)} 个建议修复的问题 (P1):")
        for i, w in enumerate(warnings, 1):
            print(f"  {i}. {w}")
    
    print("\n" + "=" * 60)
    if errors:
        print("❌ 自检未通过，请修复 P0 问题后重新运行 Phase 2")
        return False
    else:
        print("⚠️ 自检基本通过，建议修复 P1 问题以提高质量")
        return True


def main():
    if len(sys.argv) < 2:
        print("用法: python phase2_self_check.py <manifest.json路径> [phase2输出目录]")
        sys.exit(1)
    
    manifest_path = sys.argv[1]
    phase2_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(manifest_path):
        print(f"错误: 文件不存在 {manifest_path}")
        sys.exit(1)
    
    errors, warnings = run_self_check(manifest_path, phase2_dir)
    success = print_report(errors, warnings)
    sys.exit(0 if success else 1)


# ============================================================
# 以下是各检查函数的实现（与上面文档中的逻辑一致）
# ============================================================

def check_component_count(manifest):
    errors = []
    total = manifest.get("total_components", 0)
    if total < 10:
        errors.append(f"[P0] 组件总数过少: {total} < 10")
    
    types = [c["type"] for c in manifest["components"]]
    if "roof" not in types and "roof_slope" not in types:
        errors.append("[P0] 缺少屋顶组件 (roof/roof_slope)")
    if "facade" not in types:
        errors.append("[P0] 缺少墙面组件 (facade)")
    if "door" not in types:
        errors.append("[P0] 缺少门组件 (door)")
    
    window_count = types.count("window")
    if window_count > 20:
        errors.append(f"[P1] 窗户数量偏多: {window_count} > 20")
    
    reusable = manifest.get("total_reusable_types", 0)
    if reusable < 5:
        errors.append(f"[P0] 可复用类型过少: {reusable} < 5")
    
    return errors


def check_reusable_group_consistency(manifest):
    errors = []
    groups = manifest.get("reusable_groups", [])
    
    # 组内实例数校验
    for group in groups:
        expected = group.get("count", 0)
        actual = len(group.get("all_instances", []))
        if expected != actual:
            errors.append(
                f"[P0] 组 {group['type']} 实例数不一致: "
                f"count={expected}, 实际={actual}"
            )
    
    # 组内类型一致性
    for group in groups:
        instance_types = set()
        for inst in group.get("all_instances", []):
            instance_types.add(inst.get("type", ""))
        if len(instance_types) > 1:
            errors.append(
                f"[P0] 组 {group['type']} 包含多种类型: {instance_types}"
            )
    
    # 组间命名不重复
    group_names = [g["type"] for g in groups]
    seen = set()
    for name in group_names:
        if name in seen:
            errors.append(f"[P0] 存在重复的可复用组名: {name}")
        seen.add(name)
    
    # 位置词检查
    position_words = {"left", "right", "top", "bottom", "center", 
                      "upper", "lower", "1f", "2f", "3f"}
    for group in groups:
        words = set(group["type"].split("_"))
        found = words & position_words
        if found:
            errors.append(
                f"[P0] 组 {group['type']} 包含位置词: {found}"
            )
    
    return errors


def check_description_accuracy(manifest):
    errors = []
    components = manifest.get("components", [])
    
    color_keywords = ["色", "红", "蓝", "绿", "黄", "白", "黑", "灰", 
                      "棕", "粉", "紫", "金", "银", "木", "石", "铁",
                      "铜", "砖", "瓦", "陶", "泥", "玻", "布"]
    position_words = ["左侧", "右侧", "上方", "下方", "左边", "右边",
                      "上边", "下边", "左面", "右面"]
    
    for comp in components:
        cid = comp.get("id", "?")
        desc = comp.get("chinese_description", "")
        prompt = comp.get("generation_prompt", "")
        
        if len(desc) < 10:
            errors.append(f"[P1] {cid}: chinese_description 过短 ({len(desc)}字符)")
        
        if len(prompt) < 15:
            errors.append(f"[P1] {cid}: generation_prompt 过短 ({len(prompt)}字符)")
        
        has_visual = any(kw in desc for kw in color_keywords)
        if not has_visual:
            errors.append(f"[P1] {cid}: chinese_description 缺少颜色/材质描述")
        
        for pw in position_words:
            if pw in desc:
                errors.append(f"[P1] {cid}: chinese_description 包含位置词 '{pw}'")
                break
        
        game_keywords = ["45度", "等轴", "游戏", "资产"]
        has_game = any(kw in prompt for kw in game_keywords)
        if not has_game:
            errors.append(f"[P1] {cid}: generation_prompt 缺少游戏资产关键词")
    
    return errors


def check_bbox_validity(manifest):
    errors = []
    components = manifest.get("components", [])
    img_w = manifest.get("facade_width", 1024)
    img_h = manifest.get("facade_height", 1024)
    
    bboxes = []
    for comp in components:
        cid = comp.get("id", "?")
        bbox = comp.get("bbox", [])
        
        if len(bbox) != 4:
            errors.append(f"[P0] {cid}: bbox 格式错误，应为4个值")
            continue
        
        x1, y1, x2, y2 = bbox
        
        # 坐标范围（允许小范围超出）
        margin = 10
        if not (-margin <= x1 <= img_w + margin and 
                -margin <= x2 <= img_w + margin and
                -margin <= y1 <= img_h + margin and 
                -margin <= y2 <= img_h + margin):
            errors.append(f"[P0] {cid}: bbox 坐标超出图片范围 {bbox}")
        
        # 坐标顺序
        if x1 >= x2 or y1 >= y2:
            errors.append(f"[P0] {cid}: bbox 坐标顺序错误 x1={x1}>=x2={x2} 或 y1={y1}>=y2={y2}")
        
        # 最小尺寸
        bw, bh = x2 - x1, y2 - y1
        if bw < 20 or bh < 20:
            errors.append(f"[P0] {cid}: bbox 尺寸过小 {bw}x{bh} < 20x20")
        
        bboxes.append((cid, bbox))
    
    # 重叠率检查
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            cid1, bb1 = bboxes[i]
            cid2, bb2 = bboxes[j]
            iou = calculate_iou(bb1, bb2)
            if iou > 0.3:
                errors.append(f"[P1] {cid1} 和 {cid2} 重叠率过高: {iou:.1%}")
    
    return errors


def calculate_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def check_crop_quality(manifest, phase2_dir):
    errors = []
    components = manifest.get("components", [])
    groups = manifest.get("reusable_groups", [])
    
    crop_success = 0
    crop_total = len(components)
    
    for comp in components:
        cid = comp.get("id", "?")
        cropped_file = comp.get("cropped_file")
        
        if not cropped_file:
            errors.append(f"[P1] {cid}: 缺少 cropped_file")
            continue
        
        cropped_path = os.path.join(phase2_dir, cropped_file)
        if not os.path.exists(cropped_path):
            errors.append(f"[P0] {cid}: 裁剪文件不存在 {cropped_file}")
            continue
        
        try:
            from PIL import Image
            with Image.open(cropped_path) as img:
                iw, ih = img.size
                if iw < 20 or ih < 20:
                    errors.append(f"[P1] {cid}: 裁剪图尺寸过小 {iw}x{ih}")
                elif iw > 1024 or ih > 1024:
                    errors.append(f"[P1] {cid}: 裁剪图尺寸异常 {iw}x{ih}")
                else:
                    crop_success += 1
        except Exception as e:
            errors.append(f"[P0] {cid}: 无法打开裁剪图 {e}")
    
    success_rate = crop_success / crop_total if crop_total > 0 else 0
    if success_rate < 0.9:
        errors.append(
            f"[P0] 裁剪成功率过低: {crop_success}/{crop_total} = {success_rate:.1%}"
        )
    
    for group in groups:
        rep_file = group.get("representative_file")
        if not rep_file:
            errors.append(f"[P0] 组 {group['type']} 缺少代表实例文件")
            continue
        rep_path = os.path.join(phase2_dir, rep_file)
        if not os.path.exists(rep_path):
            errors.append(f"[P0] 组 {group['type']} 代表实例文件不存在: {rep_file}")
    
    return errors


def check_floor_zone_coverage(manifest):
    errors = []
    zones = manifest.get("floor_zones", [])
    total_zones = manifest.get("total_floor_zones", 0)
    
    if total_zones < 2:
        errors.append(f"[P1] 楼层区域过少: {total_zones} < 2")
    elif total_zones > 5:
        errors.append(f"[P1] 楼层区域过多: {total_zones} > 5")
    
    y_ranges = []
    for zone in zones:
        bbox = zone.get("bbox", [0, 0, 1, 1])
        if len(bbox) == 4:
            y_ranges.append((bbox[1], bbox[3]))
    
    y_ranges.sort()
    if y_ranges:
        if y_ranges[0][0] > 0.01:
            errors.append(f"[P1] 楼层区域未从顶部开始: y_start={y_ranges[0][0]}")
        if y_ranges[-1][1] < 0.99:
            errors.append(f"[P1] 楼层区域未覆盖到底部: y_end={y_ranges[-1][1]}")
    
    for zone in zones:
        zone_name = zone.get("name", "?")
        comp_count = zone.get("component_count", 0)
        if comp_count < 2:
            errors.append(f"[P1] 区域 '{zone_name}' 组件数过少: {comp_count} < 2")
    
    return errors


def check_downstream_usability(manifest):
    errors = []
    
    required_fields = ["source", "components", "reusable_groups", "floor_zones"]
    for field in required_fields:
        if field not in manifest:
            errors.append(f"[P0] manifest 缺少必须字段: {field}")
    
    components = manifest.get("components", [])
    
    ids = [c.get("id") for c in components if c.get("id")]
    seen_ids = set()
    for cid in ids:
        if cid in seen_ids:
            errors.append(f"[P0] 存在重复的组件 ID: {cid}")
        seen_ids.add(cid)
    
    names = [c.get("name") for c in components if c.get("name")]
    seen_names = set()
    for name in names:
        if name in seen_names:
            errors.append(f"[P0] 存在重复的组件 name: {name}")
        seen_names.add(name)
    
    return errors


if __name__ == "__main__":
    main()
```

---

## 四、自检结果判定

| 结果 | 条件 | 后续动作 |
|------|------|----------|
| ✅ 全部通过 | 0 errors, 0 warnings | 直接进入 Phase 3 |
| ⚠️ 基本通过 | 0 errors, N warnings | 建议人工复核 warnings，可继续 Phase 3 |
| ❌ 未通过 | N errors | **必须修复后重新运行 Phase 2** |

---

## 五、常见问题与修复建议

### Q1: 组件总数过少怎么办？
- 检查输入图片质量（分辨率、清晰度）
- 检查 VLM 提示词是否正确注入
- 尝试降低 confidence 阈值

### Q2: reusable_group 包含位置词怎么办？
- 这是 VLM 输出错误，需要在提示词中强化"不要包含位置词"的规则
- 可以在后处理中自动移除位置词

### Q3: 裁剪文件不存在怎么办？
- 检查 phase2 输出目录权限
- 检查 bbox 坐标是否合理
- 检查磁盘空间是否充足

### Q4: 描述缺少视觉特征怎么办？
- 这是 VLM 描述不够详细的问题
- 可以在提示词中要求"必须包含颜色和材质描述"

---

## 六、集成建议

### 方案 A: Phase 2 脚本内置自检
在 `phase2_module_recognition.py` 的 `main()` 函数末尾添加自检调用：

```python
# 在 manifest 保存后添加
errors, warnings = run_self_check(manifest_path, component_dir)
if errors:
    print(f"⚠️ 自检发现 {len(errors)} 个问题，请检查")
    for e in errors:
        print(f"  - {e}")
```

### 方案 B: 独立自检脚本
将上述代码保存为 `phase2_self_check.py`，在 Phase 2 完成后单独运行：

```bash
python phase2_self_check.py output/phase2/SurfCG/manifest.json
```

### 方案 C: CI/CD 集成
在自动化流水线中，将自检脚本的退出码作为质量门禁：
- exit 0 = 通过，继续下一阶段
- exit 1 = 失败，阻断流水线

---

## 七、扩展检查项（可选）

以下检查项可根据实际需求添加：

| 扩展项 | 说明 | 优先级 |
|--------|------|--------|
| VLM 置信度分布 | 统计低置信度组件比例 | P2 |
| 组件尺寸分布 | 检查是否有异常大/小的组件 | P2 |
| 描述相似度 | 检查不同组的描述是否过于相似 | P2 |
| 楼层组件类型 | 每层应包含哪些类型（屋顶层应有 roof） | P2 |
| 人工复核清单 | 生成需要人工确认的组件列表 | P2 |

---

*文档版本: v1.0*  
*适用于: phase2_module_recognition.py 输出的 manifest.json*
