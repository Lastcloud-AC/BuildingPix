"""
Phase 2.5 增强版: VLM 质检 + 正确组件列表生成
==============================================
输入：Phase 2 输出的 manifest.json + 正交原图
输出：质检报告（评分 + 问题清单）+ VLM 认为正确的组件列表

改进点：
  - VLM 先独立理解图片，生成自己认为正确的组件列表
  - 然后对比 Phase 2 的结果，找出问题
  - 输出包含：问题清单 + VLM 的正确组件列表

使用方式：
  python phase25_quality_check_enhanced.py

注意：
  - 质检 VLM 默认复用 config.py 的 VLM 配置
  - 可在 config.py 中单独配置 CHECKER_VLM_* 使用不同模型
"""

import os
import sys
import base64
import json
import requests
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional

# ─── 路径配置 ───
PHASE2_OUTPUT_DIR = Path(__file__).parent / "output" / "phase2"
OUTPUT_DIR = Path(__file__).parent / "output" / "phase25"


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


def sanitize_model_name(model: str) -> str:
    """将模型名转换为文件名安全字符串（去斜杠、冒号等）"""
    import re
    name = model.replace("/", "_").replace(":", "_").replace(" ", "")
    # 去掉连续下划线
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def get_next_report_path(output_subdir: Path, model: str) -> Path:
    """
    扫描输出目录，找到已有 quality_report_XX_*.json 的最大编号，
    返回下一个编号的路径。格式: quality_report_01_MODELNAME.json
    """
    output_subdir.mkdir(parents=True, exist_ok=True)

    max_num = 0
    safe_model = sanitize_model_name(model)

    for f in output_subdir.iterdir():
        if not f.suffix == ".json":
            continue
        name = f.stem  # e.g. quality_report_03_gemini-3-pro-preview
        if name.startswith("quality_report_"):
            # 提取编号部分
            parts = name.split("_")
            # quality_report_03_model... → parts = ["quality", "report", "03", "model..."]
            if len(parts) >= 3:
                try:
                    num = int(parts[2])
                    if num > max_num:
                        max_num = num
                except ValueError:
                    pass

    next_num = max_num + 1
    filename = f"quality_report_{next_num:02d}_{safe_model}.json"
    return output_subdir / filename


# ════════════════════════════════════════════
# 组件清单摘要构造
# ════════════════════════════════════════════

def build_component_summary(manifest: dict, max_desc_chars: int = 80) -> str:
    """从 manifest 构造给质检 VLM 看的组件清单文本（含bbox和像素尺寸）"""
    components = manifest.get("components", [])
    reusable_groups = manifest.get("reusable_groups", [])
    floor_zones = manifest.get("floor_zones", [])

    lines = []
    lines.append(f"总组件数：{len(components)}")
    lines.append(f"可复用组数：{len(reusable_groups)}")
    lines.append(f"楼层区域数：{len(floor_zones)}")

    # ── 图片尺寸信息 ──
    fw = manifest.get("facade_width", 0)
    fh = manifest.get("facade_height", 0)
    if fw and fh:
        lines.append(f"图片尺寸：{fw}×{fh} 像素")
    lines.append("")

    # ── 楼层区域摘要 ──
    if floor_zones:
        lines.append("── 楼层区域 ──")
        for z in floor_zones:
            zname = z.get("name", "?")
            zcount = z.get("component_count", 0)
            ztypes = z.get("component_types", "")
            lines.append(f"  {zname}: {zcount}个组件 ({ztypes})")
        lines.append("")

    # ── 组件逐项清单（含bbox和像素尺寸） ──
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

        # 计算 bbox 像素尺寸
        bbox_str = f"[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]" if len(bbox) == 4 else "?"
        if len(bbox) == 4:
            pw = bbox[2] - bbox[0]
            ph = bbox[3] - bbox[1]
            size_str = f"{pw}×{ph}px"
        else:
            size_str = "?"

        lines.append(f"  {cid} | {ctype} | {group} | {bbox_str} | {size_str} | {desc}")

    lines.append("")

    # ── 可复用组摘要 ──
    lines.append("── 可复用组摘要 ──")
    for g in reusable_groups:
        gtype = g.get("type", "?")
        base_type = g.get("base_type", "?")
        count = g.get("count", 0)
        desc = g.get("chinese_description", "")
        if len(desc) > max_desc_chars:
            desc = desc[:max_desc_chars] + "..."
        # 列出所有实例ID，方便VLM交叉验证
        instances = g.get("all_instances", [])
        instance_ids = [i.get("id", "?") for i in instances]
        id_list = ",".join(instance_ids)
        lines.append(f"  {gtype} (base={base_type}) | {count}实例 [{id_list}] | {desc}")

    return "\n".join(lines)


# ════════════════════════════════════════════
# 增强版 VLM 质检 Prompt（同时输出正确组件列表）
# ════════════════════════════════════════════

ENHANCED_QUALITY_CHECK_PROMPT = """你是建筑组件拆分质量审核专家。我会给你一张建筑立面图，以及一份由另一个AI生成的组件拆分清单。

你的任务有两个：
1. **独立理解图片**：先不看清单，自己分析图片中的所有建筑组件，生成你认为正确的组件列表
2. **对比找问题**：将你的理解和另一个AI的清单对比，找出问题

═══════════════════════════════════════════
【图片与建筑信息】
{building_context}
═══════════════════════════════════════════

【审核维度与问题归类】

1. 遗漏 → missing_components
   - 按楼层从上到下逐层检查：屋顶→上层→底层，确保每层组件都有覆盖
   - 对照图片中每个肉眼可见的独立构件，检查清单是否有对应条目
   - 特别注意：外观相同的多个实例（如一排窗户），清单应逐个列出
   - 数量验证：清点图片中窗户/门/阳台等主要组件的数量，与清单数量比对

2. 合并过度 → merged_components
   - 参照清单中每个组件的 bbox 和像素尺寸，判断 bbox 是否包含多种不同构件
   - 关键判断依据：如果某个组件的 bbox 像素面积明显大于同类组件（如一个"门"的bbox宽度是其他门的2倍以上），很可能存在合并
   - 具体检查项：
     a) 窗户+窗台/花台是否被合并？→ bbox高度异常
     b) 屋顶 bbox 是否包着下方墙面？→ bbox下界超出屋顶区域
     c) 门+门廊/雨棚是否被合并？→ bbox高度过大
     d) 遮阳篷+橱窗是否被合并？→ bbox包含不同功能区域
     e) facade 面板是否过大（宽度>350px或覆盖多个窗间区域）？

   【拆分策略说明 - 重要】
   以下拆分是正确的设计意图，不要报为"拆分过度"：
   - 遮阳篷(canopy) + 橱窗(shopfront) → 应拆分（商铺组件的功能子件）
   - 窗户(window) + 阳台(balcony) → 应拆分（功能子件，窗户在上，阳台在下）
   - 花箱(flower_box) + 窗台/窗户 → 应拆分（装饰子件）
   - 如果组件已被拆分为上述功能子件（如 comp_017_upper + comp_017_lower），这是正确的拆分，不要报为"拆分过度"
   - 只有当拆分导致功能丧失（如门+门廊必须一体、屋顶+屋脊必须一体）时，才报"拆分过度"
   - 注意：如果两个组件 ID 以 "_upper" 和 "_lower" 结尾，说明它们是被 Phase 2.6 正确拆分的子件，不要质疑这种拆分

3. 分组问题 → grouping_issues
   - 被归入同一个 reusable_group 的组件，视觉外观是否真的一致？
   - 检查方法：同一组内的各实例，比较它们的 bbox 尺寸范围。如果同一组内组件的像素尺寸差异超过30%，很可能不应该同组
   - 注意：Phase 2 已对 decoration 类型做了子类型分组（如 decoration_spire_iron, decoration_lamp_black_metal 等），请检查这些子分组是否正确
   - facade 面板：同组的面板应该是同一楼层、同材质的窗间墙

4. 类型标注错误 → type_errors
   - 对照图片验证每个组件的 type 标注是否准确
   - 常见混淆：
     a) facade vs base_wall：灰泥墙面用 facade，石质基座用 base_wall 或 facade_stone_base
     b) shopfront vs window：有商业橱窗特征（遮阳篷/展示窗）的用 shopfront
     c) balcony vs decoration：有功能性的突出平台用 balcony
     d) roof_slope vs decoration：实际屋顶面用 roof_slope，装饰性尖顶用 decoration

═══════════════════════════════════════════
【严重程度定义 - 必须准确区分】
high   = 该问题会导致下游Phase3生成失败（如整层缺失、核心组件合并无法生成）
medium = 影响质量但可放行（如外观相似但细节不同的组件被归同组、一个小型装饰遗漏）
low    = 优化建议（如同类窗户尺寸略有差异、细微标注偏差）

═══════════════════════════════════════════
【组件清单（另一个AI的识别结果）】

{component_summary}

═══════════════════════════════════════════
【输出 - 严格JSON，只输出这个对象，不要输出任何解释文字】

{{
  "missing_components": [
    {{
      "description": "图片右侧第2个窗户，圆形，绿色窗框",
      "approximate_position": "右侧中上部",
      "severity": "medium",
      "dimension": "completeness"
    }}
  ],
  "merged_components": [
    {{
      "component_id": "comp_005",
      "description": "这个组件同时包含了窗户和下方的花台，应拆为2个独立组件",
      "evidence": "bbox [200,400,300,600] 尺寸 100×200px，远大于同类窗户的 80×100px",
      "severity": "high",
      "dimension": "fineness"
    }}
  ],
  "grouping_issues": [
    {{
      "reusable_group": "decoration_lamp_black_metal",
      "issue": "该组中 comp_035 是圆形壁灯而 comp_037 是方形壁灯，外观不同不应同组",
      "severity": "medium",
      "dimension": "grouping"
    }}
  ],
  "type_errors": [
    {{
      "component_id": "comp_030",
      "wrong_type": "decoration",
      "correct_type": "window",
      "reason": "这是一个带装饰框的拱形窗户，不是装饰品",
      "dimension": "type_accuracy"
    }}
  ],
  "blocking": false,
  "summary": "整体基本可用，有1个组件合并过度，1个分组需调整",
  "my_component_list": [
    {{
      "id": "comp_001",
      "type": "roof",
      "subtype": "gable",
      "name": "roof_gable_main",
      "reusable_group": "roof_gable",
      "chinese_description": "红色木质双坡屋顶",
      "generation_prompt": "单个红色木质双坡屋顶，45度等轴游戏建筑资产",
      "material": "木质",
      "color": "红色",
      "bbox": [100, 50, 600, 200],
      "confidence": 0.95
    }},
    {{
      "id": "comp_002",
      "type": "wall",
      "subtype": "facade",
      "name": "wall_facade_main",
      "reusable_group": "wall_facade",
      "chinese_description": "米色灰泥外墙",
      "generation_prompt": "单个米色灰泥外墙，45度等轴游戏建筑资产",
      "material": "灰泥",
      "color": "米色",
      "bbox": [100, 200, 600, 800],
      "confidence": 0.9
    }}
  ]
}}

【重要规则】
1. 每个问题必须标注 dimension 字段：completeness / fineness / grouping / type_accuracy
2. 每个问题必须标注 severity 字段：high / medium / low
3. severity 必须精确区分——不要把所有问题都标为 medium
4. 如果某个维度没有任何问题，对应数组留空 []
5. blocking = true 仅当存在 high 严重度问题且会导致 Phase3 完全无法运行
6. summary 不超过50字，说明最关键的1-2个问题
7. **my_component_list - 完整组件列表（核心要求）**：
   - ⚠️ 你必须列出图片中**所有可见的建筑组件**，包括：
     * 大面积组件：墙面(wall)、屋顶(roof)、地面(base_wall)
     * 开口组件：窗户(window)、门(door)、阳台(balcony)
     * 装饰组件：遮阳篷(awning)、花箱(flower_box)、栏杆(railing)、装饰(decoration)
     * 功能组件：烟囱(chimney)、楼梯(staircase)、老虎窗(dormer)
   - 逐层扫描：屋顶→上层→底层，每层的每个组件都要列出
   - 数量要求：组件总数应该与另一个AI的清单相近（通常20-50个），不能只有几个
   - 每个组件必须包含：id, type, subtype, name, reusable_group, chinese_description, generation_prompt, material, color, bbox, confidence
   - bbox 必须是像素坐标 [x1, y1, x2, y2]
   - 不要参考另一个AI的清单，完全基于你对图片的理解
   - 如果你认为某个组件应该拆分或合并，按照你的理解来生成
   - **检查清单**：生成完成后，清点窗户数量、门数量、屋顶数量，确保与图片一致"""


# ════════════════════════════════════════════
# 建筑上下文构造
# ════════════════════════════════════════════

def build_building_context(manifest: dict) -> str:
    """从 manifest 和 Phase 1 数据构造建筑上下文信息"""
    lines = []

    # 图片尺寸
    fw = manifest.get("facade_width", 0)
    fh = manifest.get("facade_height", 0)
    if fw and fh:
        lines.append(f"图片尺寸：{fw}×{fh} 像素")

    # 识别模型
    model = manifest.get("recognition_model", "unknown")
    lines.append(f"识别模型：{model}")

    # 组件统计
    total = manifest.get("total_components", 0)
    groups = manifest.get("total_reusable_types", 0)
    zones = manifest.get("total_floor_zones", 0)
    lines.append(f"识别结果：{total}个组件，{groups}个可复用组，{zones}个楼层区域")

    # 楼层区域信息
    floor_zones = manifest.get("floor_zones", [])
    if floor_zones:
        lines.append("")
        lines.append("楼层区域划分：")
        for z in floor_zones:
            zname = z.get("name", "?")
            zcount = z.get("component_count", 0)
            ztypes = z.get("component_types", "")
            lines.append(f"  {zname}: {zcount}个组件 ({ztypes})")

    # 组件类型分布
    components = manifest.get("components", [])
    if components:
        type_counts = {}
        for c in components:
            t = c.get("type", "?")
            type_counts[t] = type_counts.get(t, 0) + 1
        dist = ", ".join(f"{t}×{n}" for t, n in sorted(type_counts.items()))
        lines.append(f"类型分布：{dist}")

    # 尝试加载 Phase 1 的分析结果
    phase1_dir = Path(__file__).parent / "output" / "phase1"
    phase1_analysis = None
    if phase1_dir.exists():
        for f in phase1_dir.iterdir():
            if f.suffix == ".json" and "analysis" in f.name:
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        phase1_analysis = json.load(fp)
                    break
                except Exception:
                    pass

    if phase1_analysis:
        lines.append("")
        lines.append("Phase 1 建筑语义：")
        # 提取关键信息
        building_info = phase1_analysis.get("building_analysis", phase1_analysis)
        if isinstance(building_info, dict):
            for key in ["building_style", "total_floors", "roof_type",
                        "ground_floor", "facade_material", "facade_color"]:
                val = building_info.get(key)
                if val:
                    cn_key = {
                        "building_style": "建筑风格",
                        "total_floors": "总楼层数",
                        "roof_type": "屋顶类型",
                        "ground_floor": "底层特征",
                        "facade_material": "外墙材质",
                        "facade_color": "主色调",
                    }.get(key, key)
                    lines.append(f"  {cn_key}: {val}")

    return "\n".join(lines)


# ════════════════════════════════════════════
# 质检 VLM 调用
# ════════════════════════════════════════════

def call_quality_check_vlm(ortho_b64: str, ortho_media_type: str,
                           component_summary: str, building_context: str = "",
                           max_retries: int = 3) -> dict:
    """调用 VLM 做质检，返回 JSON 结果（带重试机制）"""
    # 优先使用独立的质检 VLM 配置，否则复用 Phase 2 的 VLM 配置
    try:
        from config import (
            CHECKER_VLM_API_URL, CHECKER_VLM_API_KEY, CHECKER_VLM_MODEL
        )
        api_url = CHECKER_VLM_API_URL
        api_key = CHECKER_VLM_API_KEY
        model = CHECKER_VLM_MODEL
    except ImportError:
        from config import VLM_API_URL, VLM_API_KEY, VLM_MODEL
        api_url = VLM_API_URL
        api_key = VLM_API_KEY
        model = VLM_MODEL

    prompt = ENHANCED_QUALITY_CHECK_PROMPT.format(
        component_summary=component_summary,
        building_context=building_context if building_context else "（未提供额外建筑信息）"
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{ortho_media_type};base64,{ortho_b64}"},
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 16000,  # 增加 token 限制，确保能输出完整组件列表（通常需要12000-15000）
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"  🔍 模型: {model}")

    import time
    for attempt in range(1, max_retries + 1):
        print(f"  🔍 调用质检 VLM... (尝试 {attempt}/{max_retries})")
        try:
            resp = requests.post(api_url, json=payload, headers=headers, timeout=300)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            # 提取 JSON
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            # 尝试解析，失败则尝试修复
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # 尝试找到最后一个 }，截断并补全
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
                print(f"  ❌ 已达最大重试次数，VLM 请求持续超时")
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
# 评分计算（代码端，不依赖VLM主观打分）
# ════════════════════════════════════════════

# 每个严重程度的扣分规则
SEVERITY_DEDUCTIONS = {
    "high": 12,
    "medium": 5,
    "low": 2,
}

# 各维度权重
DIMENSION_WEIGHTS = {
    "completeness": 0.30,
    "fineness": 0.30,
    "grouping": 0.20,
    "type_accuracy": 0.20,
}

# 维度到问题列表的映射
DIMENSION_ISSUE_KEYS = {
    "completeness": "missing_components",
    "fineness": "merged_components",
    "grouping": "grouping_issues",
    "type_accuracy": "type_errors",
}


def calculate_scores(result: dict) -> dict:
    """
    根据VLM返回的问题清单，用代码计算评分。
    基准分100，按问题严重程度扣分，确保：问题不同→分数不同。
    """
    scores = {}
    score_details = {}

    for dim, issue_key in DIMENSION_ISSUE_KEYS.items():
        issues = result.get(issue_key, [])

        # 同时兼容旧格式（问题无dimension字段）和新格式（问题有dimension字段）
        dim_issues = [i for i in issues if i.get("dimension", dim) == dim]
        # 如果没有dimension字段的问题，也计入该维度（兼容旧报告）
        legacy_issues = [i for i in issues if "dimension" not in i]
        effective_issues = dim_issues if dim_issues else legacy_issues

        deduction = 0
        breakdown = []
        for issue in effective_issues:
            sev = issue.get("severity", "medium")
            deduct = SEVERITY_DEDUCTIONS.get(sev, 5)
            deduction += deduct
            breakdown.append(f"{sev}(-{deduct})")

        score = max(0, 100 - deduction)
        scores[dim] = score

        if breakdown:
            score_details[dim] = f"100-{'-'.join(breakdown)}={score}"
        else:
            score_details[dim] = "100（无问题）"

    # 加权计算综合分
    overall = sum(scores[dim] * DIMENSION_WEIGHTS[dim] for dim in scores)

    return {
        "scores": scores,
        "overall_score": round(overall, 1),
        "score_details": score_details,
    }


# ════════════════════════════════════════════
# 质检报告输出
# ════════════════════════════════════════════

def print_report(result: dict):
    """打印人类可读的质检报告"""
    scores = result.get("scores", {})
    overall = result.get("overall_score", 0)
    score_details = result.get("score_details", {})

    print(f"\n{'='*60}")
    print(f"📊 Phase 2.5 增强版质检报告")
    print(f"{'='*60}")

    # ── 评分表 ──
    print(f"\n  📈 评分明细（基准100分 - 按问题扣分）：")
    for dim, weight in DIMENSION_WEIGHTS.items():
        score = scores.get(dim, "?")
        detail = score_details.get(dim, "")
        weight_pct = int(weight * 100)
        print(f"    {dim:<18} {score}/100  (权重{weight_pct}%)  {detail}")
    print(f"    ─────────────────────")
    print(f"    综合:    {overall}/100")

    # ── 判定 ──
    blocking = result.get("blocking", False)
    if blocking:
        verdict = "🚫 BLOCKED — 存在阻塞性问题，需修复后重新质检"
    elif overall >= 80:
        verdict = "✅ PASS — 可直接进入 Phase 3"
    elif overall >= 60:
        verdict = "⚠️  CONDITIONAL PASS — 可进入 Phase 3，有需关注的问题"
    else:
        verdict = "❌ FAIL — 建议修复后重新质检"
    print(f"\n  🏷️  判定: {verdict}")

    # ── 问题详情 ──
    issues_found = False

    missing = result.get("missing_components", [])
    if missing:
        issues_found = True
        print(f"\n  🔴 遗漏组件 ({len(missing)}个)：")
        for m in missing:
            pos = f" ({m.get('approximate_position','')})" if m.get('approximate_position') else ""
            print(f"    [{m.get('severity','?')}] {m.get('description','')}{pos}")

    merged = result.get("merged_components", [])
    if merged:
        issues_found = True
        print(f"\n  🟠 合并过度 ({len(merged)}处)：")
        for m in merged:
            evidence = f" | 证据: {m.get('evidence','')}" if m.get('evidence') else ""
            print(f"    [{m.get('severity','?')}] {m.get('component_id','')} — {m.get('description','')}{evidence}")

    grouping = result.get("grouping_issues", [])
    if grouping:
        issues_found = True
        print(f"\n  🟡 分组问题 ({len(grouping)}个)：")
        for g in grouping:
            print(f"    [{g.get('severity','?')}] {g.get('reusable_group','')} — {g.get('issue','')}")

    type_errs = result.get("type_errors", [])
    if type_errs:
        issues_found = True
        print(f"\n  🟡 类型错误 ({len(type_errs)}个)：")
        for t in type_errs:
            print(f"    [{t.get('severity','?')}] {t.get('component_id','')}: {t.get('wrong_type','')} → {t.get('correct_type','')} ({t.get('reason','')})")

    if not issues_found:
        print(f"\n  ✅ 未发现问题")

    # ── VLM 的组件列表统计 ──
    my_components = result.get("my_component_list", [])
    if my_components:
        print(f"\n  📋 VLM 识别的组件列表: {len(my_components)} 个组件")
        type_counts = {}
        for c in my_components:
            t = c.get("type", "?")
            type_counts[t] = type_counts.get(t, 0) + 1
        dist = ", ".join(f"{t}×{n}" for t, n in sorted(type_counts.items()))
        print(f"    类型分布: {dist}")

    print(f"\n  📝 总结: {result.get('summary', '无')}")
    print(f"{'='*60}\n")


# ════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════

def check_module(module_dir: Path) -> Optional[dict]:
    """
    对单个 Phase 2 模块目录做增强版质检
    """
    manifest_path = module_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"  ❌ manifest.json 不存在: {manifest_path}")
        return None

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    module_name = module_dir.name
    print(f"\n{'='*60}")
    print(f"🔍 Phase 2.5 增强版: VLM 质检 + 组件列表生成 - {module_name}")
    print(f"{'='*60}")

    # ── 确定输出路径（自动编号） ──
    output_subdir = OUTPUT_DIR / module_name
    output_subdir.mkdir(parents=True, exist_ok=True)

    # 先确定质检模型名，用于文件名
    from config import VLM_MODEL
    try:
        from config import CHECKER_VLM_MODEL
        checker_model = CHECKER_VLM_MODEL
    except ImportError:
        checker_model = VLM_MODEL

    output_path = get_next_report_path(output_subdir, checker_model)
    # 修改文件名，标记为增强版
    output_path = output_path.parent / output_path.name.replace("quality_report", "enhanced_report")
    print(f"  📄 输出文件: {output_path.name}")

    # ── 找到正交原图 ──
    ortho_path = manifest.get("source", "")
    if not ortho_path or not Path(ortho_path).exists():
        print(f"  ❌ 正交原图不存在: {ortho_path}")
        return None

    print(f"  📷 正交原图: {Path(ortho_path).name}")
    print(f"  📦 组件数: {manifest.get('total_components', 0)}")
    print(f"  📦 可复用组数: {manifest.get('total_reusable_types', 0)}")

    # ── 构造建筑上下文 ──
    print(f"\n[1/5] 构造建筑上下文...")
    building_context = build_building_context(manifest)
    print(f"  📝 上下文长度: {len(building_context)} 字符")

    # ── 构造组件摘要 ──
    print(f"\n[2/5] 构造组件清单摘要...")
    component_summary = build_component_summary(manifest)
    print(f"  📝 摘要长度: {len(component_summary)} 字符")

    # ── 编码正交图 ──
    print(f"\n[3/5] 编码正交图...")
    ortho_b64 = encode_image_to_base64(ortho_path)
    ortho_media_type = get_image_media_type(ortho_path)
    print(f"  🖼️  base64 长度: {len(ortho_b64)} 字符")

    # ── 调用质检 VLM ──
    print(f"\n[4/5] 调用增强版质检 VLM...")
    try:
        result = call_quality_check_vlm(ortho_b64, ortho_media_type, component_summary, building_context)
    except Exception as e:
        print(f"  ❌ 质检 VLM 调用失败: {e}")
        return None

    # ── 用代码计算评分 ──
    print(f"\n[5/5] 计算评分（基准100分 - 按问题扣分）...")
    # 如果VLM返回了旧格式的scores，移除它（我们重新算）
    if "scores" in result:
        del result["scores"]
    if "overall_score" in result:
        del result["overall_score"]

    calc = calculate_scores(result)
    result["scores"] = calc["scores"]
    result["overall_score"] = calc["overall_score"]
    result["score_details"] = calc["score_details"]

    # ── 评分校验 ──
    all_issues = (
        result.get("missing_components", [])
        + result.get("merged_components", [])
        + result.get("grouping_issues", [])
        + result.get("type_errors", [])
    )
    no_severity = [i for i in all_issues if not i.get("severity")]
    if no_severity:
        print(f"  ⚠️  有 {len(no_severity)} 个问题缺少 severity 字段，默认按 medium 扣分")

    # ── 丰富结果，附加元信息 ──
    result["_meta"] = {
        "module": module_name,
        "ortho_source": ortho_path,
        "phase2_model": manifest.get("recognition_model", "unknown"),
        "checker_model": checker_model,
        "total_components": manifest.get("total_components", 0),
        "total_reusable_types": manifest.get("total_reusable_types", 0),
        "report_file": output_path.name,
        "enhanced_version": True,
    }

    # ── 保存 ──
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 增强版质检报告已保存: {output_path}")

    # ── 打印报告 ──
    print_report(result)

    return result


def list_phase2_modules() -> list:
    """列出所有 Phase 2 输出目录及其元信息"""
    if not PHASE2_OUTPUT_DIR.exists():
        return []

    module_dirs = sorted(
        [d for d in PHASE2_OUTPUT_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name
    )

    entries = []
    for d in module_dirs:
        mf = d / "manifest.json"
        if mf.exists():
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                model = data.get("recognition_model", "?")
                total = data.get("total_components", "?")
                groups = data.get("total_reusable_types", "?")
                zones = data.get("total_floor_zones", "?")
                info = f"模型={model}  组件={total}  可复用组={groups}  楼层区={zones}"
            except Exception:
                info = "(manifest读取失败)"
        else:
            info = "(无manifest)"
        entries.append((d, info))

    return entries


def main():
    """主入口：扫描 Phase 2 输出，让用户选择对哪个模块做增强版质检"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entries = list_phase2_modules()

    if not entries:
        print(f"⚠️  Phase 2 输出目录为空或不存在: {PHASE2_OUTPUT_DIR}")
        print(f"   请先运行 phase2_module_recognition.py")
        sys.exit(0)

    # ── 列出所有可选模块 ──
    print(f"📁 找到 {len(entries)} 个 Phase 2 模块：\n")
    for i, (d, info) in enumerate(entries, 1):
        print(f"  [{i}] {d.name}  {info}")
    print(f"  [0] 全部质检")

    # ── 用户选择 ──
    try:
        choice = input(f"\n请选择要质检的模块编号 (默认=全部): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消")
        sys.exit(0)

    if not choice or choice == "0":
        selected_dirs = [d for d, _ in entries]
    else:
        try:
            idx = int(choice)
            if idx < 1 or idx > len(entries):
                print(f"❌ 无效编号: {idx}")
                sys.exit(1)
            selected_dirs = [entries[idx - 1][0]]
        except ValueError:
            print(f"❌ 无效输入: {choice}")
            sys.exit(1)

    # ── 逐个质检 ──
    results = []
    for d in selected_dirs:
        result = check_module(d)
        if result:
            results.append(result)

    # ── 汇总 ──
    if results:
        print(f"\n{'='*60}")
        print(f"🎉 增强版质检完成，共 {len(results)} 个模块")
        print(f"{'='*60}")
        for r in results:
            meta = r.get("_meta", {})
            score = r.get("overall_score", "?")
            my_count = len(r.get("my_component_list", []))
            print(f"  {meta.get('module', '?')}: 综合分={score}, VLM识别组件数={my_count}")


if __name__ == "__main__":
    main()
