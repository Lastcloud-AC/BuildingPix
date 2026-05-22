"""
Phase 2: 模组识别
===========================
输入：phase1_standardize.py 输出的正交图（纯白背景建筑正面图）
输出：建筑组件清单 + 组件裁剪图 + 楼层区域图

流程：
  1. VLM 识别建筑组件
     - 读取 output/phase1/ 中的正交图
     - 调用 VLM 识别组件，返回 bbox + 类型 + reusable_group
     - 若已有识别结果且未指定 overwrite，直接复用

  2. 质量校验
     - 检查组件类型（必须有 roof 或 facade）
     - 检查门的位置（应在建筑底部）
     - 检查组件重叠率（IoU > 30% 告警）
     - 检查组件数量合理性

  3. 裁剪组件图
     - 根据 bbox 坐标裁剪出各组件 PNG
     - 自动检测坐标格式（归一化 vs 像素）
     - 太小的组件（<20px）跳过裁剪

  4. 构建可复用组件分组
     - 直接使用 VLM 返回的 reusable_group 分组
     - 所有组件都参与分组（包括裁剪失败的），确保 Phase3 去重映射完整
     - 选取代表实例（面积最大、置信度最高）

  5. 楼层区域归并 + 裁剪
     - 读取 Phase1 analysis.json 获取楼层数
     - 根据组件 y 中心位置聚类，找到自然分界线
     - 合并过薄的区域（高度 < 10% 图片高度）
     - 屋顶/地基：刚好裁剪（不扩展）
     - 中间层：上下各扩展 2%，避免裁断连接处

  6. 输出 manifest.json + 打印摘要

输出内容：
  - components: 所有识别到的组件实例裁剪图
  - reusable_groups: 按类型细分的可复用组件分组，含生成状态追踪
  - floor_zones: 按楼层归并的区域图（每个 zone 含 cropped_file）
  - warnings: 质量校验警告列表

使用方式：
  python phase2_module_recognition.py

注意：
  - Phase 2 输入 = Phase 1 Standarize 的输出（正交图 + analysis.json）
  - 输出到 output/phase2/，每次运行自动编号（如 SurfCG_01_qwen3.5-omni-plus/），不会覆盖
  - 提示词拼接由 Phase 3 完成，不在 Phase 2 预生成（减少 token）
"""

import os
import sys
import base64
import json
import requests
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple, Optional

# ─── 路径配置 ───
# 默认读取 phase1_standardize 的输出目录
PHASE1_OUTPUT_DIR = Path(__file__).parent / "output" / "phase1"
OUTPUT_DIR = Path(__file__).parent / "output" / "phase2"


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
    }.get(ext, "image/jpeg")


def normalize_bbox(bbox: List[float], image_width: int, image_height: int) -> List[float]:
    """
    将 bbox 转换为归一化坐标（0-1）
    
    Args:
        bbox: [x1, y1, x2, y2] 可能是归一化坐标或绝对像素坐标
        image_width: 图片宽度
        image_height: 图片高度
    
    Returns:
        归一化后的 bbox [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    
    # 检测是归一化坐标还是绝对像素坐标
    is_normalized = all(0 <= v <= 1.0 for v in [x1, y1, x2, y2])
    
    if is_normalized:
        return [x1, y1, x2, y2]
    else:
        # 转换为归一化坐标
        return [
            x1 / image_width,
            y1 / image_height,
            x2 / image_width,
            y2 / image_height,
        ]


def calculate_bbox_area(bbox: List[float]) -> float:
    """计算边界框面积（归一化）"""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def calculate_overlap(bbox1: List[float], bbox2: List[float]) -> float:
    """计算两个 bbox 的重叠率（IoU）"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    overlap_area = (x2 - x1) * (y2 - y1)
    area1 = calculate_bbox_area(bbox1)
    area2 = calculate_bbox_area(bbox2)
    union_area = area1 + area2 - overlap_area
    
    return overlap_area / union_area if union_area > 0 else 0.0


# ════════════════════════════════════════════
# Step 1: VLM 组件识别
# ════════════════════════════════════════════

MODULE_DETECTION_PROMPT = """你是建筑组件识别专家。请从建筑立面图中精确拆分出所有独立的、可复用的组件实例。

═══════════════════════════════════════════
【核心目标】
你输出的 components 是后续工作流的主数据源。
后续代码会基于 components[].reusable_group / subtype / type 聚合组件，因此你必须把每个组件的语义信息写准确。

重要原则：
1. 语义识别由你完成，代码只负责裁剪、校验、聚合和选择代表图。
2. 每个独立组件实例都必须单独输出，即使外观完全相同也不能省略。
3. reusable_group 用于表达"哪些实例属于同一种可复用资产"，同类实例必须使用完全相同的 reusable_group。
4. 不要依赖位置词来分组；left/right/top/bottom 只能出现在 name，不能出现在 reusable_group。
═══════════════════════════════════════════

═══════════════════════════════════════════
【组件拆分规则】

【1】墙面 wall
- 不要把整层墙面框成一个巨大 wall。
- wall 应按"窗间墙/门间墙/自然材质块"拆成可复用面板。
- 面板边界由窗户、门、柱子、木梁、材质分割线决定。
- 如果同一视觉外观的墙面面板出现多次，每个实例都要单独输出，但 reusable_group 保持相同。
- 墙面面板要覆盖真实墙体空白区域，不要只包住窗户周围装饰。

【2】屋顶 roof / roof_slope / dormer
- 每个独立屋顶斜面单独识别为 roof_slope。
- 主屋顶、侧屋顶、老虎窗屋顶、山墙三角面要分开。
- subtype 用视觉形态描述：steep_slope / medium_slope / shallow_slope / gable / dormer_roof 等。
- reusable_group 应包含视觉特征，例如 roof_slope_steep_green_tile。

【3】开口 window / door / shopfront
- 每扇窗、每扇门、每个商铺橱窗都单独识别。
- 不要把相邻窗户合并成一个 bbox。
- 窗框、窗台、百叶窗如果视觉上是同一个窗户资产的一部分，可以和窗户合在一起。
- 如果门板、门框、门顶装饰明显可独立复用，可以拆开；否则作为 door 的完整资产。

【4】附属结构 balcony / chimney / staircase / awning / parapet / porch
- 阳台栏杆与窗户分开。
- 烟囱、楼梯、遮阳篷、女儿墙、门廊分别单独识别。
- 如果多个实例外观一致，使用相同 reusable_group。

【5】装饰 ornament
- 小型装饰不要全部粗略写成 ornament。
- type 统一用 ornament，subtype 必须细分：spire / railing / lamp / plant / molding / sign / beam / other。
- 例如尖顶、壁灯、植物、木梁、装饰线脚必须分到不同 reusable_group。
═══════════════════════════════════════════

═══════════════════════════════════════════
【type 与 subtype 规范】

type 必须使用以下主类型之一（按 category 分组）：

**建筑结构**: wall, roof, roof_slope, window, door, shopfront, balcony, chimney, staircase, awning, parapet, dormer, base, porch, ornament

**自然元素**（场景中出现时识别）: tree, flower, grass, stone, crop

**场景道具**（场景中出现时识别）: furniture, vehicle, tool, building_material

subtype 用英文小写下划线，描述视觉细分类，例如：
- window: arch_blue_shutter, round_stained_glass, rectangular_wood_frame, shop_display
- door: arched_pink_wood, double_wood, glass_shop_door
- wall: stucco_beige_panel, wood_beam_panel, brick_panel, stone_base_panel
- roof_slope: steep_green_tile, medium_green_tile, dormer_green_tile, thatch_side
- ornament: spire, lamp_black_metal, plant_ivy, railing_black_metal, molding_stone, beam_wood
- tree: conifer_tall, broadleaf_shrub, round_deciduous
- flower: blue_small, red_small, flower_box
- furniture: bench_wood, small_table, metal_chair
- vehicle: yellow_bicycle, wheelbarrow, cart
═══════════════════════════════════════════

═══════════════════════════════════════════
【reusable_group 命名规则 - 非常重要】

格式建议：
[type]_[subtype核心]_[颜色/材质/关键特征]

必须遵守：
1. 全部使用英文小写和下划线。
2. 不要包含位置词：left, right, top, bottom, upper, lower, center, 1f, 2f。
3. 同一种视觉外观必须使用完全相同的 reusable_group。
4. 不同资产不能混到同一个 reusable_group。
5. reusable_group 表达“资产长什么样”，不是“它在哪里”。

正确例子：
- window_arch_blue_shutter
- roof_slope_steep_green_tile
- wall_stucco_beige_panel
- ornament_lamp_black_metal
- ornament_plant_ivy
- shopfront_striped_awning

错误例子：
- window_left_blue（包含位置）
- upper_window_arch（包含位置）
- decoration（太笼统）
- facade_001（无语义）
═══════════════════════════════════════════

═══════════════════════════════════════════
【bbox 坐标规则】
1. bbox 格式：[x1, y1, x2, y2]
2. 单位必须是整数像素值，不要输出小数。
3. x1 < x2, y1 < y2。
4. 坐标必须紧贴组件视觉边缘。
5. 所有坐标必须在图片范围内。
6. 不要输出归一化坐标。
═══════════════════════════════════════════

═══════════════════════════════════════════
【描述字段规范】

chinese_description：给人阅读，也作为生成兜底描述。
格式：[组件类型] + [视觉特征] + [颜色] + [材质] + [比例/尺寸感] + [装饰细节]

良好示例：
- "蓝色百叶尖拱窗，棕色木质窗框，窄高比例，带浅色石质窗台"
- "米色灰泥窗间墙面板，竖向长方形比例，带轻微砖缝纹理和木梁边框"
- "绿色陶土瓦陡坡屋顶斜面，宽大三角斜面，瓦片纹理密集"
- "黑色金属壁灯，细长灯架，暖黄色灯罩，哥特式装饰"

generation_prompt：专门给后续图像生成模型使用。
要求：描述单个可独立资产，不要写位置，不要写 bbox，不要写“左侧/右侧/上方”。
示例：
- "单个蓝色百叶尖拱窗，棕色木质窗框，浅色石质窗台，45度等轴游戏建筑资产"
- "单块米色灰泥墙面板，带木梁边框和细微砖缝纹理，45度等轴游戏建筑资产"

material：英文或中英混合均可，简短列出主要材质，例如 wood, glass, stone, stucco。
color：简短列出主要颜色，例如 blue shutter, brown frame, beige wall。
═══════════════════════════════════════════

═══════════════════════════════════════════
【输出格式 - 严格 JSON】
只输出 JSON，不要输出解释文字。components 数组中每个对象必须包含所有字段。

{
  "components": [
    {
      "id": "comp_001",
      "category": "建筑结构",
      "type": "window",
      "subtype": "arch_blue_shutter",
      "name": "window_arch_blue_shutter_001",
      "reusable_group": "window_arch_blue_shutter",
      "chinese_description": "蓝色百叶尖拱窗，棕色木质窗框，窄高比例，带浅色石质窗台",
      "generation_prompt": "单个蓝色百叶尖拱窗，棕色木质窗框，浅色石质窗台，45度等轴游戏建筑资产",
      "material": "wood, glass, stone",
      "color": "blue shutter, brown frame, pale stone sill",
      "bbox": [120, 260, 210, 430],
      "confidence": 0.94
    },
    {
      "id": "comp_002",
      "category": "建筑结构",
      "type": "wall",
      "subtype": "stucco_beige_panel",
      "name": "wall_stucco_beige_panel_001",
      "reusable_group": "wall_stucco_beige_panel",
      "chinese_description": "米色灰泥窗间墙面板，竖向长方形比例，带轻微砖缝纹理和木梁边框",
      "generation_prompt": "单块米色灰泥墙面板，带木梁边框和细微砖缝纹理，45度等轴游戏建筑资产",
      "material": "stucco, wood",
      "color": "beige wall, brown wood trim",
      "bbox": [220, 450, 480, 650],
      "confidence": 0.92
    }
  ]
}

═══════════════════════════════════════════
【自检清单】
□ 每个独立组件实例都已输出，外观相同也没有省略。
□ type 使用主类型，subtype 描述细分类。
□ reusable_group 不包含位置词，同类资产命名完全一致。
□ ornament 已通过 subtype/reusable_group 细分，不是一组混杂装饰。
□ wall 不是整层大框，而是自然墙面面板。
□ 相邻窗户、门、阳台没有被错误合并。
□ bbox 是整数像素值，且紧贴组件边缘。
□ 每个组件都有 chinese_description 和 generation_prompt。"""


def detect_modules(image_path: str, max_retries: int = 3) -> Dict:
    """调用 VLM 识别建筑组件（带重试机制）"""
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
                    {"type": "text", "text": MODULE_DETECTION_PROMPT},
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
        "max_tokens": 8000,
    }

    headers = {
        "Authorization": f"Bearer {VLM_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, max_retries + 1):
        print(f"  🔍 调用 VLM 识别组件... (尝试 {attempt}/{max_retries})")
        try:
            resp = requests.post(
                VLM_API_URL,
                json=payload,
                headers=headers,
                timeout=300,  # 300秒超时，VLM 处理大图可能较慢
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            # 提取 JSON
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            # 尝试修复被截断的JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # 尝试找到最后一个完整的组件对象，截断后面的内容
                last_brace = content.rfind("}")
                # 找到最后一个完整的 } 后尝试闭合数组和对象
                for trim_pos in range(len(content) - 1, -1, -1):
                    if content[trim_pos] == '}':
                        trial = content[:trim_pos + 1]
                        # 尝试补全 JSON 结构
                        open_brackets = trial.count('[') - trial.count(']')
                        open_braces = trial.count('{') - trial.count('}')
                        # 尝试去除末尾多余逗号
                        trial = trial.rstrip().rstrip(',')
                        trial += ']' * max(open_brackets, 0) + '}' * max(open_braces, 0)
                        try:
                            result = json.loads(trial)
                            print(f"  ⚠️  JSON被截断，已自动修复（截断位置: {trim_pos}）")
                            return result
                        except json.JSONDecodeError:
                            continue
                # 所有修复尝试都失败
                raise

        except requests.exceptions.ReadTimeout:
            print(f"  ⚠️  请求超时 (第 {attempt} 次)")
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"     等待 {wait} 秒后重试...")
                import time
                time.sleep(wait)
            else:
                print(f"  ❌ 已达最大重试次数，VLM 请求持续超时")
                raise
        except requests.exceptions.ConnectionError as e:
            print(f"  ⚠️  连接错误 (第 {attempt} 次): {e}")
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"     等待 {wait} 秒后重试...")
                import time
                time.sleep(wait)
            else:
                raise
        except json.JSONDecodeError as e:
            print(f"  ❌ VLM 返回的内容不是有效 JSON: {e}")
            print(f"     原始返回: {content[:500]}")
            raise


# ════════════════════════════════════════════
# Step 2: 质量校验
# ════════════════════════════════════════════

def validate_components(components: List[Dict], image_width: int, image_height: int) -> List[str]:
    """
    校验组件识别的合法性
    返回警告列表
    """
    warnings = []
    
    # 1. 检查是否有基本组件
    types = [c.get("type") for c in components]
    if "roof" not in types and "wall" not in types:
        warnings.append("未识别到屋顶或墙体，可能是输入图片有问题")
    
    # 2. 检查门的位置
    for comp in components:
        if comp.get("type") == "door":
            bbox = normalize_bbox(comp.get("bbox", [0, 0, 1, 1]), image_width, image_height)
            if bbox[1] < 0.5:  # 门的中点应该在底部
                warnings.append(f"门 {comp.get('id')} 位置偏上，可能识别有误")
    
    # 3. 检查组件重叠
    for i, c1 in enumerate(components):
        bbox1 = normalize_bbox(c1.get("bbox", [0, 0, 1, 1]), image_width, image_height)
        for c2 in components[i+1:]:
            bbox2 = normalize_bbox(c2.get("bbox", [0, 0, 1, 1]), image_width, image_height)
            overlap = calculate_overlap(bbox1, bbox2)
            if overlap > 0.3:
                warnings.append(f"{c1['id']} 和 {c2['id']} 重叠率 {overlap:.0%}")
    
    # 4. 检查组件数量合理性
    window_count = types.count("window")
    if window_count > 15:
        warnings.append(f"识别到 {window_count} 个窗户，数量偏多，请检查是否有误")
    
    # 5. 检查墙面数量
    wall_count = types.count("wall")
    if wall_count < 2:
        warnings.append(f"只识别到 {wall_count} 个墙面，可能需要更多楼层墙面")
    
    # 6. 检查 reusable_group 一致性
    # 如果 type + subtype + chinese_description 相同，reusable_group 也应该相同
    group_map = {}
    for comp in components:
        key = (
            comp.get("type", ""),
            comp.get("subtype", ""),
            comp.get("chinese_description", ""),
        )
        reusable_group = comp.get("reusable_group", "")
        if key in group_map:
            if group_map[key] != reusable_group:
                warnings.append(
                    f"reusable_group 不一致：{comp.get('name')} 的 "
                    f"type/subtype/description 相同但 reusable_group 为 "
                    f"'{reusable_group}'（应为 '{group_map[key]}'）"
                )
        else:
            group_map[key] = reusable_group
    
    return warnings


# ════════════════════════════════════════════
# Step 3: 裁剪组件
# ════════════════════════════════════════════

def crop_component(
    image_path: str,
    bbox: List[float],
    output_path: str,
    padding: float = 0.02
) -> bool:
    """
    根据 bbox 裁剪组件图
    
    Args:
        image_path: 原图路径
        bbox: [x1, y1, x2, y2] 可能是归一化坐标(0-1)或绝对像素坐标
        output_path: 输出路径
        padding: 边距比例（仅对归一化坐标有效）
    
    Returns:
        是否成功
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            x1, y1, x2, y2 = bbox
            
            # 检测 bbox 是归一化坐标还是绝对像素坐标
            # 归一化坐标：所有值都在 0-1 范围内
            # 绝对像素坐标：值通常 > 1（除非图片很小）
            is_normalized = all(0 <= v <= 1.0 for v in [x1, y1, x2, y2])
            
            if is_normalized:
                # 归一化坐标 → 转为像素
                # 添加边距
                pad_x = padding * (x2 - x1)
                pad_y = padding * (y2 - y1)
                
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(1, x2 + pad_x)
                y2 = min(1, y2 + pad_y)
                
                left = int(x1 * width)
                top = int(y1 * height)
                right = int(x2 * width)
                bottom = int(y2 * height)
            else:
                # 绝对像素坐标 → 直接使用
                left = int(x1)
                top = int(y1)
                right = int(x2)
                bottom = int(y2)
            
            # 确保最小尺寸（至少 20px）
            if right - left < 20 or bottom - top < 20:
                print(f"    ⚠️ 组件太小({right-left}x{bottom-top}px)，跳过裁剪")
                return False
            
            # 裁剪
            cropped = img.crop((left, top, right, bottom))
            cropped.save(output_path, "PNG")
            return True
            
    except Exception as e:
        print(f"    ❌ 裁剪失败: {e}")
        return False


def build_reusable_groups(
    components: List[Dict],
    component_dir: str,
    image_width: int,
    image_height: int,
) -> List[Dict]:
    """
    将识别到的组件按类型分组，形成可复用组件库。

    每个类型取一个代表性实例（面积最大、置信度最高的），
    同时列出所有实例，方便后续复用。

    注意：即使组件裁剪失败，其 name 也要进入 reusable_groups，
    因为 Phase3 需要通过 name 建立映射。

    Args:
        components: 组件列表（含 cropped_file, original_bbox）
        component_dir: 裁剪图所在目录路径
        image_width: 原始正交图宽度
        image_height: 原始正交图高度

    Returns:
        可复用组件分组列表
    """
    from collections import defaultdict

    # 按 VLM 返回的 reusable_group 分组（不依赖 cropped_file，所有组件都参与）
    groups = defaultdict(list)
    for comp in components:
        # 即使没有 cropped_file 也要参与分组，用于 Phase3 去重映射
        # 直接使用 VLM 返回的 reusable_group，确保组件按视觉特征正确细分
        reusable_group = comp.get("reusable_group", "")
        if not reusable_group:
            # 如果没有 reusable_group，使用 type 作为兜底
            reusable_group = comp.get("type", "unknown")
        groups[reusable_group].append(comp)

    output_dir = Path(component_dir)

    reusable_groups = []
    for comp_type, instances in groups.items():
        # 计算每个实例的实际像素尺寸（只有裁剪成功的才有）
        instance_details = []
        for inst in instances:
            cropped_file = inst.get("cropped_file")
            if cropped_file:
                cropped_path = output_dir / cropped_file
                try:
                    with Image.open(cropped_path) as img:
                        w, h = img.size
                except Exception:
                    w, h = 0, 0
            else:
                w, h = 0, 0

            instance_details.append({
                "id": inst.get("id", ""),
                "name": inst.get("name", ""),  # Phase3 需要这个建立映射
                "file": cropped_file,
                "pixel_size": [w, h],
                "confidence": inst.get("confidence", 0),
                "chinese_description": inst.get("chinese_description", ""),  # Phase3 需要这个生成 prompt
                "type": inst.get("type", ""),  # Phase3 需要这个判断组件类型
            })

        # 选择代表：优先面积最大，其次置信度最高
        # 注意：只用裁剪成功的计算面积
        valid_instances = [d for d in instance_details if d["pixel_size"][0] > 0]
        if valid_instances:
            def sort_key(inst):
                w, h = inst["pixel_size"]
                area = w * h
                return (area, inst["confidence"])

            representative = max(valid_instances, key=sort_key)
            widths = [d["pixel_size"][0] for d in valid_instances]
            heights = [d["pixel_size"][1] for d in valid_instances]
        else:
            # 全部裁剪失败，用第一个实例
            representative = instance_details[0]
            widths = [0]
            heights = [0]

        size_range = {
            "min_w": min(widths) if widths else 0,
            "max_w": max(widths) if widths else 0,
            "min_h": min(heights) if heights else 0,
            "max_h": max(heights) if heights else 0,
        }

        # 统计裁剪成功的数量
        cropped_count = sum(1 for d in instance_details if d["file"] is not None)

        reusable_groups.append({
            "type": comp_type,
            "base_type": comp_type.split("_")[0],
            "count": len(instance_details),
            "cropped_count": cropped_count,  # 实际裁剪成功的数量
            "representative_id": representative.get("id", ""),
            "representative_file": representative.get("file"),  # 可能是 None
            "size_range": size_range,
            "all_instances": instance_details,
            "chinese_description": representative.get("chinese_description", ""),  # 组的整体描述
        })

    # 按数量降序排列，数量多的排前面
    reusable_groups.sort(key=lambda g: g["count"], reverse=True)

    return reusable_groups


# ════════════════════════════════════════════
# Step 5: 楼层区域归并
# ════════════════════════════════════════════

def build_floor_zones(
    components: List[Dict],
    image_width: int,
    image_height: int,
    total_floors: int = 3,
    phase1_analysis: Optional[Dict] = None,
) -> List[Dict]:
    """
    将识别到的组件按楼层区域归并，生成楼层区域图。
    
    综合使用 Phase 1 的语义分析数据和组件聚类分析来确定楼层切分：
    1. 利用 Phase 1 的 building_style、ground_floor、roof 等语义信息辅助判断
    2. 根据组件的 y 中心位置进行聚类，找到自然的组件分组间隙
    3. 结合楼层数和关键组件位置（屋顶、门、基座）确定分界线
    4. 确保每个楼层都有对应的区域，不遗漏
    
    Args:
        components: 组件列表（含 bbox 等信息）
        image_width: 图片宽度
        image_height: 图片高度
        total_floors: 楼层数（来自 Phase 1）
        phase1_analysis: Phase 1 的完整分析结果（可选，用于辅助切分）
    
    Returns:
        楼层区域列表
    """
    if not components:
        return []

    # ── Step 1: 利用 Phase 1 语义信息提取建筑结构线索 ──
    has_ground_shop = False  # 底层是否为商铺
    has_balcony = False      # 是否有阳台层
    roof_type = ""
    
    if phase1_analysis:
        ground_desc = phase1_analysis.get("ground_floor", "")
        has_ground_shop = "shop" in ground_desc.lower() or "store" in ground_desc.lower()
        roof_type = phase1_analysis.get("roof", {}).get("type", "")
        # 检查装饰元素中是否有阳台
        decorative = phase1_analysis.get("decorative_elements", [])
        has_balcony = any("balcony" in d.lower() for d in decorative)
    
    # ── Step 2: 找出各关键组件的位置 ──
    # 注意：组件的 bbox 可能是绝对像素坐标，需要转换为归一化坐标
    roof_bottom = 0.0
    door_top = 1.0
    base_top = 1.0
    balcony_top = 1.0
    balcony_bottom = 0.0
    
    for comp in components:
        bbox = normalize_bbox(comp.get("bbox", [0, 0, 1, 1]), image_width, image_height)
        cy = (bbox[1] + bbox[3]) / 2
        if comp.get("type") == "roof":
            roof_bottom = max(roof_bottom, bbox[3])
        if comp.get("type") == "door":
            door_top = min(door_top, bbox[1])
        if comp.get("type") == "base":
            base_top = min(base_top, bbox[1])
        if comp.get("type") == "balcony":
            balcony_top = min(balcony_top, bbox[1])
            balcony_bottom = max(balcony_bottom, bbox[3])

    # ── Step 3: 用组件聚类找到自然的楼层分界 ──
    # 收集所有组件的 y 中心点，排序后找最大间隙
    comp_centers = []
    for comp in components:
        bbox = normalize_bbox(comp.get("bbox", [0, 0, 1, 1]), image_width, image_height)
        cy = (bbox[1] + bbox[3]) / 2
        comp_centers.append(cy)
    comp_centers.sort()
    
    # 计算相邻组件中心的间隙
    gaps = []
    for i in range(len(comp_centers) - 1):
        gap = comp_centers[i + 1] - comp_centers[i]
        if gap > 0.03:  # 间隙大于 3% 才有意义
            midpoint = (comp_centers[i] + comp_centers[i + 1]) / 2
            gaps.append((gap, midpoint, comp_centers[i], comp_centers[i + 1]))
    
    # 按间隙大小降序排列
    gaps.sort(key=lambda g: g[0], reverse=True)
    
    print(f"    [DEBUG] 组件聚类间隙（前5个）:")
    for gap, mid, y1, y2 in gaps[:5]:
        print(f"      间隙={gap:.2f}  中点={mid:.2f}  ({y1:.2f} ~ {y2:.2f})")

    # ── Step 4: 构建分界线 ──
    # 固定分界：屋顶上边界 (0.0)、屋顶下边界
    building_top = roof_bottom
    
    # 底层区域：从门/基座附近开始到底部
    # 优先用基座上边界，其次用门上方
    if base_top < 1.0:
        ground_start = base_top - 0.02
    elif door_top < 1.0:
        ground_start = door_top - 0.05
    else:
        ground_start = 0.75  # 默认

    # 收集候选分界线
    candidate_boundaries = [0.0, roof_bottom]
    
    # 如果有阳台层且有足够间隙，插入阳台分界线
    if has_balcony and balcony_top < 1.0:
        candidate_boundaries.append(round((balcony_top + balcony_bottom) / 2, 2))
    
    # 从间隙中选择最佳分界线（排除屋顶和底层附近的）
    for gap, mid, y1, y2 in gaps:
        if mid > roof_bottom + 0.03 and mid < ground_start - 0.03:
            # 这个间隙在主体区域内，可以作为分界线
            if mid not in [round(b, 2) for b in candidate_boundaries]:
                candidate_boundaries.append(round(mid, 2))
                break  # 只取最大的一个间隙作为中间分界线
    
    candidate_boundaries.append(round(ground_start, 2))
    candidate_boundaries.append(1.0)
    
    # 去重排序
    boundaries = sorted(set(round(b, 2) for b in candidate_boundaries))
    
    # ── Step 5: 确保分界线数量合理 ──
    # 期望的 zone 数量 = 屋顶 + (total_floors - 1) 个中间层 + 底层
    # 最少 total_floors 个 zone
    expected_zones = total_floors + 1  # 屋顶 + 各楼层（含底层）
    
    # 如果分界线不足，用等分补充
    while len(boundaries) - 1 < expected_zones:
        # 找到最大的区间，从中插入一条分界线
        max_gap = 0
        max_idx = 0
        for i in range(len(boundaries) - 1):
            gap = boundaries[i + 1] - boundaries[i]
            if gap > max_gap:
                max_gap = gap
                max_idx = i
        # 在最大区间中点插入
        mid = round((boundaries[max_idx] + boundaries[max_idx + 1]) / 2, 2)
        boundaries.insert(max_idx + 1, mid)
    
    print(f"    [DEBUG] 最终 boundaries = {boundaries} (期望 {expected_zones} 个 zone)")

    # ── Step 6: 定义楼层区域 ──
    zone_names = ["楼顶区域", "上层区域", "中层区域", "下层区域", "底层区域", "底部区域"]
    zone_types = ["roof", "upper", "middle", "lower", "ground", "base"]
    
    zones_def = []
    for i in range(len(boundaries) - 1):
        idx = min(i, len(zone_names) - 1)
        zones_def.append({
            "type": zone_types[idx],
            "name": zone_names[idx],
            "y1": boundaries[i],
            "y2": boundaries[i + 1],
        })

    zones = []
    for zone_def in zones_def:
        zone_comps = []
        for comp in components:
            bbox = normalize_bbox(comp.get("bbox", [0, 0, 1, 1]), image_width, image_height)
            cy = (bbox[1] + bbox[3]) / 2
            if zone_def["y1"] <= cy < zone_def["y2"]:
                zone_comps.append(comp)

        if not zone_comps:
            continue

        # 生成区域描述词
        comp_summary = []
        for comp in zone_comps:
            ctype = comp.get("type", "")
            cname = comp.get("name", "")
            notes = comp.get("notes", "")
            comp_summary.append(f"{cname} ({ctype})")

        zt = [c.get("type", "") for c in zone_comps]
        type_counts = {}
        for t in zt:
            type_counts[t] = type_counts.get(t, 0) + 1

        description = f"{zone_def['name']}：包含 {', '.join(comp_summary)}"
        type_desc = "、".join(f"{v}个{t}" for t, v in type_counts.items())

        zones.append({
            "type": zone_def["type"],
            "name": zone_def["name"],
            "bbox": [0.0, zone_def["y1"], 1.0, zone_def["y2"]],
            "component_count": len(zone_comps),
            "component_types": type_desc,
            "description": description,
            "components": [
                {
                    "id": c.get("id"),
                    "type": c.get("type"),
                    "name": c.get("name"),
                    "bbox": c.get("bbox"),
                    "notes": c.get("notes", ""),
                    "chinese_description": c.get("chinese_description", ""),
                }
                for c in zone_comps
            ],
        })

    return zones


# ════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════

def get_next_output_dir(output_dir: Path, building_name: str, model: str) -> Path:
    """
    扫描输出目录，找到已有 {building_name}_XX_* 格式目录的最大编号，
    返回下一个编号的目录路径。格式: {building_name}_01_{MODELNAME}

    兼容旧格式：如果存在旧格式目录（如 SurfCG/ 无编号），也会被保留，
    新编号从 max(旧编号) + 1 开始。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    max_num = 0
    import re
    safe_model = re.sub(r"[^a-zA-Z0-9._-]", "_", model)
    safe_model = re.sub(r"_+", "_", safe_model).strip("_")

    prefix = f"{building_name}_"
    for d in output_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name.startswith(prefix):
            # 提取编号部分：SurfCG_03_xxx → "03"
            rest = name[len(prefix):]
            parts = rest.split("_")
            if parts:
                try:
                    num = int(parts[0])
                    if num > max_num:
                        max_num = num
                except ValueError:
                    pass

    next_num = max_num + 1
    dirname = f"{building_name}_{next_num:02d}_{safe_model}"
    return output_dir / dirname


def process_ortho_image(image_path: str, overwrite: bool = False):
    """
    处理单张正交图：识别组件 → 裁剪 → 分组 → 楼层归并 → 生成提示词 → 输出

    每次运行自动带编号+模型名后缀保存到新目录，如 SurfCG_01_qwen2.5-vl-72b/
    不会覆盖已有目录，方便不同模型/不同运行的结果对照。

    Args:
        image_path: 正交图路径
        overwrite: 已废弃参数，保留兼容但不生效（总是创建新目录）
    """
    filename = Path(image_path).stem
    building_name = filename.replace("_ortho", "")  # 去掉 _ortho 后缀

    # 确定模型名，用于目录命名
    from config import VLM_MODEL

    # 自动编号输出目录
    output_subdir = get_next_output_dir(OUTPUT_DIR, building_name, VLM_MODEL)
    manifest_path = output_subdir / "manifest.json"

    print(f"\n{'='*60}")
    print(f"🔧 Phase 2: 模组识别 - {output_subdir.name}")
    print(f"{'='*60}")

    # 确保输出目录存在
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # 获取图片尺寸
    with Image.open(image_path) as img:
        img_width, img_height = img.size
    
    # ── Step 1: VLM 识别 ──
    print("\n[Step 1/6] 识别建筑组件...")

    result = detect_modules(image_path)
    components = result.get("components", [])
    print(f"  ✅ VLM 识别到 {len(components)} 个组件")
    
    # ── Step 2: 质量校验 ──
    print("\n[Step 2/6] 质量校验...")
    warnings = validate_components(components, img_width, img_height)
    if warnings:
        print("  ⚠️  警告:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("  ✅ 校验通过")
    
    # ── Step 3: 裁剪组件 ──
    print("\n[Step 3/6] 裁剪组件图...")
    
    cropped_components = []
    for i, comp in enumerate(components, 1):
        comp_type = comp.get("type", "unknown")
        comp_name = comp.get("name", f"component_{i}")
        
        # 生成文件名
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in comp_name)
        output_filename = f"{i:03d}_{comp_type}_{safe_name}.png"
        output_path = output_subdir / output_filename
        
        bbox = comp.get("bbox", [0, 0, 1, 1])
        # 检测是归一化还是绝对像素坐标
        is_pixel = not all(0 <= v <= 1.0 for v in bbox)
        if i <= 3:  # 调试：只打印前3个
            if is_pixel:
                print(f"    [DEBUG] bbox像素值: {bbox} → 归一化: {[v/1000 if i%2==0 else v/1000 for i,v in enumerate(bbox)]}")
            else:
                print(f"    [DEBUG] bbox归一化值: {bbox}")
        print(f"  [{i}/{len(components)}] {comp_type}: {comp_name}...", end=" ")
        success = crop_component(image_path, bbox, str(output_path))
        
        if success:
            print(f"✅ ({output_path.name})")
            cropped_components.append({
                **comp,
                "cropped_file": output_filename,
                "original_bbox": bbox,
            })
        else:
            print(f"⚠️ 跳过")
            cropped_components.append({
                **comp,
                "cropped_file": None,
                "original_bbox": bbox,
            })
    
    # ── Step 4: 按类型分组可复用组件 ──
    print("\n[Step 4/6] 构建可复用组件分组...")
    reusable_groups = build_reusable_groups(cropped_components, str(output_subdir), img_width, img_height)
    print(f"  ✅ 分为 {len(reusable_groups)} 个可复用类型:")
    for group in reusable_groups:
        repr_name = group["representative_file"]
        print(f"    📦 {group['type']:<12} ×{group['count']} 个实例 → 代表: {repr_name}")
    
    # ── Step 5: 楼层区域归并 + 裁剪 ──
    # 从 Phase 1 分析结果中读取楼层数
    phase1_analysis_path = Path(image_path).parent / f"{Path(image_path).stem.replace('_ortho', '')}_analysis.json"
    total_floors = 3  # 默认值
    phase1_analysis = None
    if phase1_analysis_path.exists():
        with open(phase1_analysis_path, "r", encoding="utf-8") as f:
            phase1_analysis = json.load(f)
        total_floors = int(phase1_analysis.get("total_floors", 3))
        print(f"  📐 Phase 1 分析结果：{total_floors} 层建筑")
        print(f"  📐 风格: {phase1_analysis.get('building_style', '?')}, 底层: {phase1_analysis.get('ground_floor', '?')}")
    else:
        print(f"  ⚠️  未找到 Phase 1 分析结果，使用默认楼层数: {total_floors}")
    
    print("\n[Step 5/6] 楼层区域归并...")
    floor_zones = build_floor_zones(components, img_width, img_height, total_floors, phase1_analysis)
    
    # ── 合并过薄的区域 ──
    MIN_ZONE_HEIGHT_RATIO = 0.10  # 区域最小高度占比（相对于图片高度），太薄的区域会导致API图片宽高比超限
    min_pixel_height = int(img_height * MIN_ZONE_HEIGHT_RATIO)
    
    merged_zones = []
    for zone in floor_zones:
        zone_height_px = int((zone["bbox"][3] - zone["bbox"][1]) * img_height)
        if zone_height_px < min_pixel_height and merged_zones:
            # 合并到前一个区域
            prev = merged_zones[-1]
            prev["bbox"][3] = zone["bbox"][3]  # 扩展前一个区域的底部
            prev["component_count"] += zone["component_count"]
            prev["component_types"] += "、" + zone["component_types"]
            prev["description"] += "、" + zone["description"]
            prev["components"].extend(zone["components"])
            print(f"    ⚠️ {zone['name']} 高度不足（{zone_height_px}px），已合并到 {prev['name']}")
        else:
            merged_zones.append(zone)
    
    floor_zones = merged_zones
    
    print(f"  ✅ 合并为 {len(floor_zones)} 个楼层区域:")
    for zone in floor_zones:
        print(f"    🏗️ {zone['name']}  [{zone['bbox'][1]:.2f} ~ {zone['bbox'][3]:.2f}]  含 {zone['component_count']} 个组件: {zone['component_types']}")

    # 裁剪楼层区域图（根据 zone 类型应用不同的扩展策略）
    zone_cropped = []
    for zi, zone in enumerate(floor_zones, 1):
        zone_filename = f"zone_{zi:02d}_{zone['type']}.png"
        zone_path = output_subdir / zone_filename
        
        # 根据 zone 类型决定扩展策略
        zone_type = zone.get("type", "")
        zone_bbox = zone["bbox"][:]  # 复制一份，不修改原数据
        
        if zone_type in ["roof", "base"]:
            # 屋顶和地基刚好裁剪，不扩展
            print(f"    📐 {zone_filename}: 刚好裁剪 (roof/base)")
        else:
            # 中间层：上下各扩展 2%，避免裁断连接处
            height = zone_bbox[3] - zone_bbox[1]
            expand_ratio = 0.02
            zone_bbox[1] = max(0.0, zone_bbox[1] - height * expand_ratio)  # 向上扩展
            zone_bbox[3] = min(1.0, zone_bbox[3] + height * expand_ratio)  # 向下扩展
            print(f"    📐 {zone_filename}: 上下各扩展 2% (middle layer)")
        
        success = crop_component(image_path, zone_bbox, str(zone_path), padding=0.01)
        if success:
            zone["cropped_file"] = zone_filename
            print(f"    ✅ {zone_filename}")
        else:
            zone["cropped_file"] = None
            print(f"    ⚠️ {zone_filename} 裁剪失败")
        zone_cropped.append(zone)

    # ── 保存 manifest ──
    from config import VLM_MODEL
    manifest = {
        "source": str(image_path),
        "source_type": "orthographic_front_view",
        "recognition_model": VLM_MODEL,
        "facade_width": img_width,
        "facade_height": img_height,
        "total_components": len(cropped_components),
        "total_reusable_types": len(reusable_groups),
        "total_floor_zones": len(zone_cropped),
        "components": cropped_components,
        "reusable_groups": reusable_groups,
        "floor_zones": zone_cropped,
        "warnings": warnings,
        "next_phase": "phase3_quadrant_split",
    }
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    # ── 打印摘要 ──
    print(f"\n{'='*60}")
    print(f"📊 Phase 2 处理完成")
    print(f"{'='*60}")
    print(f"  源文件: {Path(image_path).name}")
    print(f"  输出目录: {output_subdir.name}")
    print(f"  识别模型: {VLM_MODEL}")
    print(f"  识别实例数: {len(cropped_components)}")
    print(f"  可复用类型数: {len(reusable_groups)}")
    print(f"  组件清单: {manifest_path}")
    
    print(f"\n  实例列表:")
    type_counts = {}
    for comp in cropped_components:
        comp_type = comp.get("type", "unknown")
        type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
        status = "✅" if comp.get("cropped_file") else "⚠️"
        print(f"    {status} {comp_type:<12} - {comp.get('name', 'unknown')}")
    
    print(f"\n  实例统计: {', '.join(f'{k}×{v}' for k, v in type_counts.items())}")
    
    print(f"\n  可复用组件库:")
    for group in reusable_groups:
        sr = group["size_range"]
        print(f"    📦 {group['type']:<12} → 代表: {group['representative_file']}")
        print(f"       尺寸范围: {sr['min_w']}×{sr['min_h']} ~ {sr['max_w']}×{sr['max_h']}px")
        for inst in group["all_instances"]:
            print(f"         - {inst['file']}  ({inst['pixel_size'][0]}×{inst['pixel_size'][1]}px)")
    
    return manifest


def main():
    """主入口：扫描 Phase 1 输出目录，处理所有正交图"""
    # 确保目录存在
    PHASE1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 查找 phase1 输出的正交图
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    ortho_images = [
        f for f in PHASE1_OUTPUT_DIR.iterdir()
        if f.suffix.lower() in image_extensions and "_ortho" in f.stem
    ]
    
    # 如果目录为空，提示用户
    all_images = [
        f for f in PHASE1_OUTPUT_DIR.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not all_images:
        print(f"⚠️  phase1 输出目录为空: {PHASE1_OUTPUT_DIR}")
        print(f"   请先运行 phase1_standardize.py 生成正交图")
        print(f"   或者将正交图手动放入该目录")
        sys.exit(0)
    
    if not ortho_images:
        print(f"⚠️  未找到正交图 (*_ortho.png) 在: {PHASE1_OUTPUT_DIR}")
        print(f"   目录中的文件:")
        for f in all_images:
            print(f"     - {f.name}")
        print(f"\n   请确保文件名为 *_*_ortho.png 格式")
        sys.exit(0)
    
    print(f"📁 找到 {len(ortho_images)} 张正交图，开始处理...")
    
    # 处理每张正交图
    for img_path in ortho_images:
        process_ortho_image(str(img_path))
    
    print(f"\n{'='*60}")
    print(f"🎉 Phase 2 模组识别全部完成")
    print(f"{'='*60}")
    print(f"  输出目录: {OUTPUT_DIR}")

    # ── 历史运行一览 ──
    all_runs = sorted(d.name for d in OUTPUT_DIR.iterdir() if d.is_dir())
    if all_runs:
        print(f"\n  📋 所有 Phase 2 运行记录:")
        for run_name in all_runs:
            run_dir = OUTPUT_DIR / run_name
            mf = run_dir / "manifest.json"
            if mf.exists():
                try:
                    with open(mf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    model = data.get("recognition_model", "?")
                    total = data.get("total_components", "?")
                    groups = data.get("total_reusable_types", "?")
                    print(f"    {run_name}  模型={model}  组件={total}  可复用组={groups}")
                except Exception:
                    print(f"    {run_name}  (读取失败)")
            else:
                print(f"    {run_name}  (无manifest)")


if __name__ == "__main__":
    main()