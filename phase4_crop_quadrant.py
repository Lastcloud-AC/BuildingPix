"""
Phase 4: 四宫格裁剪
==========================
输入：phase3_quadrant_generation.py 输出的四宫格图片和 manifest
输出：裁剪后的单个组件图片

流程：
  1. 读取 Phase 3 的 manifest.json
  2. 对每个四宫格图片，按 2x2 网格裁剪为 4 个子图
  3. 根据 components 数量判断哪些格子是占位的（整体视角）
  4. 仅保留有效组件图片，舍弃占位格子
  5. 按 zone 分文件夹存储，生成裁剪后的 manifest

占位判断规则：
  - 每批最多4个组件，按 左上→右上→左下→右下 顺序排列
  - 若 components 数量 < 4，则后面的格子是"整体视角"占位格子
  - prompt 中包含"整体视角"的描述即为占位

使用方式：
  python phase4_crop_quadrant.py

依赖：
  - PIL (Pillow)
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image
from datetime import datetime

# ─── 路径配置 ───
PHASE3_OUTPUT_DIR = Path(__file__).parent / "output" / "phase3_quadrant"
OUTPUT_DIR = Path(__file__).parent / "output" / "phase4_cropped"

# ─── 四宫格裁剪参数 ───
GRID_SIZE = 2  # 2x2 网格
# 裁剪位置顺序：左上、右上、左下、右下
CROP_POSITIONS = [
    (0, 0),  # 左上
    (1, 0),  # 右上
    (0, 1),  # 左下
    (1, 1),  # 右下
]


# ════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════

def is_placeholder_component(batch: dict, position_index: int) -> bool:
    """
    判断指定位置的格子是否为占位格子

    Args:
        batch: manifest 中的 batch 数据
        position_index: 格子位置索引 (0-3)

    Returns:
        True 表示该格子是占位的（整体视角），应舍弃
    """
    components = batch.get("components", [])
    # 如果该位置索引 >= components 数量，说明是占位格子
    return position_index >= len(components)


def crop_quadrant(image_path: str) -> list:
    """
    将四宫格图片裁剪为 4 个子图

    Args:
        image_path: 四宫格图片路径

    Returns:
        4 个裁剪后的 PIL Image 对象列表，顺序为 左上、右上、左下、右下
    """
    img = Image.open(image_path)
    width, height = img.size
    cell_w = width // GRID_SIZE
    cell_h = height // GRID_SIZE

    cropped_images = []
    for col, row in CROP_POSITIONS:
        left = col * cell_w
        upper = row * cell_h
        right = left + cell_w
        lower = upper + cell_h
        cropped = img.crop((left, upper, right, lower))
        cropped_images.append(cropped)

    return cropped_images


def generate_crop_filename(zone_name: str, zone_type: str, component_name: str) -> str:
    """
    生成裁剪后组件的文件名

    Args:
        zone_name: 区域名称（如"楼顶区域"）
        zone_type: 区域类型（如"roof"）
        component_name: 组件名称

    Returns:
        文件名字符串
    """
    # 清理文件名中的特殊字符
    safe_zone_name = zone_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return f"{zone_type}_{safe_zone_name}_{component_name}.png"


# ════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════

def process_module(module_output_dir: Path, overwrite: bool = False) -> dict:
    """
    处理单个模块的 phase3 输出

    Args:
        module_output_dir: Phase3 输出的模块目录（如 output/phase3_quadrant/SurfCG）
        overwrite: 是否覆盖已有结果

    Returns:
        处理结果字典
    """
    manifest_path = module_output_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"  ❌ manifest.json 不存在，跳过: {module_output_dir.name}")
        return None

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    module_name = module_output_dir.name
    zone_results = manifest.get("zone_results", [])

    print(f"\n{'='*60}")
    print(f"✂️  Phase 4: 四宫格裁剪 - {module_name}")
    print(f"{'='*60}")
    print(f"  楼层区域数: {len(zone_results)}")

    # 创建输出目录
    output_subdir = OUTPUT_DIR / module_name
    output_subdir.mkdir(parents=True, exist_ok=True)

    # 统计
    total_components = 0
    cropped_count = 0
    skipped_placeholder = 0
    skipped_existing = 0
    failed_count = 0

    all_crop_results = []

    for zone_result in zone_results:
        zone_type = zone_result.get("zone_type", "unknown")
        zone_name = zone_result.get("zone_name", "未知区域")
        batches = zone_result.get("batches", [])

        # 为每个 zone 创建子文件夹
        zone_dir = output_subdir / f"{zone_type}_{zone_name}"
        zone_dir.mkdir(parents=True, exist_ok=True)

        zone_crop_results = []

        for batch in batches:
            batch_index = batch.get("batch_index", 1)
            status = batch.get("status", "unknown")
            output_file = batch.get("output_file")
            components = batch.get("components", [])

            if status != "generated" or not output_file:
                print(f"  ⏭️ 跳过未生成的批次: {zone_name} 批次{batch_index} (status={status})")
                continue

            quadrant_path = module_output_dir / output_file
            if not quadrant_path.exists():
                print(f"  ⚠️ 四宫格图片不存在: {output_file}")
                failed_count += 1
                continue

            print(f"\n  📦 {zone_name} - 批次{batch_index}: {len(components)} 个组件")

            # 裁剪四宫格
            cropped_images = crop_quadrant(str(quadrant_path))

            for pos_idx in range(4):
                total_components += 1

                # 判断是否为占位格子
                if is_placeholder_component(batch, pos_idx):
                    skipped_placeholder += 1
                    position_name = ["左上", "右上", "左下", "右下"][pos_idx]
                    print(f"    ⏭️ {position_name}: 占位格子（整体视角），舍弃")
                    zone_crop_results.append({
                        "position": position_name,
                        "position_index": pos_idx,
                        "component_name": None,
                        "is_placeholder": True,
                        "status": "discarded",
                    })
                    continue

                # 有效组件
                component_name = components[pos_idx]
                crop_filename = generate_crop_filename(zone_name, zone_type, component_name)
                crop_path = zone_dir / crop_filename

                # 文件存在检查
                if crop_path.exists() and not overwrite:
                    skipped_existing += 1
                    position_name = ["左上", "右上", "左下", "右下"][pos_idx]
                    print(f"    ⏭️ {position_name}: 已存在，跳过: {crop_filename}")
                    zone_crop_results.append({
                        "position": position_name,
                        "position_index": pos_idx,
                        "component_name": component_name,
                        "is_placeholder": False,
                        "status": "skipped",
                        "output_file": str(crop_path.relative_to(OUTPUT_DIR)),
                    })
                    continue

                # 保存裁剪图片
                try:
                    cropped_images[pos_idx].save(str(crop_path), "PNG")
                    cropped_count += 1
                    position_name = ["左上", "右上", "左下", "右下"][pos_idx]
                    print(f"    ✅ {position_name}: {component_name} -> {crop_filename}")
                    zone_crop_results.append({
                        "position": position_name,
                        "position_index": pos_idx,
                        "component_name": component_name,
                        "is_placeholder": False,
                        "status": "cropped",
                        "output_file": str(crop_path.relative_to(OUTPUT_DIR)),
                        "source_quadrant": output_file,
                    })
                except Exception as e:
                    failed_count += 1
                    position_name = ["左上", "右上", "左下", "右下"][pos_idx]
                    print(f"    ❌ {position_name}: 裁剪失败 - {e}")
                    zone_crop_results.append({
                        "position": position_name,
                        "position_index": pos_idx,
                        "component_name": component_name,
                        "is_placeholder": False,
                        "status": "failed",
                        "error": str(e),
                    })

        all_crop_results.append({
            "zone_type": zone_type,
            "zone_name": zone_name,
            "zone_dir": str(zone_dir.relative_to(OUTPUT_DIR)),
            "crop_results": zone_crop_results,
        })

    # 保存 phase4 manifest
    output_manifest = {
        "source_module": str(module_output_dir),
        "total_zones": len(zone_results),
        "total_components": total_components,
        "cropped_count": cropped_count,
        "skipped_placeholder": skipped_placeholder,
        "skipped_existing": skipped_existing,
        "failed_count": failed_count,
        "zone_results": all_crop_results,
        "cropped_at": datetime.now().isoformat(),
    }

    output_manifest_path = output_subdir / "manifest.json"
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(output_manifest, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"📊 Phase 4 裁剪完成")
    print(f"{'='*60}")
    print(f"  模块: {module_name}")
    print(f"  总格子数: {total_components}")
    print(f"  成功裁剪: {cropped_count}")
    print(f"  舍弃占位: {skipped_placeholder}")
    print(f"  跳过已存在: {skipped_existing}")
    print(f"  失败: {failed_count}")
    print(f"  输出目录: {output_subdir}")

    return output_manifest


def main():
    """主入口"""
    # 确保目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 查找 Phase 3 的输出目录
    if not PHASE3_OUTPUT_DIR.exists():
        print(f"⚠️ Phase 3 输出目录不存在: {PHASE3_OUTPUT_DIR}")
        print(f"   请先运行 phase3_quadrant_generation.py")
        sys.exit(0)

    module_dirs = [d for d in PHASE3_OUTPUT_DIR.iterdir() if d.is_dir()]

    if not module_dirs:
        print(f"⚠️ Phase 3 输出目录为空: {PHASE3_OUTPUT_DIR}")
        print(f"   请先运行 phase3_quadrant_generation.py")
        sys.exit(0)

    print(f"📁 找到 {len(module_dirs)} 个模块目录")

    # 处理每个模块
    results = []
    for module_dir in module_dirs:
        result = process_module(module_dir)
        if result:
            results.append(result)

    # 汇总
    print(f"\n{'='*60}")
    print(f"🎉 Phase 4 四宫格裁剪全部完成")
    print(f"{'='*60}")
    print(f"  处理模块数: {len(results)}")
    print(f"  输出目录: {OUTPUT_DIR}")

    total_cropped = sum(r.get("cropped_count", 0) for r in results)
    total_placeholder = sum(r.get("skipped_placeholder", 0) for r in results)
    print(f"  总裁剪数: {total_cropped}")
    print(f"  总舍弃占位: {total_placeholder}")


if __name__ == "__main__":
    main()