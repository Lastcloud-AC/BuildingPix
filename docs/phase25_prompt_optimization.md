# Phase 2.5 质检提示词优化方案

> **目标**：解决两次质检（Gemini-3-Pro vs Gemini-3.1-Flash-Lite）评分差异过大（81 vs 73，差8分）的问题
> **根因**：提示词信息不足 + 评分标准模糊，导致不同模型"各自发挥"
> **原则**：通过注入更多上下文和量化锚点，使不同模型的质检结果趋同

---

## 一、问题诊断：两次质检差异根因

| 差异维度 | Gemini-3-Pro (81分) | Gemini-3.1-Flash (73分) | 差异根因 |
|---------|--------------------|-----------------------|---------|
| 完整性 | 85（遗漏：右侧小建筑） | 75（遗漏：Finials + Corbels） | 无楼层逐层验证引导 |
| 精细度 | 75（门+门廊合并） | 65（遮阳篷+橱窗合并） | 组件摘要无bbox/尺寸，VLM无法量化判断 |
| 分组 | 70（两门不同组） | 70（decoration混组） | 提示词不知道Phase2已用子类型 |
| 类型 | 95（无错误） | 85（facade→base_wall） | 无类型混淆判断参照 |

### Flash-Lite 的 decoration 分组误判是关键案例

manifest 中 Phase 2 已经使用了子类型分组：
- `decoration_spire_iron`（尖顶）
- `decoration_railing_iron`（栏杆）
- `decoration_lamp_black_metal`（壁灯）
- `decoration_plant_ivy`（攀爬植物）
- `decoration_plant_box`（花箱）

但旧提示词说"decoration 类型是否混入了各种不同的东西（尖顶、壁灯、植物应该分别成组）"，Flash-Lite 据此误判为问题。Pro 模型没报这个问题，说明它理解了 reusable_group 的子类型含义。

---

## 二、优化措施

### P0 - 关键改进（直接减少评分波动）

#### 1. 组件摘要注入 bbox + 像素尺寸

**修改位置**：`build_component_summary()`

| 修改前 | 修改后 |
|-------|--------|
| `comp_024 \| type=door \| group=door_pink_arch \| 粉色的尖拱形木门...` | `comp_024 \| door \| door_pink_arch \| [462,762,542,912] \| 80×150px \| 粉色的尖拱形木门...` |

**效果**：VLM 可以通过像素尺寸判断合并问题。例如：
- comp_024 的 80×150px 是正常门尺寸 → 不合并
- 如果某个"门"是 80×300px → 明显合并了门+门廊

#### 2. 注入建筑上下文信息

**新增函数**：`build_building_context()`

注入内容：
- 图片尺寸（1024×1024px）
- 识别模型（qwen3.6-plus）
- 类型分布（facade×4, window×10, door×2...）
- 楼层区域划分（楼顶13组件、上层10组件、底层19组件）
- Phase 1 建筑语义（风格、楼层数、屋顶类型等）

**效果**：VLM 有了坐标参考系和建筑结构概览，可以"按楼层逐层核"。

#### 3. 量化评分标准

| 维度 | 修改前 | 修改后 |
|-----|--------|--------|
| 完整性 80-89 | "有小件遗漏" | "有小件遗漏但主要组件（窗/门/屋顶/主要墙面）全部覆盖" |
| 精细度 80-89 | "1个略粗" | "1个组件存在合并过度（如门+门廊），但其余正常" |
| 分组 80-89 | "1组略粗但可接受" | "1组存在明显不一致（如两扇外观不同的门被归同组）" |

每个分数段现在有明确的行为描述，减少不同模型的解读差异。

### P1 - 重要增强

#### 4. 修正 decoration 分组检查逻辑

**修改前**：
> decoration 类型是否混入了各种不同的东西（尖顶、壁灯、植物应该分别成组）？

**修改后**：
> Phase 2 已对 decoration 类型做了子类型分组（如 decoration_spire_iron, decoration_lamp_black_metal 等），请检查这些子分组是否正确，而不是笼统地认为"decoration 混在一起"

**效果**：消除 Flash-Lite 类模型的误报。

#### 5. 新增合并判断的量化依据

**修改后提示词**：
> 如果某个组件的 bbox 像素面积明显大于同类组件（如一个"门"的bbox宽度是其他门的2倍以上），很可能存在合并

这比"门和门框是否被当成一个整体"更可操作。

#### 6. 新增计数验证引导

> 数量验证：清点图片中窗户/门/阳台等主要组件的数量，与清单数量比对

### P2 - 输出格式增强

#### 7. merged_components 新增 evidence 字段

要求 VLM 在报告合并问题时，必须给出 bbox 证据：
```json
{
  "component_id": "comp_005",
  "description": "窗户+花台合并",
  "evidence": "bbox [200,400,300,600] 尺寸 100×200px，远大于同类窗户的 80×100px",
  "severity": "high"
}
```

#### 8. missing_components 新增 approximate_position 字段

便于后续定位和修复。

#### 9. grouping_issues 必须指定具体的 reusable_group

旧提示词的示例用的是笼统的 `"decoration"`，新示例改为具体的 `"decoration_lamp_black_metal"`。

---

## 三、修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `phase25_quality_check.py` | 1. 重写 `QUALITY_CHECK_PROMPT`（量化评分+类型体系+合并判断依据）<br>2. 重写 `build_component_summary()`（注入bbox+像素尺寸+楼层区域+图片尺寸）<br>3. 新增 `build_building_context()`（Phase1语义+类型分布+楼层区域）<br>4. 修改 `call_quality_check_vlm()` 签名，增加 building_context 参数<br>5. 修改 `check_module()` 流程，增加上下文构造步骤<br>6. 修改 `print_report()` 支持新字段 |

---

## 四、预期效果

| 指标 | 优化前 | 优化后预期 |
|------|--------|-----------|
| 跨模型评分波动 | ±8分（81 vs 73） | ±3分以内 |
| decoration 分组误报 | Flash 报 "decoration混组" | 清零（提示词已明确子类型体系） |
| 合并判断一致性 | Pro/Flash 发现不同合并点 | 趋同（有bbox像素证据支撑） |
| 评分可解释性 | "1-2小件遗漏" 各自解读 | 每个分数段有明确行为定义 |

---

*文档版本: v1.0*
*日期: 2026-05-17*
*变更: Phase 2.5 质检提示词系统化优化，解决跨模型评分不一致问题*
