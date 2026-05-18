# Phase 2 VLM 质检方案

> **核心思路**：1 次 VLM 调用，看图 + 看清单，判断"拆没拆全、拆没拆细"。不过分就带着上次的问题再拆一次，最多 2-3 轮。

---

## 一、为什么 1 次就够了

质检 VLM 看到的是**完整的正交原图**，上面清清楚楚的每个窗户、每扇门、每条屋顶线。它只需要对比"图上有什么"和"清单里写了什么"，就能判断三个关键问题：

| 问题 | VLM 判断方式 |
|------|------------|
| 拆全了没？ | 图上肉眼可见的组件 → 清单里都有对应条目吗？ |
| 拆细了没？ | 窗户和花台有没有被当成一个组件？屋顶 bbox 是不是还包着墙面？ |
| 分组对不对？ | 清单里说是同一种可复用组件的，图上看起来真的一样吗？ |
| 类型标对了没？ | 标成 "decoration" 的东西真的是装饰而不是窗户吗？ |

这些都是**看图说话**，一个 VLM 一次调用完全能覆盖。

---

## 二、架构：LangGraph 中的质检 + 重试闭环

```
Phase2_Generate (VLM 识别组件)
       │
       ▼
CodeRule_Check (代码规则：bbox 不越界、字段不为空、数量达标)
       │
       ├── P0 > 0 ──→ Blocked（硬伤，直接阻断）
       │
       ▼ P0 = 0
VLM_Quality_Check (1 次 VLM 调用，看图 + 看清单)
       │
       ├── score >= 80 ──→ Pass ──→ Phase3
       ├── score 60-79 ──→ Conditional_Pass ──→ Phase3 + warn
       └── score < 60  ──→ Refine（带上次问题，重跑 Phase2 VLM）
              │
              ▼
         Phase2_Refine（prompt 中注入上次的问题清单）
              │
              ▼
         VLM_Quality_Check（再检）
              │
              ├── score >= 60 ──→ Pass / Conditional_Pass
              └── score < 60  ──→ 最后一次 Refine 或 Blocked
```

---

## 三、质检 VLM 的 Prompt（唯一一次调用）

```
你是建筑组件拆分质量审核专家。我会给你一张建筑立面图，以及一份由另一个AI生成的组件拆分清单。

请从以下维度审核这份清单的质量：

【审核维度】

1. 完整性（是否有遗漏）
   - 仔细查看图片中的每个建筑构件（每扇窗、每扇门、屋顶、墙面、装饰等）
   - 判断清单是否覆盖了所有肉眼可见的独立组件
   - 注意：外观相同的多个实例（如一排相同的窗户），清单应该每个都列出

2. 拆分精细度（是否有合并过度）
   - 窗户和窗台/花台是否被错误合并为一个组件？
   - 屋顶 bbox 是否还包着下面的墙面？
   - 相邻的窗户是否被合并成了一个？
   - 门和门框是否被当成一个整体（如果可以分开）？
   - facade 面板是否过大（覆盖了多个窗间墙区域）？

3. 分组合理性
   - 被归入同一个 reusable_group 的组件，看起来真的一样吗？
   - 有没有外观明显不同的组件被归入了同一组？
   - decoration 类型是否混入了各种不同的东西（尖顶、壁灯、植物应该分到不同组）？

4. 类型标注正确性
   - roof 标的是真屋顶吗？
   - window 标的是真窗户吗？
   - decoration 标的有没有其实是 window/shopfront/balcony？

【组件清单】
总组件数：{total_components}，分为 {total_groups} 个可复用组。

{component_list}
—— 格式：id | type | reusable_group | chinese_description

{group_list}
—— 每组格式：reusable_group | 实例数 | 代表描述

【输出 - 严格JSON，只输出这个对象】
{
  "scores": {
    "completeness": 85,
    "fineness": 70,
    "grouping": 80,
    "type_accuracy": 90
  },
  "overall_score": 81,
  "missing_components": [
    {
      "description": "图片右侧第2个窗户，圆形，绿色窗框",
      "approximate_position": "右侧中上部",
      "severity": "high"
    }
  ],
  "merged_components": [
    {
      "component_id": "comp_005",
      "description": "这个组件同时包含了窗户和下方的花台，应拆为2个独立组件",
      "severity": "high"
    },
    {
      "component_id": "comp_012",
      "description": "屋顶组件底部还包含了约30px的墙面",
      "severity": "medium"
    }
  ],
  "grouping_issues": [
    {
      "reusable_group": "decoration",
      "issue": "该组包含了尖顶、壁灯、植物3种完全不同的装饰，应各自成组",
      "severity": "high"
    }
  ],
  "type_errors": [
    {
      "component_id": "comp_030",
      "wrong_type": "decoration",
      "correct_type": "window",
      "reason": "这是一个带装饰框的拱形窗户，不是装饰品"
    }
  ],
  "blocking": false,
  "summary": "整体基本可用，有1个组件合并过度（窗户+花台），1个decoration组需要拆分"
}

【评分指南】
completeness（完整性）：
  - 100：无遗漏
  - 80-99：1-2小件遗漏
  - 60-79：1个主要组件遗漏
  - <60：大面积漏检

fineness（拆分精细度）：
  - 100：每个组件边界精确，无合并过度
  - 80-99：1个组件边界略显粗糙
  - 60-79：有1-2处合并过度（如屋顶包墙面）
  - <60：多处合并过度，拆分太粗

grouping（分组合理性）：
  - 100：所有分组视觉一致
  - 80-99：1组分类略粗但可接受
  - 60-79：有1组明显混入不同类型
  - <60：多组混乱

type_accuracy（类型正确性）：
  - 100：全部正确
  - 80-99：1-2个标注错误，不影响主要逻辑
  - <80：关键组件标错

overall_score = completeness×0.3 + fineness×0.3 + grouping×0.2 + type_accuracy×0.2

blocking = true 表示存在会导致 Phase3 生成失败的严重问题
```

---

## 四、评分与路由

| overall_score | 路由 | 动作 |
|:---:|------|------|
| >= 80 | `pass` | manifest 加盖 `vlm_checked: pass`，进 Phase3 |
| 60-79 | `conditional_pass` | 放行，issues 写入 manifest.warnings |
| < 60 | `refine` | 进入重试闭环（见第五节） |

---

## 五、重试策略：带问题上下文再拆一次

### 5.1 核心思路

第一次 Phase 2 的 VLM 看不到自己的问题。第二次调用时，把质检 VLM 发现的具体问题**注入到 Phase 2 VLM 的 prompt 中**，让它有针对性地修复。

不是重头再跑，而是在上次的基础上**增量修正**。

### 5.2 第二轮 Phase 2 的 prompt 扩展

在第一轮的 `MODULE_DETECTION_PROMPT` 末尾追加：

```
═══════════════════════════════════════════
【上一轮识别的问题 - 请修正】
质检发现了以下问题，请在本轮修正：

## 遗漏的组件
- 图片右侧第2个窗户，圆形，绿色窗框 ← 请补充识别

## 合并过度的组件
- comp_005（窗户+花台）：请拆为"窗户"和"花台"两个独立组件
- comp_012（屋顶包墙面）：请缩小屋顶 bbox，不要包含下方墙面

## 分组问题
- decoration 组混合了尖顶、壁灯、植物，请拆分为：
  decoration_spire / decoration_lamp / decoration_plant

## 类型错误
- comp_030 当前标为 decoration，应改为 window

【要求】
- 保留上一轮正确的识别结果，只修正上述问题
- 输出完整的新 components 数组（包含修正后的全部组件）
═══════════════════════════════════════════
```

### 5.3 混合策略（可选，效果更好）

如果担心 Phase 2 VLM "全重来"会丢失第一轮正确的部分，可以用**结果合并**方式：

1. 第一轮识别 → 质检发现问题 → 只对**问题区域**重新调用 Phase 2 VLM
2. 问题区域的识别结果替换掉原始结果中对应的条目
3. 合并后重新质检

这样既修复了问题，又保留了第一轮的正确输出。

### 5.4 重试次数与兜底

| 轮次 | 动作 | 
|:---:|------|
| 第 1 轮 | Phase 2 → 质检 |
| 第 2 轮 | 注入问题 → 重新识别 → 质检 |
| 第 3 轮 | 注入问题 → 重新识别 → 质检 |

3 轮后仍 < 60 → `blocked`，换模型或人工介入。

---

## 六、成本

| 项目 | 次数 / 建筑 |
|------|:---:|
| VLM 质检调用 | 1 次（正常） / 2-3 次（需要重试时） |
| 重试时额外 Phase 2 调用 | 0（正常） / 1-2 次（需要重试时） |
| 总 VLM 调用（正常路径） | Phase2: 1 + 质检: 1 = **2 次** |
| 总 VLM 调用（1 次重试路径） | Phase2: 2 + 质检: 2 = **4 次** |

单次质检输入：正交原图 base64（~1500 tokens）+ 组件清单文本（~2000 tokens）≈ **3500 tokens input + ~500 tokens output ≈ 4000 tokens**，约 **¥0.04**。

---

## 七、LangGraph Node 伪代码

```python
from langgraph.graph import StateGraph, END

class Phase2State(TypedDict):
    ortho_image_path: str
    ortho_b64: str
    manifest: dict
    code_errors: list
    quality_result: dict | None
    refine_round: int
    route: str


def build_phase2_graph() -> StateGraph:
    graph = StateGraph(Phase2State)

    graph.add_node("phase2_generate", phase2_generate_node)
    graph.add_node("code_rule_check", code_rule_check_node)
    graph.add_node("vlm_quality_check", vlm_quality_check_node)
    graph.add_node("phase2_refine", phase2_refine_node)

    graph.set_entry_point("phase2_generate")
    graph.add_edge("phase2_generate", "code_rule_check")

    graph.add_conditional_edges("code_rule_check", route_after_code, {
        "blocked": END,
        "ok": "vlm_quality_check"
    })

    graph.add_conditional_edges("vlm_quality_check", route_after_quality, {
        "pass": END,
        "conditional_pass": END,
        "refine": "phase2_refine",
        "blocked": END,
    })

    graph.add_edge("phase2_refine", "vlm_quality_check")

    return graph


def route_after_quality(state: Phase2State) -> str:
    result = state["quality_result"]
    score = result["overall_score"]

    if result.get("blocking"):
        return "blocked"
    if score >= 80:
        return "pass"
    if score >= 60:
        return "conditional_pass"

    state["refine_round"] += 1
    if state["refine_round"] > 3:
        return "blocked"  # 3轮后放弃
    return "refine"
```

---

## 八、总结

| | 旧方案（v2.0） | 最终方案（v3.0） |
|---|---|---|
| VLM 调用次数 | 4-6 次（Check A+B + N zones） | **1 次** |
| 检查内容 | 分散在多轮调用 | **一张图 + 一份清单，一次性全部判断** |
| 重试机制 | 无 | 带问题反馈，最多 3 轮 |
| 单建筑成本 | ~¥0.09 | **~¥0.04** |
| 适用范围 | 过度设计 | 实际可用 |

**核心设计哲学**：质检 VLM 做的事就是"看图说话"——图上有什么，清单写了什么，对得上就对，对不上就指出哪里不对。不需要检查裁剪图，不需要检查 bbox 精度，不需要逐区验证。一张图、一份清单、一个评分、一套问题反馈。

---

*文档版本: v3.0*  
*变更：从多轮多图 VLM 调用精简为单次调用 + 重试闭环，与 LangGraph 工作流深度集成*
