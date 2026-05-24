"""
对比 Phase 2.5 旧版和增强版的结果
==================================
读取旧版和增强版的质检报告，对比分析：
1. 问题发现数量差异
2. VLM 识别的组件列表 vs Phase 2 的组件列表
3. 找出增强版新发现的问题

使用方式：
  python compare_phase25_results.py
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

# ─── 路径配置 ───
PHASE25_OUTPUT_DIR = Path(__file__).parent / "output" / "phase25"


def find_reports(module_name: str) -> tuple:
    """
    查找指定模块的旧版和增强版质检报告

    Returns:
        (old_report_path, enhanced_report_path) 或 (None, None)
    """
    module_dir = PHASE25_OUTPUT_DIR / module_name
    if not module_dir.exists():
        return None, None

    # 查找旧版报告
    old_reports = sorted(module_dir.glob("quality_report_*.json"))
    old_report = old_reports[-1] if old_reports else None

    # 查找增强版报告
    enhanced_reports = sorted(module_dir.glob("enhanced_report_*.json"))
    enhanced_report = enhanced_reports[-1] if enhanced_reports else None

    return old_report, enhanced_report


def load_report(path: Path) -> Optional[dict]:
    """加载质检报告"""
    if not path or not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ❌ 加载失败 {path}: {e}")
        return None


def compare_issues(old_issues: list, new_issues: list, issue_type: str) -> dict:
    """
    对比新旧问题列表

    Returns:
        {
            "old_count": int,
            "new_count": int,
            "new_issues": list,  # 新版新增的问题
            "fixed_issues": list,  # 旧版有但新版没有的问题
        }
    """
    old_count = len(old_issues)
    new_count = len(new_issues)

    # 提取问题的关键标识
    def get_issue_key(issue: dict, issue_type: str) -> str:
        if issue_type == "missing":
            return issue.get("description", "")[:50]
        elif issue_type == "merged":
            return issue.get("component_id", "")
        elif issue_type == "grouping":
            return issue.get("reusable_group", "")
        elif issue_type == "type_error":
            return issue.get("component_id", "")
        return str(issue)

    old_keys = {get_issue_key(i, issue_type): i for i in old_issues}
    new_keys = {get_issue_key(i, issue_type): i for i in new_issues}

    # 新增的问题
    new_issues_list = []
    for key, issue in new_keys.items():
        if key not in old_keys:
            new_issues_list.append(issue)

    # 修复的问题
    fixed_issues = []
    for key, issue in old_keys.items():
        if key not in new_keys:
            fixed_issues.append(issue)

    return {
        "old_count": old_count,
        "new_count": new_count,
        "new_issues": new_issues_list,
        "fixed_issues": fixed_issues,
    }


def compare_reports(old_report: dict, enhanced_report: dict):
    """对比新旧质检报告"""

    print(f"\n{'='*60}")
    print(f"📊 Phase 2.5 新旧版本对比分析")
    print(f"{'='*60}")

    # ── 基本信息 ──
    old_meta = old_report.get("_meta", {})
    enhanced_meta = enhanced_report.get("_meta", {})

    print(f"\n  📄 旧版报告: {old_meta.get('report_file', '?')}")
    print(f"  📄 增强版报告: {enhanced_meta.get('report_file', '?')}")

    # ── 评分对比 ──
    old_score = old_report.get("overall_score", 0)
    enhanced_score = enhanced_report.get("overall_score", 0)

    print(f"\n  📈 评分对比:")
    print(f"    旧版综合分: {old_score}")
    print(f"    增强版综合分: {enhanced_score}")
    print(f"    差异: {enhanced_score - old_score:+.1f}")

    # 各维度评分
    old_scores = old_report.get("scores", {})
    enhanced_scores = enhanced_report.get("scores", {})

    print(f"\n    各维度对比:")
    for dim in ["completeness", "fineness", "grouping", "type_accuracy"]:
        old = old_scores.get(dim, 0)
        new = enhanced_scores.get(dim, 0)
        diff = new - old
        print(f"      {dim:<18} 旧版={old}  增强版={new}  差异={diff:+d}")

    # ── 问题数量对比 ──
    print(f"\n  🔍 问题数量对比:")

    # missing_components
    missing_compare = compare_issues(
        old_report.get("missing_components", []),
        enhanced_report.get("missing_components", []),
        "missing"
    )
    print(f"\n    遗漏组件 (missing_components):")
    print(f"      旧版: {missing_compare['old_count']} 个")
    print(f"      增强版: {missing_compare['new_count']} 个")
    if missing_compare["new_issues"]:
        print(f"      新增: {len(missing_compare['new_issues'])} 个")
        for issue in missing_compare["new_issues"]:
            print(f"        - [{issue.get('severity', '?')}] {issue.get('description', '')[:60]}")
    if missing_compare["fixed_issues"]:
        print(f"      修复: {len(missing_compare['fixed_issues'])} 个")

    # merged_components
    merged_compare = compare_issues(
        old_report.get("merged_components", []),
        enhanced_report.get("merged_components", []),
        "merged"
    )
    print(f"\n    合并过度 (merged_components):")
    print(f"      旧版: {merged_compare['old_count']} 处")
    print(f"      增强版: {merged_compare['new_count']} 处")
    if merged_compare["new_issues"]:
        print(f"      新增: {len(merged_compare['new_issues'])} 处")
        for issue in merged_compare["new_issues"]:
            print(f"        - [{issue.get('severity', '?')}] {issue.get('component_id', '')}: {issue.get('description', '')[:50]}")
    if merged_compare["fixed_issues"]:
        print(f"      修复: {len(merged_compare['fixed_issues'])} 处")

    # grouping_issues
    grouping_compare = compare_issues(
        old_report.get("grouping_issues", []),
        enhanced_report.get("grouping_issues", []),
        "grouping"
    )
    print(f"\n    分组问题 (grouping_issues):")
    print(f"      旧版: {grouping_compare['old_count']} 个")
    print(f"      增强版: {grouping_compare['new_count']} 个")
    if grouping_compare["new_issues"]:
        print(f"      新增: {len(grouping_compare['new_issues'])} 个")
        for issue in grouping_compare["new_issues"]:
            print(f"        - [{issue.get('severity', '?')}] {issue.get('reusable_group', '')}: {issue.get('issue', '')[:50]}")
    if grouping_compare["fixed_issues"]:
        print(f"      修复: {len(grouping_compare['fixed_issues'])} 个")

    # type_errors
    type_compare = compare_issues(
        old_report.get("type_errors", []),
        enhanced_report.get("type_errors", []),
        "type_error"
    )
    print(f"\n    类型错误 (type_errors):")
    print(f"      旧版: {type_compare['old_count']} 个")
    print(f"      增强版: {type_compare['new_count']} 个")
    if type_compare["new_issues"]:
        print(f"      新增: {len(type_compare['new_issues'])} 个")
        for issue in type_compare["new_issues"]:
            print(f"        - [{issue.get('severity', '?')}] {issue.get('component_id', '')}: {issue.get('wrong_type', '')} → {issue.get('correct_type', '')}")
    if type_compare["fixed_issues"]:
        print(f"      修复: {len(type_compare['fixed_issues'])} 个")

    # ── VLM 组件列表分析 ──
    my_components = enhanced_report.get("my_component_list", [])
    if my_components:
        print(f"\n  📋 VLM 识别的组件列表分析:")
        print(f"    组件数量: {len(my_components)}")

        # 类型分布
        type_counts = {}
        for c in my_components:
            t = c.get("type", "?")
            type_counts[t] = type_counts.get(t, 0) + 1
        dist = ", ".join(f"{t}×{n}" for t, n in sorted(type_counts.items()))
        print(f"    类型分布: {dist}")

        # 与 Phase 2 的组件数量对比
        old_total = old_meta.get("total_components", 0)
        new_total = len(my_components)
        print(f"\n    与 Phase 2 对比:")
        print(f"      Phase 2 组件数: {old_total}")
        print(f"      VLM 识别组件数: {new_total}")
        print(f"      差异: {new_total - old_total:+d}")

    # ── 总结 ──
    total_new = (
        len(missing_compare["new_issues"])
        + len(merged_compare["new_issues"])
        + len(grouping_compare["new_issues"])
        + len(type_compare["new_issues"])
    )
    total_fixed = (
        len(missing_compare["fixed_issues"])
        + len(merged_compare["fixed_issues"])
        + len(grouping_compare["fixed_issues"])
        + len(type_compare["fixed_issues"])
    )

    print(f"\n  📝 总结:")
    print(f"    增强版新发现的问题: {total_new} 个")
    print(f"    增强版修复的问题: {total_fixed} 个")
    print(f"    净变化: {total_new - total_fixed:+d} 个问题")

    if total_new > 0:
        print(f"\n    ⚠️  增强版发现了更多问题，说明 VLM 独立理解图片后能发现更多遗漏")
    elif total_new == 0 and total_fixed > 0:
        print(f"\n    ✅ 增强版修复了一些问题，说明 VLM 的理解更准确")
    else:
        print(f"\n    ℹ️  两个版本结果相似")


def main():
    """主入口"""
    if not PHASE25_OUTPUT_DIR.exists():
        print(f"⚠️  Phase 2.5 输出目录不存在: {PHASE25_OUTPUT_DIR}")
        return

    # 列出所有模块
    module_dirs = sorted(
        [d for d in PHASE25_OUTPUT_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name
    )

    if not module_dirs:
        print(f"⚠️  Phase 2.5 输出目录为空")
        return

    print(f"📁 找到 {len(module_dirs)} 个模块：\n")
    for i, d in enumerate(module_dirs, 1):
        old_report, enhanced_report = find_reports(d.name)
        has_old = "✅" if old_report else "❌"
        has_enhanced = "✅" if enhanced_report else "❌"
        print(f"  [{i}] {d.name}  旧版={has_old}  增强版={has_enhanced}")

    # 用户选择
    try:
        choice = input(f"\n请选择要对比的模块编号 (默认=全部): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消")
        return

    if not choice or choice == "0":
        selected_dirs = module_dirs
    else:
        try:
            idx = int(choice)
            if idx < 1 or idx > len(module_dirs):
                print(f"❌ 无效编号: {idx}")
                return
            selected_dirs = [module_dirs[idx - 1]]
        except ValueError:
            print(f"❌ 无效输入: {choice}")
            return

    # 逐个对比
    for d in selected_dirs:
        old_report, enhanced_report = find_reports(d.name)

        if not old_report:
            print(f"\n⚠️  {d.name}: 旧版报告不存在，跳过")
            continue

        if not enhanced_report:
            print(f"\n⚠️  {d.name}: 增强版报告不存在，请先运行 phase25_quality_check_enhanced.py")
            continue

        old_data = load_report(old_report)
        enhanced_data = load_report(enhanced_report)

        if old_data and enhanced_data:
            compare_reports(old_data, enhanced_data)


if __name__ == "__main__":
    main()
