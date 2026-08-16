"""不依赖外部模型的评估诊断规则。

当 LLM 超时、网络失败或返回内容不符合结构约束时，系统使用本模块生成
明确标记为 fallback 的诊断，保证实验流程仍可继续且不会伪装成 LLM 结果。
"""

from __future__ import annotations

import json

from toolwear_agent.schemas.diagnosis import (
    DiagnosisAdvice,
    DiagnosisFinding,
    DiagnosisRecommendation,
    EvaluationFacts,
)


def build_rule_based_advice(facts: EvaluationFacts) -> DiagnosisAdvice:
    """根据同一份验证事实生成稳定、可审计的降级建议。"""

    findings: list[DiagnosisFinding] = []
    gap = facts.generalization_gap_macro_f1
    if gap is not None and gap > 0.15:
        findings.append(
            DiagnosisFinding(
                finding_id="generalization-gap",
                severity="warning",
                category="overfitting",
                title="训练与验证差距偏大",
                detail=f"Train 与 validation Macro-F1 相差 {gap:.4f}。",
                evidence=(
                    f"train={facts.train_macro_f1:.4f}, "
                    f"validation={facts.validation_macro_f1:.4f}"
                ),
            )
        )

    findings.append(
        DiagnosisFinding(
            finding_id="weakest-class",
            severity=(
                "critical"
                if facts.weakest_class_f1 < 0.5
                else "warning"
                if facts.weakest_class_f1 < 0.8
                else "info"
            ),
            category="class_performance",
            title=f"最弱阶段：{facts.weakest_class}",
            detail=(
                f"该阶段 F1={facts.weakest_class_f1:.4f}，"
                f"Recall={facts.weakest_class_recall:.4f}。"
            ),
            evidence=f"basis=validation, class={facts.weakest_class}",
        )
    )

    if facts.support_imbalance_ratio >= 3:
        findings.append(
            DiagnosisFinding(
                finding_id="class-imbalance",
                severity="warning",
                category="class_balance",
                title="阶段样本量不均衡",
                detail=(
                    "最大与最小阶段 support 比为 "
                    f"{facts.support_imbalance_ratio:.2f}。"
                ),
                evidence=json.dumps(
                    facts.class_support,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
        )

    if facts.top_confusions:
        item = facts.top_confusions[0]
        findings.append(
            DiagnosisFinding(
                finding_id="top-confusion",
                severity="warning" if item.row_rate >= 0.1 else "info",
                category="confusion",
                title="主要类别混淆",
                detail=(
                    f"{item.actual_label} 被预测为 "
                    f"{item.predicted_label} 共 {item.count} 次。"
                ),
                evidence=f"validation row rate={item.row_rate:.4f}",
            )
        )

    recommendations = [
        DiagnosisRecommendation(
            recommendation_id="focus-weak-class",
            action_type="inspect_data",
            target=facts.weakest_class,
            suggestion="先核查该阶段样本覆盖、相邻阶段边界和主要混淆方向。",
            rationale="优先处理 validation 中最明确的薄弱点。",
            priority="high",
        )
    ]
    if gap is not None and gap > 0.15:
        recommendations.append(
            DiagnosisRecommendation(
                recommendation_id="reduce-overfit",
                action_type="adjust_parameter",
                target="current_pipeline",
                suggestion=(
                    "在新 revision 中降低模型复杂度或增强正则化，"
                    "再用同一 validation 比较。"
                ),
                rationale="当前训练分数明显高于 validation。",
                priority="high",
            )
        )

    if facts.completed_mini_runs >= facts.max_mini_runs:
        action = "stop"
    elif (
        facts.validation_macro_f1 >= 0.85
        and facts.weakest_class_f1 >= 0.75
        and (gap is None or gap <= 0.15)
    ):
        action = "approve_full"
        recommendations.append(
            DiagnosisRecommendation(
                recommendation_id="approve-full",
                action_type="approve_full",
                target="current_pipeline",
                suggestion="保留当前 Pipeline，提交人工审批后进入完整训练。",
                rationale="validation 总体与最弱类别均达到首轮通过条件。",
                priority="medium",
            )
        )
    elif facts.validation_macro_f1 < 0.55:
        action = "change_pipeline"
    else:
        action = "adjust_parameters"

    risk = (
        "high"
        if facts.validation_macro_f1 < 0.6 or facts.weakest_class_f1 < 0.5
        else "medium"
        if action != "approve_full"
        else "low"
    )
    return DiagnosisAdvice(
        overall_conclusion=(
            f"Validation Macro-F1={facts.validation_macro_f1:.4f}，最弱阶段为"
            f"{facts.weakest_class}；建议动作：{action}。"
        ),
        risk_level=risk,
        findings=tuple(findings),
        recommendations=tuple(recommendations),
        recommended_action=action,
    )


__all__ = ["build_rule_based_advice"]
