"""
State definition for the Personal Situation Judgment Test (PSJT) system.
"""
from typing import TypedDict, List, Dict, Any


class PSJTState(TypedDict, total=False):
    """State type for the PSJT generation workflow."""
    request_id: str
    trait_name: str                     # 目标特质，例如：尽责性-条理性
    bfi_item: str                       # 原始自陈题干文本
    population_profile: str             # 目标群体画像（如："中国大学生"、"基层管理者"）
    language: str                       # 输出语言（"zh"/"en"）

    trait_analysis: dict[str, Any]      # 特质解析结果（认知/情感/行为 -> 线索/依据/反馈 映射）
    situation_outline: dict[str, Any]   # 情境大纲（原型、场景、任务、冲突、线索清单...）
    scenario_text: str                  # 场景叙事（最终题干）
    options: list[dict[str, Any]]       # 反应选项（A-E：行为+理由+特质水平）

    quality: dict[str, Any]             # 质量评估指标与说明
    revise_notes: str                   # 需要修订的反馈（供下一轮）
    iter: int                           # 迭代计数