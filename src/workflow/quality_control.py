"""
Quality control module for evaluating and revising generated content.
"""
import json
import textwrap
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

from .state import PSJTState
from .llm_utils import llm, json_guard


def _distinct_ratio(options: List[Dict[str, Any]]) -> float:
    """Calculate the distinctiveness ratio of behavioral options."""
    if not options:
        return 0.0
    
    texts = [
        json.dumps({"a": o.get("action"), "r": o.get("rationale")}, ensure_ascii=False)
        for o in options
    ]
    unique = len(set(texts))
    return unique / max(1, len(texts))


def _level_coverage(options: List[Dict[str, Any]]) -> float:
    """Calculate trait level coverage of behavioral options."""
    levels = {o.get("trait_level", "?").lower() for o in options}
    want = {"high", "mid", "low"}
    return len(levels & want) / len(want)


def quality_check_node(state: PSJTState) -> PSJTState:
    """
    Quality check node that evaluates generated content and determines if revision is needed.
    
    Args:
        state: Current PSJT state
        
    Returns:
        Updated state with quality metrics and revision notes
    """
    options = state.get("options", [])
    cues = set(state.get("situation_outline", {}).get("cues", []))
    
    # Calculate quality metrics
    distinct = _distinct_ratio(options)
    coverage = _level_coverage(options)
    
    cue_hit = 0.0
    for o in options:
        joined = json.dumps(o, ensure_ascii=False)
        cue_hit += sum(1 for c in cues if c and (c in joined))
    cue_hit = cue_hit / max(1, len(options))

    ok = (distinct >= 0.9) and (coverage >= 0.67) and (cue_hit >= 1.0)
    
    state["quality"] = {
        "distinct_ratio": round(distinct, 3),
        "coverage": round(coverage, 3),
        "avg_cue_mentions": round(cue_hit, 3),
        "pass": bool(ok),
    }

    if ok:
        state["revise_notes"] = ""
        return state

    # Generate revision notes
    lacks = []
    if distinct < 0.9:
        lacks.append("选项语义重复/差异不足")
    if coverage < 0.67:
        lacks.append("未覆盖 high/mid/low 全部水平")
    if cue_hit < 1.0:
        lacks.append("选项与情境线索耦合不足")
    
    state["revise_notes"] = "；".join(lacks) or "请加强区分度、覆盖度与情境耦合。"
    return state


def should_revise(state: PSJTState) -> str:
    """
    Determine if revision is needed based on quality assessment.
    
    Args:
        state: Current PSJT state
        
    Returns:
        Either "revise" or "stop"
    """
    if state.get("iter", 0) >= 2:  # Max 2 revisions to avoid infinite loops
        return "stop"
    if not state.get("quality", {}).get("pass", False):
        return "revise"
    return "stop"


REVISE_SYSTEM = SystemMessage(content=textwrap.dedent(
    """
    你是"题目修订助手"。根据质量反馈，重写【情境叙事（若需要）】与【行为选项】：
    - 保持原 outline 的原型/约束不变；
    - 优先提升：选项差异度、水平覆盖度、与情境线索的一致性；
    - 输出 JSON：{"scenario_text": "...（可选）", "options": [{...} * 5]}
    """
))

REVISE_PROMPT = ChatPromptTemplate.from_messages([
    ("human", textwrap.dedent(
        """
        质量问题：{issues}
        题干：{scenario_text}
        情境大纲：{outline}
        现有选项：{options}
        请修订并返回 JSON。
        """
    ))
])


def revise_node(state: PSJTState) -> PSJTState:
    """
    Revision node that improves content based on quality feedback.
    
    Args:
        state: Current PSJT state with quality issues
        
    Returns:
        Updated state with revised content
    """
    messages = [REVISE_SYSTEM] + REVISE_PROMPT.format_messages(
        issues=state.get("revise_notes", ""),
        scenario_text=state.get("scenario_text", ""),
        outline=json.dumps(state.get("situation_outline", {}), ensure_ascii=False),
        options=json.dumps(state.get("options", []), ensure_ascii=False),
    )
    
    output = llm.invoke(messages)
    data = json_guard(output.content)
    
    if data.get("scenario_text"):
        state["scenario_text"] = data["scenario_text"]
    if data.get("options"):
        state["options"] = data["options"]
    
    state["iter"] = state.get("iter", 0) + 1
    return state