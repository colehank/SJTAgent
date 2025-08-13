"""
Trait analysis expert module for decomposing personality traits.
"""
import textwrap
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

from .state import PSJTState
from .llm_utils import llm, json_guard


TRAIT_ANALYSIS_SYSTEM = SystemMessage(content=textwrap.dedent(
    """
    你是一名"特质解析专家"。任务：
    1) 将目标人格特质拆解为【认知成分】【情感成分】【行为倾向】；
    2) 把这三类成分映射为情境可用元素：【可观测线索】【行为依据】【结果反馈】；
    3) 输出 JSON，字段：trait, components, mapping, evidence_glossary。
       - components: {cognition: [...], affect: [...], behavior: [...]}（每类 3-6 条）
       - mapping: {observable_cues:[...], action_bases:[...], outcome_feedback:[...]}（各 3-6 条）
       - evidence_glossary: 针对该特质的典型高/低表现词表（各 8-15 个词/短语）
    要求：紧扣心理学理论内涵，避免空泛；用简短要点语句；中文输出。
    """
))

TRAIT_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("human", textwrap.dedent(
        """
        目标特质：{trait_name}
        参考自陈题：{bfi_item}
        目标群体画像：{population_profile}
        请完成上述 1-3 的解析与映射，严格以 JSON 返回。
        """
    ))
])


def trait_analysis_node(state: PSJTState) -> PSJTState:
    """
    Trait analysis expert node that decomposes personality traits into
    cognitive, affective, and behavioral components.
    
    Args:
        state: Current PSJT state
        
    Returns:
        Updated state with trait analysis results
    """
    messages = [TRAIT_ANALYSIS_SYSTEM] + TRAIT_ANALYSIS_PROMPT.format_messages(
        trait_name=state["trait_name"],
        bfi_item=state["bfi_item"],
        population_profile=state["population_profile"],
    )
    
    output = llm.invoke(messages)
    data = json_guard(output.content)
    state["trait_analysis"] = data
    
    return state