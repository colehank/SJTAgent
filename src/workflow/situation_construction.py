"""
Situation construction expert module for generating scenario outlines and narratives.
"""
import json
import textwrap
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

from .state import PSJTState
from .llm_utils import llm, json_guard


SITUATION_SYSTEM = SystemMessage(content=textwrap.dedent(
    """
    你是一名"情境建构专家"。任务：
    A) 依据"特质解析专家"的 mapping & 词表，构建总体情境大纲（JSON）：
       - prototype: 情境原型（如：团队协作/时间压力/道德困境/服务冲突…）
       - setting: 背景（行业/地点/角色/重要他人）
       - task: 需要完成的目标或交付物
       - trigger: 触发事件/冲突
       - cues: 需要出现在叙事中的 observable_cues（≥3 条，来自上游）
       - constraints: 资源/时间/制度等约束
       - success_criteria: 成功判据
    B) 生成可直接作为 SJT 题干的"情境叙事"一段文本（scenario_text，200-300 字）。
    风格：贴近目标群体经验；线索密度充足；避免行业黑话；中文输出。
    """
))

SITUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("human", textwrap.dedent(
        """
        目标群体：{population_profile}
        目标特质：{trait_name}
        上游特质映射：{trait_mapping}
        原始自陈题：{bfi_item}
        输出 JSON：{{"outline": {{...}}, "scenario_text": "..."}}
        """
    ))
])


def situation_construction_node(state: PSJTState) -> PSJTState:
    """
    Situation construction expert node that generates scenario outlines and narratives.
    
    Args:
        state: Current PSJT state with trait analysis
        
    Returns:
        Updated state with situation outline and scenario text
    """
    mapping = state["trait_analysis"].get("mapping", {})
    
    messages = [SITUATION_SYSTEM] + SITUATION_PROMPT.format_messages(
        population_profile=state["population_profile"],
        trait_name=state["trait_name"],
        trait_mapping=json.dumps(mapping, ensure_ascii=False),
        bfi_item=state["bfi_item"],
    )
    
    output = llm.invoke(messages)
    data = json_guard(output.content)
    
    state["situation_outline"] = data.get("outline", {})
    state["scenario_text"] = data.get("scenario_text", "")
    
    return state