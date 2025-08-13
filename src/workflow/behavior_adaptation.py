"""
Behavior adaptation expert module for creating behavioral response options.
"""
import json
import textwrap
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

from .state import PSJTState
from .llm_utils import llm, json_guard


BEHAVIOR_SYSTEM = SystemMessage(content=textwrap.dedent(
    """
    你是一名"行为适配专家"。任务：基于给定情境，为题干设计 5 个可区分的行为选项 A-E：
     - 每个选项字段：{"label":"A","action":"…","rationale":"…","trait_level":"high|mid|low"}
     - 覆盖从高到低特质水平的"行为连续体"（至少含 high/mid/low），避免语义重叠；
     - 贴合 outline.cues 与约束，不要"超能力"或信息缺失下的不合理操作；
     - 避免社会赞许性泄露（不要明显"正确答案"措辞）。
    输出 JSON：{"options": [{...} * 5]}
    """
))

BEHAVIOR_PROMPT = ChatPromptTemplate.from_messages([
    ("human", textwrap.dedent(
        """
        题干：{scenario_text}
        情境大纲：{outline}
        特质词表（可帮助区分行为强度）：{glossary}
        请输出 5 个行为选项（JSON）。
        """
    ))
])


def behavior_adaptation_node(state: PSJTState) -> PSJTState:
    """
    Behavior adaptation expert node that creates differentiated behavioral options
    covering high to low trait levels.
    
    Args:
        state: Current PSJT state with scenario and trait analysis
        
    Returns:
        Updated state with behavioral options
    """
    outline = state.get("situation_outline", {})
    glossary = state["trait_analysis"].get("evidence_glossary", {})
    
    messages = [BEHAVIOR_SYSTEM] + BEHAVIOR_PROMPT.format_messages(
        scenario_text=state.get("scenario_text", ""),
        outline=json.dumps(outline, ensure_ascii=False),
        glossary=json.dumps(glossary, ensure_ascii=False),
    )
    
    output = llm.invoke(messages)
    data = json_guard(output.content)
    state["options"] = data.get("options", [])
    
    return state