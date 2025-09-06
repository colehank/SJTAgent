# %%
from __future__ import annotations
import os, json, re, uuid, textwrap
from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass

# LangGraph / LangChain
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
# %%
# -----------------------------
# 1) 状态定义
# -----------------------------
class PSJTState(TypedDict, total=False):
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


# -----------------------------
# 2) LLM 工具
tdef = "gpt-4o-mini"  # 你可以换成任意兼容 ChatCompletion 的模型
llm = ChatOpenAI(model=tdef, temperature=0.4)


def _json_guard(s: str) -> Any:
    """从 LLM 文本中抓取第一个 JSON 对象，避免异常格式导致崩溃。"""
    try:
        return json.loads(s)
    except Exception:
        # 宽松抓取 {...} 第一段
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        raise ValueError("LLM 未返回可解析 JSON。原始输出：\n" + s)


# -----------------------------
# 3) 三个专家模块（节点）
# -----------------------------

# 3.1 特质解析专家：把抽象特质 -> 认知/情感/行为成分；并映射到情境三要素
TRAIT_ANALYSIS_SYS = SystemMessage(content=textwrap.dedent(
    """
    你是一名“特质解析专家”。任务：
    1) 将目标人格特质拆解为【认知成分】【情感成分】【行为倾向】；
    2) 把这三类成分映射为情境可用元素：【可观测线索】【行为依据】【结果反馈】；
    3) 输出 JSON，字段：trait, components, mapping, evidence_glossary。
       - components: {cognition: [...], affect: [...], behavior: [...]}（每类 3-6 条）
       - mapping: {observable_cues:[...], action_bases:[...], outcome_feedback:[...]}（各 3-6 条）
       - evidence_glossary: 针对该特质的典型高/低表现词表（各 8-15 个词/短语）
    要求：紧扣心理学理论内涵，避免空泛；用简短要点语句；中文输出。
    """
))

TRAIT_ANALYSIS_HUMAN = ChatPromptTemplate.from_messages([
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
    msgs = [TRAIT_ANALYSIS_SYS] + TRAIT_ANALYSIS_HUMAN.format_messages(
        trait_name=state["trait_name"],
        bfi_item=state["bfi_item"],
        population_profile=state["population_profile"],
    )
    out = llm.invoke(msgs)
    data = _json_guard(out.content)
    state["trait_analysis"] = data
    return state


# 3.2 情境建构专家：生成“总体情境大纲”+“具象题干叙事”，融入特质映射线索
SITUATION_SYS = SystemMessage(content=textwrap.dedent(
    """
    你是一名“情境建构专家”。任务：
    A) 依据“特质解析专家”的 mapping & 词表，构建总体情境大纲（JSON）：
       - prototype: 情境原型（如：团队协作/时间压力/道德困境/服务冲突…）
       - setting: 背景（行业/地点/角色/重要他人）
       - task: 需要完成的目标或交付物
       - trigger: 触发事件/冲突
       - cues: 需要出现在叙事中的 observable_cues（≥3 条，来自上游）
       - constraints: 资源/时间/制度等约束
       - success_criteria: 成功判据
    B) 生成可直接作为 SJT 题干的“情境叙事”一段文本（scenario_text，200-300 字）。
    风格：贴近目标群体经验；线索密度充足；避免行业黑话；中文输出。
    """
))

SITUATION_HUMAN = ChatPromptTemplate.from_messages([
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
    mapping = state["trait_analysis"].get("mapping", {})
    msgs = [SITUATION_SYS] + SITUATION_HUMAN.format_messages(
        population_profile=state["population_profile"],
        trait_name=state["trait_name"],
        trait_mapping=json.dumps(mapping, ensure_ascii=False),
        bfi_item=state["bfi_item"],
    )
    out = llm.invoke(msgs)
    data = _json_guard(out.content)
    state["situation_outline"] = data.get("outline", {})
    state["scenario_text"] = data.get("scenario_text", "")
    return state


# 3.3 行为适配专家：在既定场景中嵌入 A-E 行为选项（覆盖高→低特质连续体）
BEHAVIOR_SYS = SystemMessage(content=textwrap.dedent(
    """
    你是一名“行为适配专家”。任务：基于给定情境，为题干设计 5 个可区分的行为选项 A-E：
     - 每个选项字段：{"label":"A","action":"…","rationale":"…","trait_level":"high|mid|low"}
     - 覆盖从高到低特质水平的“行为连续体”（至少含 high/mid/low），避免语义重叠；
     - 贴合 outline.cues 与约束，不要“超能力”或信息缺失下的不合理操作；
     - 避免社会赞许性泄露（不要明显“正确答案”措辞）。
    输出 JSON：{"options": [{...} * 5]}
    """
))

BEHAVIOR_HUMAN = ChatPromptTemplate.from_messages([
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
    outline = state.get("situation_outline", {})
    glossary = state["trait_analysis"].get("evidence_glossary", {})
    msgs = [BEHAVIOR_SYS] + BEHAVIOR_HUMAN.format_messages(
        scenario_text=state.get("scenario_text", ""),
        outline=json.dumps(outline, ensure_ascii=False),
        glossary=json.dumps(glossary, ensure_ascii=False),
    )
    out = llm.invoke(msgs)
    data = _json_guard(out.content)
    state["options"] = data.get("options", [])
    return state


# -----------------------------
# 4) 质量评估 & 迭代修订
# -----------------------------

def _distinct_ratio(options: list[dict[str, Any]]) -> float:
    if not options: return 0.0
    texts = [json.dumps({"a":o.get("action"),"r":o.get("rationale")}, ensure_ascii=False) for o in options]
    unique = len(set(texts))
    return unique / max(1, len(texts))


def _level_coverage(options: list[dict[str, Any]]) -> float:
    levels = {o.get("trait_level","?").lower() for o in options}
    want = {"high","mid","low"}
    return len(levels & want) / len(want)


def quality_check_node(state: PSJTState) -> PSJTState:
    options = state.get("options", [])
    cues = set(state.get("situation_outline",{}).get("cues", []))
    # 规则化指标
    distinct = _distinct_ratio(options)
    coverage = _level_coverage(options)
    cue_hit = 0.0
    for o in options:
        joined = json.dumps(o, ensure_ascii=False)
        cue_hit += sum(1 for c in cues if c and (c in joined))
    cue_hit = cue_hit / max(1, len(options))

    ok = (distinct >= 0.9) and (coverage >= 0.67) and (cue_hit >= 1.0)
    state["quality"] = {
        "distinct_ratio": round(distinct,3),
        "coverage": round(coverage,3),
        "avg_cue_mentions": round(cue_hit,3),
        "pass": bool(ok),
    }

    if ok:
        state["revise_notes"] = ""
        return state

    # 生成“修订提示”，供下轮行为适配/情境建构参考
    lacks = []
    if distinct < 0.9: lacks.append("选项语义重复/差异不足")
    if coverage < 0.67: lacks.append("未覆盖 high/mid/low 全部水平")
    if cue_hit < 1.0: lacks.append("选项与情境线索耦合不足")
    state["revise_notes"] = "；".join(lacks) or "请加强区分度、覆盖度与情境耦合。"
    return state


# 依据评估决定是否迭代（将决定图的边走向）
def should_revise(state: PSJTState) -> str:
    if state.get("iter", 0) >= 2:  # 最多两轮修订，避免死循环
        return "stop"
    if not state.get("quality",{}).get("pass", False):
        return "revise"
    return "stop"


# 修订：将反馈注入“行为适配专家”，必要时也会要求“微调情境叙事”
REVISE_SYS = SystemMessage(content=textwrap.dedent(
    """
    你是“题目修订助手”。根据质量反馈，重写【情境叙事（若需要）】与【行为选项】：
    - 保持原 outline 的原型/约束不变；
    - 优先提升：选项差异度、水平覆盖度、与情境线索的一致性；
    - 输出 JSON：{"scenario_text": "...（可选）", "options": [{...} * 5]}
    """
))

REVISE_HUMAN = ChatPromptTemplate.from_messages([
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
    msgs = [REVISE_SYS] + REVISE_HUMAN.format_messages(
        issues=state.get("revise_notes",""),
        scenario_text=state.get("scenario_text",""),
        outline=json.dumps(state.get("situation_outline",{}), ensure_ascii=False),
        options=json.dumps(state.get("options",[]), ensure_ascii=False),
    )
    out = llm.invoke(msgs)
    data = _json_guard(out.content)
    if data.get("scenario_text"):
        state["scenario_text"] = data["scenario_text"]
    if data.get("options"):
        state["options"] = data["options"]
    state["iter"] = state.get("iter", 0) + 1
    return state

def pack_item(state: PSJTState) -> dict[str, Any]:
    return {
        "id": state.get("request_id"),
        "trait": state.get("trait_name"),
        "population_profile": state.get("population_profile"),
        "bfi_item": state.get("bfi_item"),
        "scenario_text": state.get("scenario_text"),
        "options": state.get("options", []),
        "quality": state.get("quality", {}),
        "outline": state.get("situation_outline", {}),
        "trait_analysis": state.get("trait_analysis", {}),
        "meta": {
            "tool": "langgraph-psjt",
            "version": "0.2.0",
            "model": tdef,
            "iterations": state.get("iter", 0),
        },
    }

def build_graph() -> StateGraph:
    g = StateGraph(PSJTState)
    g.add_node("trait_analysis", trait_analysis_node)
    g.add_node("situation_construction", situation_construction_node)
    g.add_node("behavior_adaptation", behavior_adaptation_node)
    g.add_node("quality_check", quality_check_node)
    g.add_node("revise", revise_node)

    g.set_entry_point("trait_analysis")
    g.add_edge("trait_analysis", "situation_construction")
    g.add_edge("situation_construction", "behavior_adaptation")
    g.add_edge("behavior_adaptation", "quality_check")

    # 条件分支：是否需要修订
    g.add_conditional_edges(
        "quality_check",
        should_revise,
        {
            "revise": "revise",
            "stop": END,
        },
    )
    # 修订后回到行为适配重新评估
    g.add_edge("revise", "behavior_adaptation")
    return g

# %%
# -----------------------------
# 7) 演示入口
# -----------------------------

DEMO_BFI = {
    "尽责性-条理性": "我做事有计划并保持条理。",
    "外向性-社交主动": "我愿意主动开启与他人的交流。",
}

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    graph = build_graph().compile()

    reqs = [
        dict(
            trait_name=k, 
            bfi_item=v, 
            population_profile="中国在读大学生", 
            language="zh"
            )
        for k, v in DEMO_BFI.items()
    ]

    for r in reqs:
        state: PSJTState = {
            "request_id": str(uuid.uuid4()),
            "iter": 0,
            **r,
        }
        final = graph.invoke(state)
        packed = pack_item(final)
        fn = f"output/{packed['id']}.json"
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(packed, f, ensure_ascii=False, indent=2)
        print("\n=== 生成完成：", r["trait_name"], "===")
        print("题干：", packed["scenario_text"]) 
        for opt in packed["options"]:
            print(f"  {opt.get('label')}. {opt.get('action')}  [{opt.get('trait_level')}]")
        print("质量：", packed["quality"]) 
        print("已写入：", fn)
#%%
