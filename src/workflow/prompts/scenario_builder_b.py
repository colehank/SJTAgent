from __future__ import annotations

from string import Template

sys_prompt = """
#GOAL
你是一个心理学家, 请基于用户输入的触发事件，以及事件关联的心理特质(构念)，及其蕴含的认知，行为与情绪特征，
思考并设计能够激活此心理特质(构念)的具体情景。

# REQUIREMENT
该情景应符合情景判断测验(SJT)的设计原则，能够有效激活关联的心理特质(构念)所蕴含的认知，行为与情绪特征。
根据用户所要求的情景数量，设计多个不同的情景。
情景需要言简意赅的交代故事的发生背景与事件经过。

# CONSTRAINT
输出的情景应符合情景判断测验的范式，包含以下要素：
1. 情景简洁明了：言简意赅地描述情景，避免冗余信息。长度限制在3句话以内。
2. 情境具有足够的生态效度，贴近现实生活
3. 情境能有效引发不同特质（构念）水平个体的表现差异。
4. 情景结束后，以“你会怎么做？”结尾。

# OUTPUT
以JSON格式返回结果，包含以下字段：
{
    situation: [
        "情景1",
        "情景2",
        "情景3"
    ]
}

请确保分析准确深入，抓住题目核心测量的心理过程。只返回JSON格式数据，不要有其他文字。
"""

conditioned_frame = """
请在触发事件的约束下，根据构念及其相关的认知，行为与情绪特征设计能够激活特质的 $n_situ 个触发事件：

触发事件: $cue
心理构念(特质): $trait_name
认知特征: $cognitive
情感特征: $emotional
行为特征: $behavioral
"""

trait_name = "外向性-社交"
cue = "在一个大型社交活动上，你发现自己周边有很多陌生人。"
n_situ = 1
cognitive = "对自身社交能力和外向性特质的自我认知和评估"
emotional = "在社交场合中感受到的兴奋和愉悦等积极情绪的程度"
behavioral = "积极参与社交活动和频繁与他人互动的倾向"

one_shot_output="""
{
    situation: ["在社团活动中，一位陌生人突然邀请你加入他们的游戏活动，你会怎么做？"]
}"""

user_prompt = Template(conditioned_frame).substitute(
    cue=cue,
    trait_name=trait_name,
    n_situ=n_situ,
    cognitive=cognitive,
    emotional=emotional,
    behavioral=behavioral,
)

prompt_template = [
    {'role': 'system', 'content': sys_prompt},
    {'role': 'user', 'content': user_prompt},
    {'role': 'assistant', 'content': one_shot_output},
    {'role': 'user', 'content': conditioned_frame},
]