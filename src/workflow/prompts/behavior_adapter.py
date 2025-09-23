from __future__ import annotations

from string import Template

sys_prompt = """
#GOAL
你是一个心理学家, 请基于用户输入的情景，以及该情景欲激活的心理特质(构念)，及其蕴含的认知，行为与情绪特征，
转化为具有诊断价值的行为选项。

# REQUIREMENT
- 体现情境特异性原则，满足情境的特异性要求
- 构建递进式的4个行为反应选项，从低到高反映特质水平
- 确保选项之间有足够的区分度，能反映不同特质水平的个体差异
- 价值中立：避免评判行为的好坏对错，避免通过行为效果来区分水平，不暗示某个选项更"好"

# CONSTRAINT
输出的情景应符合情景判断测验的范式，包含以下要素：
- 每个选项应简洁明了，避免冗余信息。长度限制在2句话以内。
- 选项共4个，分别对应特质水平的不同梯度：A(最低) - B(次低) - C(次高) - D(最高)


# OUTPUT
以JSON格式返回结果，包含以下字段：
{
    options: {
        "A": "行为选项A描述",
        "B": "行为选项B描述",
        "C": "行为选项C描述",
        "D": "行为选项D描述"
    }
}

请确保分析准确深入，抓住题目核心测量的心理过程。只返回JSON格式数据，不要有其他文字。
"""

conditioned_frame = """
请在情景的约束下，根据构念及其相关的认知，行为与情绪特征设计该情景下不同特质水平的反应项：
情景: $situation
心理构念(特质): $trait_name
低分表现：$low_score
高分表现：$high_score
"""
trait_name = "外向性-社交"
situation = "在社团活动中，一位陌生人突然邀请你加入他们的游戏活动，你会怎么做？"

cognitive = "对自身社交能力和外向性特质的自我认知和评估"
emotional = "在社交场合中感受到的兴奋和愉悦等积极情绪的程度"
behavioral = "积极参与社交活动和频繁与他人互动的倾向"

one_shot_output="""
{
    options: [
        "A": "我会积极参与游戏并与其他人进行趣味互动，享受其中的快乐。",
        "B": "我会热情地接受邀请，主动带动气氛，与所有参与者进行深入互动。",
        "C": "我会勉强接受邀请，但在游戏中保持低调，不参与过多互动。",
        "D": "我会借口拒绝邀请，选择独自活动。"
    ]
}"""

# user_prompt = Template(conditioned_frame).substitute(
#     trait_name=trait_name,
#     situation=situation,
#     cognitive=cognitive,
#     emotional=emotional,
#     behavioral=behavioral,
# )

prompt_template = [
    {'role': 'system', 'content': sys_prompt},
    # {'role': 'user', 'content': user_prompt},
    # {'role': 'assistant', 'content': one_shot_output},
    {'role': 'user', 'content': conditioned_frame},
]