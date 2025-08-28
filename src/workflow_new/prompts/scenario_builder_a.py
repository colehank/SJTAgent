from __future__ import annotations

from string import Template

sys_prompt = """你是一个心理学家, 
请基于用户输入的情景主题，以及心理特质(构念)，及其蕴含的认知，行为与情绪特征，
思考并设计能够激活此心理特质(构念)的具体触发事件

以JSON格式返回结果，包含以下字段：
{
    cues: ["触发事件1", "触发事件2", "..."]
}

请确保分析准确深入，抓住题目核心测量的心理过程。只返回JSON格式数据，不要有其他文字。
"""

conditioned_frame = """
请在情景主题的框架下，根据构念及其相关的认知，行为与情绪特征设计能够激活特质的 $n_cue 个触发事件：

情景主题: $situation_theme
心理构念(特质): $trait_name
认知特征: $cognitive
情感特征: $emotional
行为特征: $behavioral
"""

trait_name = "外向性-社交"
situation_theme = "大学校园里的日常生活"
n_cue = 3
cognitive = "对自身社交能力和外向性特质的自我认知和评估"
emotional = "在社交场合中感受到的兴奋和愉悦等积极情绪的程度"
behavioral = "积极参与社交活动和频繁与他人互动的倾向"

one_shot_output="""
{
    "cues": [
        "在一个大型社交活动上，你发现自己周边有很多陌生人。",
        "你被邀请参加一个由同学组织的聚会，认识一些新朋友。",
        "你在校园里遇到一个老朋友，TA邀请你一起参加一个有趣的活动。"
    ]
}"""

user_prompt = Template(conditioned_frame).substitute(
    trait_name=trait_name,
    situation_theme=situation_theme,
    n_cue=n_cue,
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