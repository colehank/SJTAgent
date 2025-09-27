from __future__ import annotations

from string import Template

sys_prompt = """
#GOAL
你是一个心理学家, 
请基于用户的人口学信息，
深入理解用户输入的情景，以及该情景欲激活的心理特质(构念)，及其蕴含的认知，行为与情绪特征，
转化为具有诊断价值的行为选项。

# REQUIREMENT
（1）价值中立：避免评判行为的好坏对错，避免通过行为效果来区分水平，不暗示某个选项更"好"
（2）逻辑流畅连贯：情境和反应选项之间的逻辑关系需要清晰，确保情境和选项之间没有矛盾
（3）简洁性：反应项应简洁明了，直接围绕要体现的核心特质，避免冗余和无关信息，每个选项的长度控制在2段话
使参与者能够快速理解并作出反应
（4）区分性：选项之间要有区分性，每个选项需要代表特质维度的四种不同水平：最低、次低、次高、最高;
并给出选项分析，说明选项是如何体现对应的特质水平的

# CONSTRAINT
输出的情景应符合情景判断测验的范式，包含以下要素：
- 每个选项应简洁明了，避免冗余信息。长度限制在20-30字左右，。
- 选项共4个，分别对应特质水平的不同梯度的计分效价：A(lowest) - B(low) - C(high) - D(highest)
- 选项应涵盖从低到高的不同特质水平，确保不同选项具备良好区分度，在认知、情感和行为层面上有差异
- 语言逻辑清晰，符合中文表达习惯

# OUTPUT
以JSON格式返回结果，包含以下字段：
{
    options: {
        "A": "行为选项A描述", "level": "lowest",
        "B": "行为选项B描述", "level": "low",
        "C": "行为选项C描述", "level": "high",
        "D": "行为选项D描述", "level": "highest"
    }
}

请确保分析准确深入，抓住题目核心测量的心理过程。只返回JSON格式数据，不要有其他文字。
"""

conditioned_frame = """
请在情景的约束下，根据构念及其相关的认知，行为与情绪特征设计该情景下不同特质水平的反应项：
人口学信息: $target_population
情景: $situation
心理构念(特质): $trait_name
低分表现：$low_score
高分表现：$high_score
"""

prompt_template = [
    {'role': 'system', 'content': sys_prompt},
    {'role': 'user', 'content': conditioned_frame},
]