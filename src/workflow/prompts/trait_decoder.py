from __future__ import annotations

from string import Template




##修改：根据特质的描述+对应量表题目，一题一对应，每题可以生成n个题目；n由用户进行调整
sys_prompt = """你是一个心理学家, 
请分析用户输入的心理构念(特质)与对应的量表题目，提取其中隐含的心理特质成分，需要分别包括特质的高水平表现与低水平表现：

具体来说，你需要提取以下三个成分：
1. 认知成分：该题目测量的认知评估或信念倾向
2. 情感成分：该题目关联的情绪体验或感受
3. 行为倾向：该题目反映的行为模式或反应方式

以JSON格式返回结果，包含以下字段：
high_score:{
    cognitive: "认知成分描述",
    emotional: "情感成分描述",
    behavioral: "行为倾向描述"
}
low_score:{
    cognitive: "认知成分描述",
    emotional: "情感成分描述",
    behavioral: "行为倾向描述"
}
"""

conditioned_frame = """
请分析以下特质与量表题目，提取其中隐含的心理特质成分：

心理构念(特质): $trait_name
特质描述：$trait_description
量表题目: $item
"""

trait = "神经质-焦虑"
item = """我是一个充满烦恼的人。"""
trait_description ="""焦虑的个体忧虑、恐惧、容易担忧、紧张、神经过敏。得高分的人更可能有自由浮动的焦虑和恐惧。低分的人则是平静的、放松的。他们不会总是担心事情可能会出问题。
高分特点：焦虑，容易感觉到危险和威胁，容易紧张、恐惧、担忧、不安。
低分特点：心态平静，放松，不容易感到害怕，不会总是担心事情可能会出问题，情绪平静、放松、稳定"""

one_shot_output = """
high_score:
{"cognitive": "倾向于认为自己容易被各种烦恼所困扰，对生活中的压力和问题感到难以应对。",
"emotional": "经常体验到焦虑、担忧、紧张等负面情绪，情绪状态不稳定，容易受到外界因素的影响。",
"behavioral": "在面对压力或挑战时，容易情绪化，行为表现可能较为冲动或不稳定，难以保持冷静和理智。"}
low_score:
{"cognitive": "倾向于认为自己能够有效地管理生活中的压力和挑战，展现出较高的心理韧性。",
"emotional": "通常保持平和稳定的情绪状态，能够以积极的心态面对生活中的起伏。",
"behavioral": "在面对压力或挑战时，能够保持冷静和理智，展现出良好的情绪调节能力，行为表现沉着且有条理。"}
"""

user_prompt = Template(conditioned_frame).substitute(
    trait_name=trait,
    item=item,
    trait_description=trait_description,
)

prompt_template = [
    {'role': 'system', 'content': sys_prompt},
    {'role': 'user', 'content': user_prompt},
    {'role': 'assistant', 'content': one_shot_output},
    {'role': 'user', 'content': conditioned_frame},
]

