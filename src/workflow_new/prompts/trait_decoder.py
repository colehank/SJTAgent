from __future__ import annotations

from string import Template

sys_prompt = """你是一个心理学家, 
请分析用户输入的心理构念(特质)与对应的量表题目，提取其中隐含的心理特质成分：

具体来说，你需要提取以下三个成分：
1. 认知成分：该题目测量的认知评估或信念倾向
2. 情感成分：该题目关联的情绪体验或感受
3. 行为倾向：该题目反映的行为模式或反应方式

以JSON格式返回结果，包含以下字段：
{
    cognitive: "认知成分描述",
    emotional: "情感成分描述",
    behavioral: "行为倾向描述"
}
"""

conditioned_frame = """
请分析以下特质与量表题目，提取其中隐含的心理特质成分：

心理构念(特质): $trait_name
量表题目: $item
"""

trait = "神经质"
item = """我不是一个充满烦恼的人。"""

one_shot_output = """{"cognitive": "对烦恼的低敏感性",
"emotional": "情绪稳定，较少感到焦虑或忧虑",
"behavioral": "在面对压力时表现出较强的适应能力"}"""

user_prompt = Template(conditioned_frame).substitute(
    trait_name=trait,
    item=item,
)

prompt_template = [
    {'role': 'system', 'content': sys_prompt},
    {'role': 'user', 'content': user_prompt},
    {'role': 'assistant', 'content': one_shot_output},
    {'role': 'user', 'content': conditioned_frame},
]