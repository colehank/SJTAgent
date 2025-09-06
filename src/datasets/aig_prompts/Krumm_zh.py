from __future__ import annotations
conditioned_frame="""
你能否创建一个用于测量 $Trait 的情境判断测试题？
该题目的情境描述应至少包含两句话，并且设计要使高水平与低水平的 $Trait 个体在应对该情境时可能表现出明显不同的行为。
请为该情境提供四个应对选项，其中两个反映高水平 $Trait 个体的行为，另外两个反映低水平 $Trait 个体的行为。


请以 JSON 格式输出，包含以下键：
{
    "situation": "描述的情境"+"你会怎么做",
    "options": {
        "A": "第一个选项",
        "B": "第二个选项", 
        "C": "第三个选项",
        "D": "第四个选项"
    },
}
"""

prompt_template = [
    {"role": "user", "content":conditioned_frame}
]