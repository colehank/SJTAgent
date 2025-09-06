from __future__ import annotations
conditioned_frame="""
Please continue generating up to Scenario $Nitem to measure the level of $Trait.

$TraitDescription.

The specific requirements are as follows:
1. Role and Task Objective: Please act as a psychometrics expert, focusing on designing scenarios that reflect the level of $Trait, making them applicable to everyday life or common workplace environments.

2. Constraints:
(1) The scenario descriptions must be detailed, diverse, and closely related to $Trait;
(2) The scenarios should end with the question "What would you do?";
(3) Each scenario should provide four options, which should be realistic and contextually relevant. Two options should reflect a high level of $Trait (scoring 1), and two should reflect a low level of $Trait (scoring 0).
(4) The language should be fluent, conform to the linguistic norms and grammar rules of English, and align with psychological paradigms.

3. Example
###
$Example
###

4. Let's think through this step by step. After generating the questions, please explain the basic principles behind the scenario design and scoring for each option, and relate them to the characteristics of $Trait. 
Your work is important to my research!

Based on the above requirements and examples, please continue generating up to Scenario $Nitem.

Output in JSON format, including the following keys,
where options A and B represent high levels of $Trait, and C and D represent low levels of $Trait:
{
        "1":{
            "situation": "描述的情境",
            "options": {
                "A": "第一个选项",
                "B": "第二个选项",
                "C": "第三个选项",
                "D": "第四个选项",
            }
        }
        ...
        "$Nitem":{
            "situation": "描述的情境",
            "options": {
                "A": "第一个选项",
                "B": "第二个选项",
                "C": "第三个选项",
                "D": "第四个选项",
            }
        }
}
"""

prompt_template = [
    {'role': 'user', 'content':conditioned_frame},
]
