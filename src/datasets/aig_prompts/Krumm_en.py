from __future__ import annotations
conditioned_frame="""
Can you create an SJT item measuring $Trait? 
The described situation in the SJT should be at least two sentences long and should be generated such that individuals with high, 
compared to low levels of $Trait will likely behave differently. 
It should come with four response options. 
Two of the four response options should reflect behaviours of individuals with high levels of $Trait and the other two behaviours of individuals with low levels of $Trait. 
Please indicate which response options refer to high and low levels of $Trait.

Response in JSON format with the following keys
{
    "situation": "The situation described in the SJT",
    "options": [
        {
            "A": "The first response option",
            
        },
        {
            "B": "The second response option",
        },
        {
            "C": "The third response option"
        },
        {
            "D": "The fourth response option"
        }
    ]
}
"""

prompt_template = [
    {"role": "user", "content":conditioned_frame}
]