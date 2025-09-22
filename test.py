# %%
from lmitf import TemplateLLM

trait_decoder_prompt = TemplateLLM(
    "/Users/yunyun/Documents/项目/多模态测评/SJTAgent/src/workflow/prompts/trait_decoder.py")

res = trait_decoder_prompt.call(
    trait_name="神经质",
    item="我不是一个充满烦恼的人。",
    trait_description="焦虑的个体忧虑、恐惧、容易担忧、紧张、神经过敏。得高分的人更可能有自由浮动的焦虑和恐惧。低分的人则是平静的、放松的。他们不会总是担心事情可能会出问题。",
    response_format='json', 
)
# %%
