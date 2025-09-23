# %%
from lmitf import TemplateLLM
from src.datasets.load_data import DataLoader


TRAIT = 'self-consciousness'
data_loader = DataLoader()
trait_defs = data_loader.load('_traits_definition', 'zh')

trait_polisher = TemplateLLM(
    "src/workflow/prompts/trait_polisher.py")

trait_decoder = TemplateLLM(
    "src/workflow/prompts/trait_decoder.py")

scenario_builder_a = TemplateLLM(
    "src/workflow/prompts/scenario_builder_a.py")

scenario_builder_b = TemplateLLM(
    "src/workflow/prompts/scenario_builder_b.py")

behavior_adapter = TemplateLLM(
    "src/workflow/prompts/behavior_adapter.py")
# %%
res_td = trait_decoder_prompt.call(
    trait_name="神经质-焦虑",
    item="我是一个充满烦恼的人。",
    trait_description="焦虑的个体忧虑、恐惧、容易担忧、紧张、神经过敏。得高分的人更可能有自由浮动的焦虑和恐惧。低分的人则是平静的、放松的。他们不会总是担心事情可能会出问题。",
    response_format='json', 
)
#%%
res_tp = trait_polisher_prompt.call(
    trait_name="神经质",
    trait_description="焦虑的个体忧虑、恐惧、容易担忧、紧张、神经过敏。得高分的人更可能有自由浮动的焦虑和恐惧。低分的人则是平静的、放松的。他们不会总是担心事情可能会出问题。",
    high_score= res_td['high_score'],
    low_score= res_td['low_score'],
    response_format='json', 
)
#%%
