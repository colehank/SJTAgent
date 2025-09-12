# %%
from lmitf import TemplateLLM

trait_decoder_prompt = TemplateLLM(
    "/Users/yunyun/Documents/项目/多模态测评/SJTAgent/src/workflow/prompts/trait_decoder.py")

# %%
trait_decoder_prompt.call(
    trait_name="神经质",
    item="我不是一个充满烦恼的人。",
    response_format="json",
)
# %%
scenario_builder_a_prompt = TemplateLLM(
    "/Users/yunyun/Documents/项目/多模态测评/SJTAgent/src/workflow/prompts/scenario_builder_a.py")
# %%
scenario_builder_a_prompt
# %%