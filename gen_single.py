# %%
from src import SJTAgent, DataLoader
import os.path as op
import os
import json
#%%
SITUATION_THEME = "日常生活中的普遍情景" #情境主题
TARGET_POPULATION = '普通成年人' #目标人群

SCALE = "NEO-PI-R" #使用的量表
TRAIT = "O4" #选取特质
ITEM_ID = 1 # 自陈量表的哪个题目

N_ITEM = 4 #每个自陈题生成多少道sjt题
MODEL = 'deepseek-v3.1-thinking' #使用的基座LLM

RESULT_DIR = op.join('output', 'single')  # 保存结果的目录
RESULT_SJT_FN = f"sjt-text-{TRAIT}" # 保存结果的文件名
RESULT_DETAILED_SJT_FN = f"sjt-text-{TRAIT}_detailed" # 保存结果的详细文件名
# %%

data_loader = DataLoader()# 数据加载器
neopir = data_loader.load(SCALE, 'zh')# 这里是加载NEO-PI-R数据，zh指中文，可以en即英文
neopir_meta = data_loader.load_meta(SCALE)# 这里是加载NEO-PI-R的元数据

available_items_idxs = list(neopir[TRAIT]['items'].keys())
map_item_id = available_items_idxs[ITEM_ID - 1]
item = neopir[TRAIT]['items'][map_item_id]['item'] # 题干
whole_trait_name = f"{neopir_meta[TRAIT]['domain']}-{neopir_meta[TRAIT]['facet_name']}" # 特质的完整名称
trait_description = neopir_meta[TRAIT]['description']
trait_high_score = neopir_meta[TRAIT]['high_score']
trait_low_score = neopir_meta[TRAIT]['low_score']
#%%
agent = SJTAgent(
    situation_theme=SITUATION_THEME,
    target_population=TARGET_POPULATION,
    )

items = agent.run(
    item=item,
    trait_name=whole_trait_name,
    trait_description=trait_description,
    low_score=trait_low_score,
    high_score=trait_high_score,
    n_item=N_ITEM,
    model=MODEL,
)
# %%
os.makedirs(RESULT_DIR, exist_ok=True)

with open(op.join(RESULT_DIR, f'{RESULT_DETAILED_SJT_FN}.json'), 'w', encoding='utf-8') as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

pure_items = {}
for idx, sjt in enumerate(items['items'], 1):
    pure_items[f'{idx}'] = sjt
with open(os.path.join(RESULT_DIR, f'{RESULT_SJT_FN}.json'), 'w', encoding='utf-8') as f:
    json.dump(pure_items, f, ensure_ascii=False, indent=2)
#%%
