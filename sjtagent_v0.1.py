# %%
from src.workflow import SJTAgent
import src
import concurrent.futures

# data_loader是数据加载器
data_loader = src.DataLoader()

# 这里是加载NEO-PI-R数据，语言是中文，可以选zh，可以选en
neopir = data_loader.load("NEO-PI-R", 'zh') 
# 这里是加载NEO-PI-R的元数据
neopir_meta = data_loader.load_meta("NEO-PI-R")
trait_descriptions = data_loader.load('_traits_definition', 'zh')
#%%
traits = ["O5", "C5", "E2", "A4", "N4"] #Mussel's 5 big traits

neopir_items = {
    trait: [
        neopir[trait]['items'][key]["item"] for key in neopir[trait]['items']
    ] for trait in traits
}
# %%
generator = SJTAgent(
        situation_theme="大学校园里的日常生活",
    )
# %%
def generate_for_trait(
    trait,
    items,
    n_sjt_per_item=3,
    model='gpt-5'
    ):
    domain = neopir_meta[trait]['domain']
    facet = neopir_meta[trait]['facet_name']
    trait_name = f"{domain}-{facet}"
    trait_description = trait_descriptions[domain]['facet_name'][facet]['description']
    low_score = trait_descriptions[domain]['facet_name'][facet]['low_score']
    high_score = trait_descriptions[domain]['facet_name'][facet]['high_score']

    results = []
    for item in items:
        sjts = generator.generate_items(
            trait_name,
            trait_description,
            low_score,
            high_score,
            item,
            n_sjt_per_item, 
            model=model
        )
        results.append(sjts)
    return trait, results

all_items = {trait[0]: [] for trait in traits}
with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(generate_for_trait, trait, neopir_items[trait]) for trait in traits]
    for future in concurrent.futures.as_completed(futures):
        trait, res = future.result()
        all_items[trait[0]].append(res)
#%%
import os.path as op
import json
with open (op.join('results', 'exp_SJTs', 'ours_24_per_dim.json'), 'w', encoding='utf-8') as f:
    json.dump(all_items, f, ensure_ascii=False, indent=2)
#%%
final_item = {}
for trait in all_items:
    final_item[trait] = {}
    this_trait_idx = 0
    for res in all_items[trait]:
        for item_res in res:
            for sjts in item_res['items']:
                this_trait_idx += 1
                final_item[trait][f'{this_trait_idx}'] = sjts
with open (op.join('results', 'SJTs', 'SJTAgent_v0.1.json'), 'w', encoding='utf-8') as f:
    json.dump(final_item, f, ensure_ascii=False, indent=2)
#%%
