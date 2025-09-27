# %%
from src import SJTRunner, DataLoader
#%%
N_ITEM = 4 #每个自陈题生成多少道sjt题
SCALE = "NEO-PI-R" #使用的量表
TRAITS = ["O4", "C4", "E5", "A5", "N5"] #选取特质
SITUATION_THEME = "大学校园里的日常生活" #情境主题
RESULT_DIR = 'output'  # 保存结果的目录
RESULT_SJT_FN = 'sjt-text' # 保存结果的文件名
RESULT_DETAILED_SJT_FN = 'sjt-text_detailed' # 保存结果的详细文件名
MODEL = 'gpt-5-mini' #使用的基座LLM

data_loader = DataLoader()# 数据加载器
neopir = data_loader.load(SCALE, 'zh')# 这里是加载NEO-PI-R数据，zh指中文，可以en即英文
neopir_meta = data_loader.load_meta(SCALE)# 这里是加载NEO-PI-R的元数据

available_traits = list(neopir.keys())
assert all(trait in available_traits for trait in TRAITS), \
    f"Some traits are not available. Available traits: {available_traits}"

trait_knowledge = {trait: neopir[trait]['facet_name'] for trait in available_traits}
available_traits_names = [f'{neopir[trait]['facet_name']}' for trait in available_traits]

# %%
runner = SJTRunner(
    situation_theme=SITUATION_THEME,
    scale=neopir,
    meta=neopir_meta
    )

all_items = runner.cook_async(
    TRAITS,
    n_item=N_ITEM,
    model='gpt-5-mini',
    save_results=True,
    results_dir=RESULT_DIR,
    detailed_fname=RESULT_DETAILED_SJT_FN,
    fname=RESULT_SJT_FN,
)
#%%
