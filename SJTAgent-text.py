# %%
from src import SJTRunner, DataLoader
#%%
N_ITEM = 4 #每个自陈题生成多少道sjt题
SCALE = "NEO-PI-R" #使用的量表
TRAITS = ["O4", "C4", "E2", "A5", "N4"] #选取特质
SITUATION_THEME = "日常生活中的普遍情景" #情境主题
TARGET_POPULATION = '普通成年人' #目标人群
MODEL = 'deepseek-v3.1-thinking' #使用的基座LLM

RESULT_DIR = 'output'  # 保存结果的目录
RESULT_SJT_FN = 'sjt-text' # 保存结果的文件名
RESULT_DETAILED_SJT_FN = 'sjt-text_detailed' # 保存结果的详细文件名

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
    meta=neopir_meta,
    target_population='普通成年人',
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
