# %%
from src.workflow import generate_psjt_item

DEMO_BFI = {
    "尽责性-条理性": "我做事有计划并保持条理。",
    "外向性-社交主动": "我愿意主动开启与他人的交流。",
}

results = []
for trait_name, bfi_item in DEMO_BFI.items():
    print(f"\n=== Processing: {trait_name} ===")
    
    result = generate_psjt_item(
        trait_name=trait_name,
        bfi_item=bfi_item,
        population_profile="中国在读大学生",
        language="zh"
    )
    
    results.append(result)
    
    print(f"题干：{result['scenario_text']}")
    for option in result['options']:
        print(f"  {option.get('label')}. {option.get('action')}  [{option.get('trait_level')}]")
    print(f"质量：{result['quality']}")