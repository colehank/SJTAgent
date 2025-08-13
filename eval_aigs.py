# %%
import os
import os.path as op


from src.eval import (
    PsychologicalTestEvaluator,
    EvaluationConfig,
    CostConfig,
    DimensionManager,
    save_evaluation_results
)
# %%
TRAITS = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
DATA_PATHS = {
    'mussel': op.join('datasets', 'SJTs', 'Mussel_zh.json'),
    'li': op.join('results', 'SJTs', 'LiSJT_zh.json'), 
    'krumm': op.join('results', 'SJTs', 'KrummSJT_zh.json')
}
AIG_NAMES = list(DATA_PATHS.keys())
# %%
cost_config = CostConfig(
    input_token_rate=0.8,
    output_token_rate=2.0
)

config = EvaluationConfig(
    traits=TRAITS,
    data_paths=DATA_PATHS,
    cost_config=cost_config,
    batch_size=1000,
    max_concurrent=1000,
    show_progress=True
)

dimensions = DimensionManager.get_dimensions(TRAITS)

evaluator = PsychologicalTestEvaluator(
    config,
    aig_names=AIG_NAMES,
    dimensions=dimensions
)
# %%
results = evaluator.run_evaluation(model='gpt-4o-mini')
figures = evaluator.create_visualizations(results, save_plots=False)
evaluator.print_summary(results)
#%%
results_dir = op.join('results', 'eval')
save_evaluation_results(results, figures, results_dir)
#%%
