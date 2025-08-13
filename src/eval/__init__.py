from .aig_eval import (
    PsychologicalTestEvaluator,
    EvaluationConfig,
    CostConfig,
    DimensionManager,
)
from .item_eval import PsychologicalItemEvaluator


def save_evaluation_results(results, figures, results_dir):
    """
    Save evaluation results to files.

    Args:
        results: Dictionary containing evaluation results
        figures: Dictionary containing matplotlib figures
        results_dir: Base directory to save results
    """
    import os
    import os.path as op
    import json
    detailed_result = op.join(results_dir, 'detailed')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(detailed_result, exist_ok=True)

    # Save trait-specific results as CSV
    for trait in results['results']:
        df = results['results'][trait]
        df.to_csv(op.join(detailed_result, f'{trait}_eval.csv'), index=False)

    # Save win rates as JSON
    detailed_result = op.join(results_dir, 'detailed')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(detailed_result, exist_ok=True)
    
    # Save trait-specific results as CSV
    for trait in results['results']:
        df = results['results'][trait]
        df.to_csv(op.join(detailed_result, f'{trait}_eval.csv'), index=False)

    # Save win rates as JSON
    with open(op.join(detailed_result, 'win_rates.json'), 'w', encoding='utf-8') as f:
        json.dump(results['win_rates'], f, ensure_ascii=False, indent=4)

    with open(op.join(results_dir, 'overall_win_rates.json'), 'w', encoding='utf-8') as f:
        json.dump(results['overall_win_rates'], f, ensure_ascii=False, indent=4)

    # Save figures
    figures['multi_trait'].savefig(op.join(results_dir, 'multi_trait_eval.png'), dpi=300, bbox_inches='tight', transparent=True)
    figures['overall'].savefig(op.join(results_dir, 'overall_eval.png'), dpi=300, bbox_inches='tight', transparent=True)