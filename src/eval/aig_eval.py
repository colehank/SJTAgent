# %%
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from .item_eval import PsychologicalItemEvaluator, CostConfig
import re
# %%
@dataclass
class EvaluationConfig:
    """Configuration for psychological item evaluation"""
    traits: list[str]
    data_paths: dict[str, str]
    cost_config: CostConfig
    batch_size: int = 3000
    max_concurrent: int = 1000
    show_progress: bool = True

class DimensionManager:
    """Manages evaluation dimensions for psychological traits"""
    
    @staticmethod
    def get_dimensions(traits: list[str]) -> list[dict[str, str]]:
        """Generate evaluation dimensions for a given trait"""
        dimensions = {
            trait: [
            {
                "name": 'NecessityOfTheSituation',
                "description": f"对比题目 A 与题目 B，选择哪一个情境更不可或缺、更能准确反映{trait}的人格特质"
            },
            {
                "name": "RationalityOfOptions", 
                "description": f"对比题目 A 与题目 B，在选项的现实性与情境相关性方面，哪一个更符合生活情境、更具合理性"
            },
            {
                "name": "RationalityOfScoring",
                "description": f"高{trait}水平为选项A与B，低{trait}水平为选项C与D，对比题目 A 与题目 B，哪一个更准确合理"
            },
            {
                "name": "OverallItemQuality",
                "description": f"对比题目 A 与题目 B，在语法准确度，测量{trait}的情境丰富度、心理真实性及评分方式等方面，哪一个整体更优"
            }
        ] for trait in traits
        }
        return dimensions

class DataLoader:
    """Handles loading and preprocessing of psychological test data"""
    
    def __init__(self, data_paths: dict[str, str]):
        self.data_paths = data_paths
        self.logger = logging.getLogger(__name__)
        
    def load_json_data(self, filepath: str) -> dict:
        """Load JSON data with error handling"""
        try:
            with open(filepath, encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"File not found: {filepath}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in file: {filepath}")
            raise
    
    def load_all_datasets(
        self, 
        traits: list[str],
        aig_names: list
        ) -> dict[str, dict]:
        """Load all datasets for given traits"""
        datasets = {}
        
        for aig in aig_names:
            if aig not in self.data_paths:
                self.logger.warning(f"Data path for {aig} not found in configuration.")
                break
            
            # Load raw data
            try:
                raw_data = self.load_json_data(self.data_paths[aig])
            except Exception as e:
                self.logger.error(f"Failed to load data for {aig}: {e}")
                continue
            
            # Organize by trait
            for trait in traits:
                if trait not in datasets:
                    datasets[trait] = {}
                datasets[trait][aig] = raw_data.get(trait, {})

        return self._flatten_datasets(datasets)
    
    def _flatten_datasets(self, datasets: dict) -> dict:
        """Flatten nested dataset structure for evaluation"""
        flattened = {}
        for trait, tests in datasets.items():
            flattened[trait] = {}
            for test_name, test_data in tests.items():
                for idx, item in test_data.items():
                    flattened[trait][f'{test_name}_{idx}'] = item
        return flattened
    
class WinRateCalculator:
    """Calculates win rates for different test types across dimensions"""
    
    @staticmethod
    def calculate_win_rates(
        df: pd.DataFrame, 
        dimensions: Optional[list[str]] = None,
        test_types: Optional[list[str]] = None
        ) -> dict[str, dict[str, float]]:
        """Calculate win rates for each test type and dimension"""
        win_rates = {}
        if test_types is None:
            raise ValueError("test_types must be provided")
        
        if dimensions is None:
            dimensions = df['dimension'].unique()
        
        for dim in dimensions:
            win_rates[dim] = {}
            for test_type in test_types:
                wins = 0
                total = 0
                
                for _, row in df.iterrows():
                    if row['dimension'] != dim:
                        continue
                        
                    type_A = row['A'].split('_')[0]
                    type_B = row['B'].split('_')[0]

                    total += 1
                    if ((row['winner'] == 'A' and type_A == test_type) or 
                        (row['winner'] == 'B' and type_B == test_type)):
                        wins += 1
                
                win_rates[dim][test_type] = wins / total if total > 0 else 0
        
        return win_rates
    
    @staticmethod
    def calculate_overall_win_rates(
        win_rates_by_trait: dict[str, dict],                           
        traits: list[str],
        test_types: Optional[list[str]],
        dimensions: Optional[list[str]] = None,
        ) -> dict[str, dict[str, float]]:
        """Calculate overall win rates across all traits"""
        overall_win_rates = {}
        if test_types is None:
            raise ValueError("test_types must be provided")

        if dimensions is None:
            all_dimensions = set()
            for trait_win_rates in win_rates_by_trait.values():
                all_dimensions.update(trait_win_rates.keys())
            dimensions = list(all_dimensions)
        
        for dim in dimensions:
            overall_win_rates[dim] = {}
            for test_type in test_types:
                total_rate = 0
                count = 0
                for trait in traits:
                    if (dim in win_rates_by_trait[trait] and 
                        test_type in win_rates_by_trait[trait][dim]):
                        total_rate += win_rates_by_trait[trait][dim][test_type]
                        count += 1
                overall_win_rates[dim][test_type] = total_rate / count if count > 0 else 0
        
        return overall_win_rates
    
class RadarChartVisualizer:
    """Creates elegant radar charts for psychological test evaluation results"""

    def __init__(self, test_types: list[str]):
        self.setup_matplotlib()
        self.test_types = test_types
        self.colors = plt.cm.Set2(np.linspace(0, 1, len(test_types)))

    def setup_matplotlib(self):
        """Configure matplotlib for better visualization"""
        plt.rcParams['font.sans-serif'] = ['Times New Roman', 'Songti SC', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
    
    def create_multi_trait_radar(self, win_rates: dict[str, dict], 
                               traits: list[str],
                               figsize: tuple[int, int] = (20, 4),
                               save_path: Optional[str] = None) -> Figure:
        """Create radar charts for multiple traits"""
        fig, axes = plt.subplots(1, len(traits), figsize=figsize, 
                               subplot_kw=dict(projection='polar'))
        
        if len(traits) == 1:
            axes = [axes]
        
        for idx, trait in enumerate(traits):
            self._plot_single_radar(axes[idx], win_rates[trait], trait, show_legend=(idx == 0))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def create_overall_radar(self, overall_win_rates: dict[str, dict[str, float]], 
                           figsize: tuple[int, int] = (8, 8),
                           save_path: Optional[str] = None) -> Figure:
        """Create overall performance radar chart"""
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        dimensions = list(overall_win_rates.keys())
        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, test_type in enumerate(self.test_types):
            values = [overall_win_rates[dimension][test_type] for dimension in dimensions]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=3, label=test_type, 
                   color=self.colors[i], markersize=8)
            ax.fill(angles, values, alpha=0.2, color=self.colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self._format_dimension_labels(dimensions), fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Test Performance\n(Average across all traits)', 
                    size=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def _plot_single_radar(self, ax, trait_win_rates: dict[str, dict[str, float]], 
                          trait: str, show_legend: bool = False):
        """Plot radar chart for a single trait"""
        dimensions = list(trait_win_rates.keys())
        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, test_type in enumerate(self.test_types):
            values = [trait_win_rates[dimension][test_type] for dimension in dimensions]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2.5, label=test_type, 
                   color=self.colors[i], markersize=6)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self._format_dimension_labels(dimensions), fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(trait, size=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        
        if show_legend:
            ax.legend(loc='upper left', bbox_to_anchor=(-0.3, 1.1), fontsize=10)
    
    def _format_dimension_labels(self, dimensions: list[str]) -> list[str]:
        """Format dimension labels for better readability"""
        formatted = []
        for d in dimensions:
            # 自动识别驼峰命名并在大写字母前添加换行
            formatted_d = re.sub(r'(?<!^)([A-Z])', r'\n\1', d)
            formatted.append(formatted_d)
        return formatted
        
class PsychologicalTestEvaluator:
    """Main class for psychological test evaluation with iterative analysis"""
    
    def __init__(
        self, 
        config: EvaluationConfig, 
        aig_names: list[str],
        dimensions: Optional[dict[str, list[dict[str, str]]]] = None,
    ):
        self.config = config
        self.aig_names = aig_names
        self.data_loader = DataLoader(config.data_paths)
        self.dimension_manager = DimensionManager()
        self.win_rate_calculator = WinRateCalculator()
        self.visualizer = RadarChartVisualizer(test_types=self.aig_names)
        self.logger = self._setup_logging()
        
        if dimensions is None:
            self.dimensions = self.dimension_manager.get_dimensions(config.traits)
        else:
            self.dimensions = dimensions

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the evaluation process"""
        logging.basicConfig(
            level=logging.WARN,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def run_evaluation(self, model='qwen-plus') -> dict[str, Any]:
        """Run complete evaluation pipeline"""
        self.logger.info("Starting psychological test evaluation...")
        
        # Load and prepare data
        self.datasets = self.data_loader.load_all_datasets(
            self.config.traits,
            self.aig_names
        )
        
        # Run evaluations for each trait
        results = {}
        for trait in self.config.traits:
            self.logger.info(f"Evaluating trait: {trait}")
            self.evaluator = PsychologicalItemEvaluator(
                cost_config=self.config.cost_config,
                model_name=model
            )

            results[trait] = self.evaluator.evaluate_test_items(
                self.datasets[trait],
                self.dimensions[trait],
                batch_size=self.config.batch_size,
                max_concurrent=self.config.max_concurrent,
                show_progress=self.config.show_progress
            )
        
        # Calculate win rates
        win_rates = {
            trait: self.win_rate_calculator.calculate_win_rates(
                results_df,
                test_types=self.aig_names,
                )
            for trait, results_df in results.items()
        }
        
        # Calculate overall win rates
        overall_win_rates = self.win_rate_calculator.calculate_overall_win_rates(
            win_rates, self.config.traits, test_types=self.aig_names,
        )
        
        self.logger.info("Evaluation completed successfully!")
        
        return {
            'results': results,
            'win_rates': win_rates,
            'overall_win_rates': overall_win_rates
        }
    
    def create_visualizations(self, evaluation_results: dict[str, Any], 
                            save_plots: bool = True, output_dir: str = "./plots") -> dict[str, Figure]:
        """Create all visualization plots"""
        if save_plots:
            Path(output_dir).mkdir(exist_ok=True)
        
        figures = {}
        
        # Multi-trait radar chart
        multi_trait_path = f"{output_dir}/multi_trait_radar.png" if save_plots else None
        figures['multi_trait'] = self.visualizer.create_multi_trait_radar(
            evaluation_results['win_rates'], 
            self.config.traits,
            save_path=multi_trait_path
        )
        
        # Overall radar chart
        overall_path = f"{output_dir}/overall_radar.png" if save_plots else None
        figures['overall'] = self.visualizer.create_overall_radar(
            evaluation_results['overall_win_rates'],
            save_path=overall_path
        )
        
        return figures
        
    def export_results(self, evaluation_results: dict[str, Any], 
                      output_path: str = "./results.json"):
        """Export evaluation results to JSON"""
        # Convert DataFrames to dictionaries for JSON serialization
        exportable_results = {
            'win_rates': evaluation_results['win_rates'],
            'overall_win_rates': evaluation_results['overall_win_rates'],
            'config': {
                'traits': self.config.traits,
                'batch_size': self.config.batch_size,
                'max_concurrent': self.config.max_concurrent
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(exportable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results exported to {output_path}")
    
    def print_summary(self, evaluation_results: dict[str, Any]):
        """Print summary of evaluation results"""
        print("\n" + "="*60)
        print("PSYCHOLOGICAL TEST EVALUATION SUMMARY")
        print("="*60)
        
        overall_win_rates = evaluation_results['overall_win_rates']
        dimensions = list(overall_win_rates.keys())
        test_types = self.aig_names

        print("\nOverall Win Rates (Average across all traits):")
        for dim in dimensions:
            print(f"\n{dim}:")
            for test_type in test_types:
                rate = overall_win_rates[dim][test_type]
                print(f"  {test_type:8}: {rate:.3f}")
        
        print("\n" + "="*60)

# %%
if __name__ == "__main__":
    """Example usage of the PsychologicalTestEvaluator"""
    
    # Define configuration
    traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    data_paths = {
        'mussel': 'datasets/SJTs/Mussel.json',
        'li': 'results/SJTs/LiSJT_en.json', 
        'krumm': 'results/SJTs/KrummSJT_en.json'
    }
    
    cost_config = CostConfig(
        input_token_rate=0.8,
        output_token_rate=2.0
    )
    
    config = EvaluationConfig(
        traits=traits,
        data_paths=data_paths,
        cost_config=cost_config,
        batch_size=2000,
        max_concurrent=2000,
        show_progress=True
    )
    # Initialize evaluator
    evaluator = PsychologicalTestEvaluator(config)
    # %%
    # Run evaluation
    results = evaluator.run_evaluation(model='DeepSeek-V3')
    
    # Create visualizations
    figures = evaluator.create_visualizations(results, save_plots=False)
    
    # Print summary
    evaluator.print_summary(results)
    
    # Show plots
    plt.show()

#%%
