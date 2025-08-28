"""SJT (Situational Judgment Test) Generator

This script generates SJTs for personality traits using two different approaches:
- Krumm approach: Iterative generation
- Li approach: Batch generation with examples
"""

import asyncio
import json
import os
import os.path as op
from typing import Dict, List, Any

import lmitf
from lmitf import TemplateLLM
import src

from tqdm.autonotebook import tqdm
import argparse
from dotenv import load_dotenv
load_dotenv()
parser = argparse.ArgumentParser(description='Generate SJT items using Krumm and Li approaches')
parser.add_argument('--language', type=str, default='zh', choices=['zh', 'en'], 
                    help='Language for generation (default: zh)')
parser.add_argument('--model', type=str, default='gpt-4o',
                    help='Model to use for generation (default: gpt-4o)')
parser.add_argument('--n_items', type=int, default=22,
                    help='Number of items to generate per trait (default: 22)')

args = parser.parse_args()

# Update global variables with command line arguments
LANGUAGE = args.language
MODEL = args.model
N_ITEMS = args.n_items
TRAITS = [
    "Openness-Openness to Ideas",
    "Conscientiousness-Self-Discipline", 
    "Extraversion-Gregariousness",
    "Agreeableness-Compliance",
    "Neuroticism-Self-Consciousness",
]
krumm_prompt = f"Krumm_{LANGUAGE}.py"
li_prompt = f"Li_{LANGUAGE}.py"
# File paths

data_loader = src.DataLoader()
KRUMM_PROMPT_PATH = data_loader.load("aig_prompts_Krumm", LANGUAGE)
LI_PROMPT_PATH = data_loader.load("aig_prompts_Li", LANGUAGE)

DATASET_DIR = op.join('datasets')
RESULT_DIR = op.join('results', 'SJTs')
MUSSEL_SJT_PATH = op.join(DATASET_DIR, 'SJTs', 'Mussel.json')
TRAIT_DEF_PATH = op.join(DATASET_DIR, 'trait_knowledge', 'def_bf.json')


def load_data():
    """Load required datasets and templates."""
    dataloader = src.DataLoader()
    mussel_sjt = dataloader.load("PSJT-Mussel", LANGUAGE)
    trait_def = dataloader.load("_traits_definition", "en")  # Trait definitions

    krumm_aig = TemplateLLM(KRUMM_PROMPT_PATH)
    li_aig = TemplateLLM(LI_PROMPT_PATH)
    
    return mussel_sjt, trait_def, krumm_aig, li_aig

class BaseSJTGenerator:
    """Base class for SJT generators."""
    
    def __init__(self, template: TemplateLLM):
        self.template_llm = template
        self.base_llm = lmitf.BaseLLM()
        self.history = None

    def _build_context_message(self, role: str, content: str) -> Dict[str, str]:
        """Build a message dict for context."""
        return {'role': role, 'content': content}

class KrummGenerator(BaseSJTGenerator):
    """Krumm approach: Iterative SJT generation."""

    def generate(self, trait: str, n_items: int, model: str = MODEL) -> Dict[str, Any]:
        """Generate SJT items iteratively."""
        if LANGUAGE == 'en':
            iter_prompt = (f"Can you create another SJT item measuring {trait}? "
                          "The described situation in the SJT should be different from the ones created before...")
        elif LANGUAGE == 'zh':
            iter_prompt = (f"你能够创建另一个测量{trait}的情境判断测验(SJT)项目吗?"
                          "SJT中描述的情景应与之前创建的情境不同...")

        context = self.template_llm.prompt_template.copy()
        sjts = {trait: {}}
        
        for i in range(n_items):
            if i == 0:
                # First iteration - use base template
                response = self._call_llm(context, model)
            else:
                # Subsequent iterations - add iteration prompt
                context.append(self._build_context_message('user', iter_prompt))
                response = self._call_llm(context, model)
            
            context.append(self._build_context_message('assistant', str(response)))
            sjts[trait][str(i+1)] = response

        self.history = context
        return sjts
    
    def _call_llm(self, context: List[Dict], model: str) -> Any:
        """Call LLM with error handling."""
        return self.base_llm.call(
            messages=context,
            model=model,
            response_format='json',
        )


class LiGenerator(BaseSJTGenerator):
    """Li approach: Batch SJT generation with examples."""

    def generate_reference_items(self, trait: str, dataset: Dict, n_ref: int = 2) -> str:
        """Generate reference items string from dataset."""
        ref_items = []
        for i in range(n_ref):
            item = dataset[trait][str(i+1)]
            ref_items.append({f"Scenario {i+1}": item})
        return str(ref_items)

    def generate(self, trait: str, trait_definition: str, dataset: Dict, n_items: int, 
                model: str = MODEL) -> Dict[str, Any]:
        """Generate SJT items using template approach."""
        sjts = self.template_llm.call(
            Trait=trait,
            TraitDescription=trait_definition,
            Example=self.generate_reference_items(trait.split('-')[0], dataset),
            Nitem=n_items,
            model=model,
            response_format='json',
            temperature=1,
        )
        
        self.history = self.template_llm.prompt_template.copy()
        return {trait: sjts}

async def generate_trait_sjts(trait: str, pbar, mussel_sjt: Dict, trait_def: Dict, 
                              krumm_aig: TemplateLLM, li_aig: TemplateLLM) -> tuple:
    """Generate SJTs for a single trait using both approaches."""
    try:
        # Create separate instances for each trait to avoid conflicts
        krumm = KrummGenerator(krumm_aig)
        li = LiGenerator(li_aig)
        
        # Run both generators concurrently for this trait
        krumm_task = asyncio.to_thread(
            krumm.generate, trait=trait, n_items=N_ITEMS, model=MODEL
        )
        li_task = asyncio.to_thread(
            li.generate, trait=trait, trait_definition=trait_def[trait.split('-')[0]], 
            dataset=mussel_sjt, n_items=N_ITEMS, model=MODEL
        )

        krumm_result, li_result = await asyncio.gather(krumm_task, li_task)
        
        # Update progress bar
        pbar.update(1)
        pbar.set_description(f"Completed {trait}")
        
        return trait, krumm_result, li_result
        
    except Exception as e:
        pbar.write(f"Error processing {trait}: {e}")
        raise
    
async def process_all_traits(mussel_sjt: Dict, trait_def: Dict, 
                           krumm_aig: TemplateLLM, li_aig: TemplateLLM) -> tuple:
    """Process all traits concurrently and return results."""
    krumm_sjt = {}
    li_sjt = {}
    
    # Create progress bar for traits
    with tqdm(total=len(TRAITS), desc="Processing traits") as pbar:
        # Process all traits concurrently
        tasks = [
            generate_trait_sjts(trait, pbar, mussel_sjt, trait_def, krumm_aig, li_aig) 
            for trait in TRAITS
        ]
        results = await asyncio.gather(*tasks)
        
    # Collect results
    for trait_name, krumm_result, li_result in results:
        krumm_sjt.update(krumm_result)
        li_sjt.update(li_result)

    return krumm_sjt, li_sjt
    
def filter_sjt_keys(sjt_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only situation and options from SJT items."""
    filtered = {}
    for trait, items in sjt_dict.items():
        filtered[trait] = {}
        for idx, item in items.items():
            filtered[trait][idx] = {
                "situation": item["situation"],
                "options": item["options"]
            }
    return filtered


def save_results(krumm_sjt: Dict, li_sjt: Dict) -> None:
    """Save SJT results to JSON files."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    krumm_sjt_ = {k.split('-')[0]: v for k, v in krumm_sjt.items()}
    li_sjt_ = {k.split('-')[0]: v for k, v in li_sjt.items()}
    with open(op.join(RESULT_DIR, f'KrummSJT_{LANGUAGE}.json'), 'w') as f:
        json.dump(krumm_sjt_, f, indent=4, ensure_ascii=False)

    with open(op.join(RESULT_DIR, f'LiSJT_{LANGUAGE}.json'), 'w') as f:
        json.dump(li_sjt_, f, indent=4, ensure_ascii=False)

def main() -> None:
    """Main execution function."""
    # Load data and templates
    mussel_sjt, trait_def, krumm_aig, li_aig = load_data()
    
    # Process all traits
    krumm_sjt, li_sjt = asyncio.run(
        process_all_traits(mussel_sjt, trait_def, krumm_aig, li_aig)
    )
    
    # Filter and save results
    krumm_sjt = filter_sjt_keys(krumm_sjt)
    li_sjt = filter_sjt_keys(li_sjt)
    save_results(krumm_sjt, li_sjt)

if __name__ == "__main__":
    
    main()
