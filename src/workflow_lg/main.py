"""
Main module for the Personal Situation Judgment Test (PSJT) generation system.

This module coordinates all components to generate situation-based personality assessment items.
"""

import os
import json
import uuid
from typing import Dict, Any

from .graph_builder import build_graph, pack_item
from .state import PSJTState



def generate_psjt_item(
    trait_name: str,
    bfi_item: str,
    population_profile: str = "中国在读大学生",
    language: str = "zh"
) -> dict[str, Any]:
    """
    Generate a single PSJT item for the given personality trait.
    
    Args:
        trait_name: Target personality trait (e.g., "尽责性-条理性")
        bfi_item: Original self-report item text
        population_profile: Target population description
        language: Output language ("zh" or "en")
        
    Returns:
        Generated PSJT item with scenario, options, and quality metrics
    """
    graph = build_graph().compile()
    
    state: PSJTState = {
        "request_id": str(uuid.uuid4()),
        "trait_name": trait_name,
        "bfi_item": bfi_item,
        "population_profile": population_profile,
        "language": language,
        "iter": 0,
    }
    
    final_state = graph.invoke(state)
    return pack_item(final_state)


