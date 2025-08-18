"""
Graph builder module for constructing the LangGraph workflow.
"""
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from .state import PSJTState
from .trait_analysis import trait_analysis_node
from .situation_construction import situation_construction_node
from .behavior_adaptation import behavior_adaptation_node
from .quality_control import quality_check_node, revise_node, should_revise
from .llm_utils import TDEF


def pack_item(state: PSJTState) -> dict[str, Any]:
    """
    Package the final state into a structured output format.
    
    Args:
        state: Final PSJT state
        
    Returns:
        Structured output dictionary
    """
    return {
        "id": state.get("request_id"),
        "trait": state.get("trait_name"),
        "population_profile": state.get("population_profile"),
        "bfi_item": state.get("bfi_item"),
        "scenario_text": state.get("scenario_text"),
        "options": state.get("options", []),
        "quality": state.get("quality", {}),
        "outline": state.get("situation_outline", {}),
        "trait_analysis": state.get("trait_analysis", {}),
        "meta": {
            "tool": "langgraph-psjt",
            "version": "0.2.0",
            "model": TDEF,
            "iterations": state.get("iter", 0),
        },
    }


def build_graph() -> StateGraph:
    """
    Build and configure the LangGraph workflow.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    graph = StateGraph(PSJTState)
    
    # Add nodes
    graph.add_node("trait_analysis", trait_analysis_node)
    graph.add_node("situation_construction", situation_construction_node)
    graph.add_node("behavior_adaptation", behavior_adaptation_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("revise", revise_node)

    # Set entry point
    graph.set_entry_point("trait_analysis")
    
    # Add sequential edges
    graph.add_edge("trait_analysis", "situation_construction")
    graph.add_edge("situation_construction", "behavior_adaptation")
    graph.add_edge("behavior_adaptation", "quality_check")

    # Add conditional branching for revision
    graph.add_conditional_edges(
        "quality_check",
        should_revise,
        {
            "revise": "revise",
            "stop": END,
        },
    )
    
    # Revision loops back to behavior adaptation for re-evaluation
    graph.add_edge("revise", "behavior_adaptation")
    
    return graph