"""
LLM utilities and helper functions for the PSJT system.
"""
import json
import re
from typing import Any
from langchain_openai import ChatOpenAI


# Model definition - you can change this to any ChatCompletion compatible model
TDEF = "gpt-4o-mini"
llm = ChatOpenAI(model=TDEF, temperature=0.4)


def json_guard(s: str) -> Any:
    """
    Extract the first JSON object from LLM text to avoid crashes from abnormal formatting.
    
    Args:
        s: Raw text from LLM that should contain JSON
        
    Returns:
        Parsed JSON object
        
    Raises:
        ValueError: If no valid JSON can be extracted
    """
    try:
        return json.loads(s)
    except Exception:
        # Fallback: extract first {...} block
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        raise ValueError("LLM did not return parseable JSON. Raw output:\n" + s)