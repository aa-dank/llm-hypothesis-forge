# rag_orchestration/models.py

from pydantic import BaseModel
from typing import List

"""
rag_orchestration.models

Defines Pydantic models for validating LLM-generated query plans.
"""

class ServiceQuery(BaseModel):
    """
    Represents a set of search queries generated for a specific retrieval service.

    Attributes:
        service (str):
            The name of the retrieval service (must match a ServiceQueryInfo.name).
        queries (List[str]):
            A list of query strings to run against the service.
    """
    service: str
    queries: List[str]

class LLMQueryResponse(BaseModel):
    """
    Schema for validating the LLM's response containing queries for multiple services.

    Attributes:
        queries (List[ServiceQuery]):
            A list of ServiceQuery objects, each specifying
            one service and its associated search queries.
    """
    queries: List[ServiceQuery]
