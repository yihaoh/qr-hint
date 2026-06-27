# path: backend/models/query.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class QueryRequest:
    """Model for query request"""
    query: str
    database: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

    def validate(self) -> bool:
        """Validate query request"""
        if not self.query or not isinstance(self.query, str):
            return False
        if self.query.strip() == '':
            return False
        return True


@dataclass
class QueryResponse:
    """Model for query response"""
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class HintRequest:
    """Model for hint generation request"""
    query: str
    database: Optional[str] = None
    optimization_level: Optional[int] = 1
    options: Optional[Dict[str, Any]] = None


@dataclass
class HintResponse:
    """Model for hint response"""
    hint: str
    confidence: float
    suggestions: List[str]
    estimated_improvement: Optional[str] = None


@dataclass
class QueryPlan:
    """Model for query execution plan"""
    query: str
    plan: Dict[str, Any]
    cost: float
    execution_time: Optional[float] = None


@dataclass
class OptimizationResult:
    """Model for optimization result"""
    original_query: str
    optimized_query: str
    improvement: str
    hint_applied: Optional[str] = None
    before_cost: Optional[float] = None
    after_cost: Optional[float] = None
