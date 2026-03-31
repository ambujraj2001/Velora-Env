from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ActionType(str, Enum):
    SELECT_SOURCE = "select_source"
    INSPECT_SCHEMA = "inspect_schema"
    GENERATE_SQL = "generate_sql"
    EXECUTE_QUERY = "execute_query"
    REFINE_QUERY = "refine_query"
    GENERATE_INSIGHT = "generate_insight"
    FINISH = "finish"


class Action(BaseModel):
    action_type: ActionType
    source_name: Optional[str] = None
    query_string: Optional[str] = None
    text: Optional[str] = None


class Observation(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    question: str
    available_sources: List[str]
    current_source: Optional[str] = None
    schema_: Optional[Dict[str, List[str]]] = Field(default=None, alias="schema")
    last_result: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    history_summary: List[str] = Field(default_factory=list)
    metrics_summary: Dict[str, Any] = Field(default_factory=dict)
    step_count: int
    max_steps: int


class Reward(BaseModel):
    value: float
    components: Dict[str, float] = Field(default_factory=dict)
    rationale: str


class QueryRecord(BaseModel):
    source_name: str
    query: str
    result: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    cost: float = 0.0
    partial: bool = False
    matched_result_index: Optional[int] = None


class HistoryEntry(BaseModel):
    step: int
    action: Action
    reward: float
    note: str
    metrics_snapshot: Dict[str, Any] = Field(default_factory=dict)


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TaskSpec(BaseModel):
    task_id: str
    difficulty: TaskDifficulty
    question: str
    description: str
    expected_sources: List[str]
    expected_sql: List[str]
    ground_truth_result: List[Dict[str, Any]]
    expected_results: List[List[Dict[str, Any]]] = Field(default_factory=list)
    ground_truth_insight: str
    required_insight_keywords: List[str]
    reasoning_groups: List[List[str]] = Field(default_factory=list)
    max_steps: int = 9


class EnvState(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    task_id: str
    question: str
    available_sources: List[str]
    current_source: Optional[str] = None
    schema_: Optional[Dict[str, List[str]]] = Field(default=None, alias="schema")
    query_result: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    history: List[HistoryEntry] = Field(default_factory=list)
    step_count: int = 0
    max_steps: int = 9
    done: bool = False
    last_sql: Optional[str] = None
    drafted_insight: Optional[str] = None
    selected_sources: List[str] = Field(default_factory=list)
    query_records: List[QueryRecord] = Field(default_factory=list)
    task_metadata: Dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0
    sql_error_count: int = 0
    repeated_error_count: int = 0
    source_switch_count: int = 0
    total_cost: float = 0.0
    last_error_signature: Optional[str] = None
    successful_recoveries: int = 0
    matched_results: List[int] = Field(default_factory=list)
    source_path: List[str] = Field(default_factory=list)


class GradeBreakdown(BaseModel):
    query_correctness: float
    result_accuracy: float
    insight_quality: float
    efficiency: float
    overall: float
    details: Dict[str, Any] = Field(default_factory=dict)


class SourceConfig(BaseModel):
    name: str
    description: str
    tables: Dict[str, List[str]]
    allowed_tables: List[str]
    distractor: bool = False
    row_count_estimate: int = 10


class PolicyDecision(BaseModel):
    action_type: Literal[
        "select_source",
        "inspect_schema",
        "generate_sql",
        "execute_query",
        "refine_query",
        "generate_insight",
        "finish",
    ]
    source_name: Optional[str] = None
    query_string: Optional[str] = None
    text: Optional[str] = None
