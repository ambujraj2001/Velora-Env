from __future__ import annotations

import math
from typing import Any, Dict, List

from .models import EnvState, GradeBreakdown, TaskSpec


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().replace(";", " ;").split())


def _rows_match(actual: List[Dict[str, Any]] | None, expected: List[Dict[str, Any]]) -> bool:
    if actual is None:
        return False
    if len(actual) != len(expected):
        return False
    for actual_row, expected_row in zip(actual, expected):
        if actual_row.keys() != expected_row.keys():
            return False
        for key, value in expected_row.items():
            actual_value = actual_row[key]
            if isinstance(value, float):
                if not math.isclose(float(actual_value), value, rel_tol=1e-9, abs_tol=1e-9):
                    return False
            else:
                if actual_value != value:
                    return False
    return True


def _query_correctness(state: EnvState, task: TaskSpec) -> float:
    expected_queries = {_normalize_query(query) for query in task.expected_sql}
    executed_queries = {
        _normalize_query(record.query)
        for record in state.query_records
        if record.error is None and record.query
    }
    source_hits = len(set(state.selected_sources) & set(task.expected_sources))
    source_score = source_hits / len(task.expected_sources)
    if not expected_queries:
        return source_score
    query_hits = len(expected_queries & executed_queries)
    query_score = query_hits / len(expected_queries)
    return round((0.4 * source_score) + (0.6 * query_score), 4)


def _result_accuracy(state: EnvState, task: TaskSpec) -> float:
    expected_results = task.expected_results or [task.ground_truth_result]
    matched = 0
    for expected in expected_results:
        if any(_rows_match(record.result, expected) for record in state.query_records):
            matched += 1
    return round(matched / len(expected_results), 4)


def _insight_quality(state: EnvState, task: TaskSpec) -> float:
    if not state.drafted_insight:
        return 0.0
    text = state.drafted_insight.lower()
    keyword_hits = sum(1 for keyword in task.required_insight_keywords if keyword.lower() in text)
    keyword_score = keyword_hits / len(task.required_insight_keywords)
    if not task.reasoning_groups:
        return round(keyword_score, 4)
    reasoning_hits = sum(1 for group in task.reasoning_groups if any(keyword.lower() in text for keyword in group))
    reasoning_score = reasoning_hits / len(task.reasoning_groups)
    return round((0.6 * keyword_score) + (0.4 * reasoning_score), 4)


def _efficiency(state: EnvState) -> float:
    max_steps = max(state.max_steps, 1)
    step_component = max(0.0, 1.0 - ((state.step_count - 1) / max_steps))
    error_count = sum(1 for record in state.query_records if record.error)
    expensive_count = sum(1 for record in state.query_records if record.cost >= 4.0)
    partial_count = sum(1 for record in state.query_records if record.partial)
    penalty = min(
        0.9,
        (error_count * 0.12)
        + (expensive_count * 0.08)
        + (partial_count * 0.05)
        + (state.retry_count * 0.03)
        + (state.source_switch_count * 0.02),
    )
    return round(max(0.0, step_component - penalty), 4)


def grade_episode(state: EnvState, task: TaskSpec) -> GradeBreakdown:
    query_correctness = _query_correctness(state, task)
    result_accuracy = _result_accuracy(state, task)
    insight_quality = _insight_quality(state, task)
    efficiency = _efficiency(state)
    overall = round(
        (0.35 * query_correctness)
        + (0.35 * result_accuracy)
        + (0.2 * insight_quality)
        + (0.1 * efficiency),
        4,
    )
    return GradeBreakdown(
        query_correctness=query_correctness,
        result_accuracy=result_accuracy,
        insight_quality=insight_quality,
        efficiency=efficiency,
        overall=overall,
        details={
            "selected_sources": state.selected_sources,
            "steps_taken": state.step_count,
            "queries_executed": len(state.query_records),
            "source_switches": state.source_switch_count,
            "sql_errors": state.sql_error_count,
            "retries": state.retry_count,
            "cost": round(state.total_cost, 2),
        },
    )
