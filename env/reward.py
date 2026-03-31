from __future__ import annotations

from typing import Dict, Tuple

from .grader import grade_episode
from .models import Action, ActionType, EnvState, Reward, TaskSpec


def _build_reward(value: float, components: Dict[str, float], rationale: str) -> Reward:
    return Reward(value=round(value, 4), components=components, rationale=rationale)


def compute_step_reward(
    state: EnvState,
    task: TaskSpec,
    action: Action,
    action_success: bool,
    metadata: Dict[str, object],
) -> Reward:
    components: Dict[str, float] = {}

    if action.action_type == ActionType.SELECT_SOURCE:
        if action.source_name in task.expected_sources:
            components["source_selection"] = 2.0
            if metadata.get("switched_from_wrong_source"):
                components["course_correction"] = 1.5
            rationale = f"Selected a relevant source: {action.source_name}."
        else:
            components["source_selection"] = -2.0
            if metadata.get("selected_distractor"):
                components["distractor_penalty"] = -1.0
            rationale = f"Selected an irrelevant source: {action.source_name}."
        return _build_reward(sum(components.values()), components, rationale)

    if action.action_type == ActionType.INSPECT_SCHEMA:
        components["progress"] = 0.5 if state.current_source else -1.0
        rationale = "Schema inspection increases context." if state.current_source else "Inspecting schema without selecting a source is unhelpful."
        return _build_reward(sum(components.values()), components, rationale)

    if action.action_type in {ActionType.GENERATE_SQL, ActionType.REFINE_QUERY}:
        if action_success:
            components["valid_sql"] = 3.0
            cost_penalty = float(metadata.get("draft_cost_penalty", 0.0))
            if cost_penalty:
                components["query_cost"] = -cost_penalty
            if metadata.get("fixed_after_error"):
                components["error_recovery"] = 1.0
            rationale = "Drafted a syntactically valid SQL query."
        else:
            components["sql_error"] = -3.0
            if state.repeated_error_count:
                components["repeat_error_penalty"] = -1.0
            rationale = "Drafted invalid or disallowed SQL."
        return _build_reward(sum(components.values()), components, rationale)

    if action.action_type == ActionType.EXECUTE_QUERY:
        if not action_success:
            components["execution_error"] = -3.0
            rationale = "Query execution failed."
            return _build_reward(sum(components.values()), components, rationale)

        if metadata.get("result_correct"):
            components["query_result"] = 5.0
            rationale = "Query returned a task-relevant result."
        elif metadata.get("partial_result"):
            components["query_result"] = 2.0
            rationale = "Cheap query returned only a partial result."
        else:
            components["query_result"] = 1.0
            rationale = "Query executed, but the result was only partially useful."

        query_cost = float(metadata.get("query_cost", 0.0))
        if metadata.get("result_correct") and query_cost < 3.5:
            components["cost_efficiency"] = 1.0
        elif query_cost >= 4.0:
            components["expensive_query"] = -2.0
            rationale += " The query was more expensive than needed."
        if metadata.get("recovered_error"):
            components["recovery_execution"] = 1.0
        return _build_reward(sum(components.values()), components, rationale)

    if action.action_type == ActionType.GENERATE_INSIGHT:
        keywords_hit = int(metadata.get("keywords_hit", 0))
        keyword_count = max(int(metadata.get("keyword_count", 1)), 1)
        reasoning_groups_hit = int(metadata.get("reasoning_groups_hit", 0))
        reasoning_group_count = max(int(metadata.get("reasoning_group_count", 1)), 1)
        components["insight_progress"] = round(
            (6.0 * (keywords_hit / keyword_count)) + (4.0 * (reasoning_groups_hit / reasoning_group_count)),
            4,
        )
        rationale = "Generated an insight draft grounded in evidence."
        return _build_reward(sum(components.values()), components, rationale)

    if action.action_type == ActionType.FINISH:
        grade = grade_episode(state, task)
        components["completion"] = round(10.0 * grade.overall, 4)
        if not metadata.get("task_complete"):
            components["premature_finish"] = -2.0
            rationale = "Finished before producing a complete answer."
        else:
            rationale = "Finished with a complete answer."
        return _build_reward(sum(components.values()), components, rationale)

    components["unnecessary_step"] = -1.0
    return _build_reward(sum(components.values()), components, "Action did not advance the task.")


def loop_penalty(state: EnvState) -> Tuple[float, str]:
    repeated_sql = len(state.query_records) >= 2 and all(
        record.query == state.query_records[-1].query for record in state.query_records[-2:]
    )
    if repeated_sql:
        return -1.5, "Repeated the same query without improvement."
    if state.repeated_error_count >= 2:
        return -1.0, "Repeated the same SQL mistake."
    if state.step_count > state.max_steps:
        return -2.0, "Exceeded the maximum number of steps."
    return 0.0, ""
