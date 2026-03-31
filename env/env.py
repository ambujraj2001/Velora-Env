from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel

from .grader import grade_episode
from .models import (
    Action,
    ActionType,
    EnvState,
    HistoryEntry,
    Observation,
    QueryRecord,
    SourceConfig,
    TaskSpec,
)
from .reward import compute_step_reward, loop_penalty
from .tasks import TASKS, get_task


class StepInfo(BaseModel):
    reward_components: Dict[str, float]
    reward_rationale: str
    grade: Dict[str, Any]
    task_id: str
    metrics: Dict[str, Any]


class VeloraEnv:
    def __init__(self, data_dir: str | Path | None = None, task_id: str | None = None, max_steps: int = 12):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = Path(data_dir) if data_dir else self.base_dir / "data"
        self.max_steps = max_steps
        self.sources = self._build_sources()
        self.tasks = {task.task_id: task for task in TASKS}
        self.task = get_task(task_id=task_id, index=0)
        self._task_index = 0
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self._load_data()
        self.state_data: Optional[EnvState] = None

    def _build_sources(self) -> Dict[str, SourceConfig]:
        return {
            "orders": SourceConfig(
                name="orders",
                description="Transactional order records from the commerce platform.",
                tables={"orders": ["order_id", "customer_id", "order_date", "revenue", "product_category", "channel", "region"]},
                allowed_tables=["orders"],
                row_count_estimate=17,
            ),
            "sales_warehouse": SourceConfig(
                name="sales_warehouse",
                description="Curated warehouse view combining orders and customers.",
                tables={
                    "orders": ["order_id", "customer_id", "order_date", "revenue", "product_category", "channel", "region"],
                    "customers": ["customer_id", "customer_name", "segment", "region", "signup_date"],
                },
                allowed_tables=["orders", "customers"],
                row_count_estimate=25,
            ),
            "marketing": SourceConfig(
                name="marketing",
                description="Monthly performance marketing spend and conversion data.",
                tables={"marketing": ["campaign_id", "campaign_month", "channel", "spend", "leads", "conversions"]},
                allowed_tables=["marketing"],
                row_count_estimate=6,
            ),
            "logs": SourceConfig(
                name="logs",
                description="Operational product logs with checkout and platform incidents.",
                tables={"logs": ["event_id", "event_date", "service", "severity", "message", "error_code", "sessions_impacted"]},
                allowed_tables=["logs"],
                row_count_estimate=8,
            ),
            "legacy_orders_backup": SourceConfig(
                name="legacy_orders_backup",
                description="Legacy backup export with stale order records and partial March coverage.",
                tables={"legacy_orders_backup": ["order_id", "customer_id", "order_date", "revenue", "product_category", "channel", "region"]},
                allowed_tables=["legacy_orders_backup"],
                distractor=True,
                row_count_estimate=30,
            ),
            "random_logs": SourceConfig(
                name="random_logs",
                description="Sandbox and QA logs that resemble production incidents but are not business-relevant.",
                tables={"random_logs": ["event_id", "event_date", "service", "severity", "message", "error_code", "sessions_impacted"]},
                allowed_tables=["random_logs"],
                distractor=True,
                row_count_estimate=24,
            ),
            "test_data": SourceConfig(
                name="test_data",
                description="Synthetic analytics fixtures used by internal QA and dashboards.",
                tables={"test_data": ["record_id", "dataset_name", "metric_name", "metric_value", "notes"]},
                allowed_tables=["test_data"],
                distractor=True,
                row_count_estimate=18,
            ),
        }

    def _load_csv(self, filename: str, table_name: str, columns: List[Tuple[str, str]]) -> None:
        self.conn.execute(
            f"CREATE TABLE {table_name} ({', '.join(f'{name} {kind}' for name, kind in columns)})"
        )
        file_path = self.data_dir / filename
        with file_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [tuple(row[column] for column, _ in columns) for row in reader]
        placeholders = ", ".join(["?"] * len(columns))
        self.conn.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", rows)
        self.conn.commit()

    def _load_logs(self) -> None:
        self.conn.execute(
            "CREATE TABLE logs (event_id TEXT, event_date TEXT, service TEXT, severity TEXT, message TEXT, error_code TEXT, sessions_impacted INTEGER)"
        )
        with (self.data_dir / "logs.json").open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        self.conn.executemany(
            "INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["event_id"],
                    row["event_date"],
                    row["service"],
                    row["severity"],
                    row["message"],
                    row["error_code"],
                    row["sessions_impacted"],
                )
                for row in rows
            ],
        )
        self.conn.commit()

    def _load_json_table(self, filename: str, table_name: str, columns: List[Tuple[str, str]]) -> None:
        self.conn.execute(
            f"CREATE TABLE {table_name} ({', '.join(f'{name} {kind}' for name, kind in columns)})"
        )
        with (self.data_dir / filename).open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        self.conn.executemany(
            f"INSERT INTO {table_name} VALUES ({', '.join(['?'] * len(columns))})",
            [tuple(row[column] for column, _ in columns) for row in rows],
        )
        self.conn.commit()

    def _load_data(self) -> None:
        self._load_csv(
            "orders.csv",
            "orders",
            [
                ("order_id", "TEXT"),
                ("customer_id", "TEXT"),
                ("order_date", "TEXT"),
                ("revenue", "REAL"),
                ("product_category", "TEXT"),
                ("channel", "TEXT"),
                ("region", "TEXT"),
            ],
        )
        self._load_csv(
            "customers.csv",
            "customers",
            [
                ("customer_id", "TEXT"),
                ("customer_name", "TEXT"),
                ("segment", "TEXT"),
                ("region", "TEXT"),
                ("signup_date", "TEXT"),
            ],
        )
        self._load_csv(
            "marketing.csv",
            "marketing",
            [
                ("campaign_id", "TEXT"),
                ("campaign_month", "TEXT"),
                ("channel", "TEXT"),
                ("spend", "REAL"),
                ("leads", "INTEGER"),
                ("conversions", "INTEGER"),
            ],
        )
        self._load_logs()
        self._load_csv(
            "legacy_orders_backup.csv",
            "legacy_orders_backup",
            [
                ("order_id", "TEXT"),
                ("customer_id", "TEXT"),
                ("order_date", "TEXT"),
                ("revenue", "REAL"),
                ("product_category", "TEXT"),
                ("channel", "TEXT"),
                ("region", "TEXT"),
            ],
        )
        self._load_json_table(
            "random_logs.json",
            "random_logs",
            [
                ("event_id", "TEXT"),
                ("event_date", "TEXT"),
                ("service", "TEXT"),
                ("severity", "TEXT"),
                ("message", "TEXT"),
                ("error_code", "TEXT"),
                ("sessions_impacted", "INTEGER"),
            ],
        )
        self._load_csv(
            "test_data.csv",
            "test_data",
            [
                ("record_id", "TEXT"),
                ("dataset_name", "TEXT"),
                ("metric_name", "TEXT"),
                ("metric_value", "REAL"),
                ("notes", "TEXT"),
            ],
        )

    def reset(self, task_id: str | None = None) -> Observation:
        if task_id is not None:
            self.task = get_task(task_id=task_id)
        else:
            self.task = TASKS[self._task_index % len(TASKS)]
            self._task_index += 1

        self.state_data = EnvState(
            task_id=self.task.task_id,
            question=self.task.question,
            available_sources=list(self.sources.keys()),
            max_steps=min(self.max_steps, self.task.max_steps),
            task_metadata={"description": self.task.description, "difficulty": self.task.difficulty.value},
        )
        return self._observation()

    def state(self) -> Dict[str, Any]:
        if self.state_data is None:
            raise RuntimeError("Environment must be reset before calling state().")
        return self.state_data.model_dump()

    def _observation(self) -> Observation:
        assert self.state_data is not None
        return Observation(
            question=self.state_data.question,
            available_sources=self.state_data.available_sources,
            current_source=self.state_data.current_source,
            schema_=self.state_data.schema_,
            last_result=self.state_data.query_result,
            error=self.state_data.error,
            history_summary=[
                f"step {entry.step}: {entry.action.action_type.value}"
                + (f" ({entry.action.source_name})" if entry.action.source_name else "")
                + f" -> {entry.note}"
                for entry in self.state_data.history[-4:]
            ],
            metrics_summary={
                "source_switches": self.state_data.source_switch_count,
                "sql_errors": self.state_data.sql_error_count,
                "retries": self.state_data.retry_count,
                "total_cost": round(self.state_data.total_cost, 2),
                "recoveries": self.state_data.successful_recoveries,
            },
            step_count=self.state_data.step_count,
            max_steps=self.state_data.max_steps,
        )

    def _validate_query(self, query: str, source_name: str) -> Tuple[bool, Optional[str], float]:
        normalized = query.lower()
        if not normalized.strip().startswith("select"):
            return False, "Only SELECT statements are allowed.", 0.0
        if ";" in normalized.strip()[:-1]:
            return False, "Only single SELECT statements are allowed.", 0.0
        if re.search(r"(^|\s)(drop|delete|update|insert|alter)\s", normalized) and not normalized.strip().startswith("select"):
            return False, "Mutation statements are not allowed.", 0.0

        allowed_tables = self.sources[source_name].allowed_tables
        for table_name in ["orders", "customers", "marketing", "logs"]:
            if table_name in normalized and table_name not in allowed_tables:
                return False, f"Table '{table_name}' is not available in source '{source_name}'.", 0.0

        source = self.sources[source_name]
        join_count = normalized.count(" join ")
        cost = 0.75 + (source.row_count_estimate / 20.0)
        if "*" in normalized:
            cost += 1.5
        cost += join_count * 1.0
        if " where " not in normalized:
            cost += 1.25
        if " group by " in normalized:
            cost += 0.5
        if " order by " in normalized:
            cost += 0.25
        if " limit " not in normalized and source.row_count_estimate >= 10:
            cost += 0.5
        return True, None, cost

    def _rows_to_dicts(self, rows: Iterable[sqlite3.Row]) -> List[Dict[str, Any]]:
        return [dict(row) for row in rows]

    def _expected_results(self) -> List[List[Dict[str, Any]]]:
        return self.task.expected_results or [self.task.ground_truth_result]

    def _matching_result_index(self, rows: List[Dict[str, Any]]) -> Optional[int]:
        for index, expected in enumerate(self._expected_results()):
            if rows == expected:
                return index
        return None

    def _result_correct_for_task(self, rows: List[Dict[str, Any]]) -> bool:
        return self._matching_result_index(rows) is not None

    def _insight_keyword_hits(self, text: str) -> int:
        lowered = text.lower()
        return sum(1 for keyword in self.task.required_insight_keywords if keyword.lower() in lowered)

    def _reasoning_groups_hit(self, text: str) -> int:
        lowered = text.lower()
        return sum(1 for group in self.task.reasoning_groups if any(keyword.lower() in lowered for keyword in group))

    def _degrade_result_if_cheap(
        self,
        rows: List[Dict[str, Any]],
        query: str,
        source_name: str,
        cost: float,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        if not rows:
            return rows, False
        normalized = " ".join(query.lower().split())
        is_expected_query = any(" ".join(expected.lower().split()) == normalized for expected in self.task.expected_sql)
        if is_expected_query or cost >= 2.8:
            return rows, False
        if self.sources[source_name].row_count_estimate < 10:
            return rows, False
        cutoff = max(1, len(rows) // 2)
        return rows[:cutoff], True

    def _metrics_snapshot(self) -> Dict[str, Any]:
        assert self.state_data is not None
        return {
            "cost": round(self.state_data.total_cost, 2),
            "errors": self.state_data.sql_error_count,
            "steps": self.state_data.step_count,
            "retries": self.state_data.retry_count,
            "source_switches": self.state_data.source_switch_count,
            "recoveries": self.state_data.successful_recoveries,
        }

    def step(self, action: Action | Dict[str, Any]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.state_data is None:
            raise RuntimeError("Environment must be reset before calling step().")
        if self.state_data.done:
            raise RuntimeError("Episode already finished. Call reset() to start a new episode.")

        action = action if isinstance(action, Action) else Action.model_validate(action)
        self.state_data.step_count += 1
        action_success = True
        metadata: Dict[str, Any] = {}
        note = ""

        if action.action_type == ActionType.SELECT_SOURCE:
            if action.source_name not in self.sources:
                action_success = False
                self.state_data.error = f"Unknown source: {action.source_name}"
                note = self.state_data.error
            else:
                previous_source = self.state_data.current_source
                self.state_data.current_source = action.source_name
                self.state_data.schema_ = None
                self.state_data.query_result = None
                self.state_data.last_sql = None
                self.state_data.error = None
                if previous_source and previous_source != action.source_name:
                    self.state_data.source_switch_count += 1
                if not self.state_data.source_path or self.state_data.source_path[-1] != action.source_name:
                    self.state_data.source_path.append(action.source_name)
                if action.source_name not in self.state_data.selected_sources:
                    self.state_data.selected_sources.append(action.source_name)
                metadata["switched_from_wrong_source"] = bool(
                    previous_source
                    and previous_source not in self.task.expected_sources
                    and action.source_name in self.task.expected_sources
                )
                metadata["selected_distractor"] = self.sources[action.source_name].distractor
                note = f"Selected source {action.source_name}."

        elif action.action_type == ActionType.INSPECT_SCHEMA:
            if not self.state_data.current_source:
                action_success = False
                self.state_data.error = "Select a source before inspecting its schema."
                note = self.state_data.error
            else:
                self.state_data.schema_ = self.sources[self.state_data.current_source].tables
                self.state_data.error = None
                note = f"Inspected schema for {self.state_data.current_source}."

        elif action.action_type in {ActionType.GENERATE_SQL, ActionType.REFINE_QUERY}:
            if not self.state_data.current_source:
                action_success = False
                self.state_data.error = "Select a source before drafting SQL."
                note = self.state_data.error
            elif not action.query_string:
                action_success = False
                self.state_data.error = "A query_string is required for SQL actions."
                note = self.state_data.error
            else:
                is_valid, error, cost = self._validate_query(action.query_string, self.state_data.current_source)
                if not is_valid:
                    action_success = False
                    self.state_data.error = error
                    self.state_data.sql_error_count += 1
                    self.state_data.retry_count += 1
                    signature = f"{self.state_data.current_source}:{action.query_string}:{error}"
                    if self.state_data.last_error_signature == signature:
                        self.state_data.repeated_error_count += 1
                    self.state_data.last_error_signature = signature
                    note = error or "Invalid query."
                else:
                    self.state_data.last_sql = action.query_string
                    self.state_data.error = None
                    metadata["draft_cost_penalty"] = 1.0 if cost >= 4.0 else 0.0
                    metadata["fixed_after_error"] = self.state_data.last_error_signature is not None
                    note = "Stored SQL draft."

        elif action.action_type == ActionType.EXECUTE_QUERY:
            if not self.state_data.current_source:
                action_success = False
                self.state_data.error = "Select a source before executing SQL."
                note = self.state_data.error
            elif not self.state_data.last_sql:
                action_success = False
                self.state_data.error = "Generate SQL before executing it."
                note = self.state_data.error
            else:
                is_valid, error, cost = self._validate_query(self.state_data.last_sql, self.state_data.current_source)
                if not is_valid:
                    action_success = False
                    self.state_data.error = error
                    note = error or "Invalid query."
                else:
                    try:
                        raw_rows = self._rows_to_dicts(self.conn.execute(self.state_data.last_sql).fetchall())
                        rows, partial = self._degrade_result_if_cheap(
                            raw_rows,
                            self.state_data.last_sql,
                            self.state_data.current_source,
                            cost,
                        )
                        matched_result_index = self._matching_result_index(rows)
                        self.state_data.query_result = rows
                        self.state_data.error = None
                        metadata["query_cost"] = cost
                        metadata["partial_result"] = partial
                        metadata["result_correct"] = matched_result_index is not None
                        metadata["matched_result_index"] = matched_result_index
                        metadata["recovered_error"] = self.state_data.last_error_signature is not None
                        if matched_result_index is not None and matched_result_index not in self.state_data.matched_results:
                            self.state_data.matched_results.append(matched_result_index)
                        self.state_data.total_cost += cost
                        self.state_data.query_records.append(
                            QueryRecord(
                                source_name=self.state_data.current_source,
                                query=self.state_data.last_sql,
                                result=rows,
                                cost=cost,
                                partial=partial,
                                matched_result_index=matched_result_index,
                            )
                        )
                        if self.state_data.last_error_signature is not None and matched_result_index is not None:
                            self.state_data.successful_recoveries += 1
                            self.state_data.last_error_signature = None
                        note = f"Executed SQL and returned {len(rows)} rows."
                    except sqlite3.Error as exc:
                        action_success = False
                        self.state_data.query_result = None
                        self.state_data.error = str(exc)
                        self.state_data.sql_error_count += 1
                        self.state_data.retry_count += 1
                        signature = f"{self.state_data.current_source}:{self.state_data.last_sql}:{exc}"
                        if self.state_data.last_error_signature == signature:
                            self.state_data.repeated_error_count += 1
                        self.state_data.last_error_signature = signature
                        self.state_data.query_records.append(
                            QueryRecord(
                                source_name=self.state_data.current_source,
                                query=self.state_data.last_sql,
                                error=str(exc),
                                cost=cost,
                            )
                        )
                        note = self.state_data.error

        elif action.action_type == ActionType.GENERATE_INSIGHT:
            if not action.text:
                action_success = False
                self.state_data.error = "Insight text is required."
                note = self.state_data.error
            else:
                self.state_data.drafted_insight = action.text
                self.state_data.error = None
                metadata["keywords_hit"] = self._insight_keyword_hits(action.text)
                metadata["keyword_count"] = len(self.task.required_insight_keywords)
                metadata["reasoning_groups_hit"] = self._reasoning_groups_hit(action.text)
                metadata["reasoning_group_count"] = len(self.task.reasoning_groups) or 1
                if metadata["reasoning_groups_hit"] == metadata["reasoning_group_count"] and len(self.state_data.matched_results) == len(self._expected_results()):
                    self.state_data.done = True
                note = "Stored insight draft."

        elif action.action_type == ActionType.FINISH:
            grade = grade_episode(self.state_data, self.task)
            metadata["task_complete"] = bool(grade.result_accuracy >= 0.75 and grade.insight_quality >= 0.7)
            self.state_data.done = True
            note = "Episode finished."

        reward = compute_step_reward(self.state_data, self.task, action, action_success, metadata)
        extra_penalty, penalty_reason = loop_penalty(self.state_data)
        reward_value = reward.value + extra_penalty
        rationale = reward.rationale
        components = dict(reward.components)
        if extra_penalty:
            components["loop_penalty"] = extra_penalty
            rationale = f"{rationale} {penalty_reason}".strip()

        if self.state_data.step_count >= self.state_data.max_steps and not self.state_data.done:
            self.state_data.done = True
            rationale = f"{rationale} Reached the maximum number of steps.".strip()

        self.state_data.history.append(
            HistoryEntry(
                step=self.state_data.step_count,
                action=action,
                reward=round(reward_value, 4),
                note=note,
                metrics_snapshot=self._metrics_snapshot(),
            )
        )

        grade = grade_episode(self.state_data, self.task)
        info = StepInfo(
            reward_components=components,
            reward_rationale=rationale,
            grade=grade.model_dump(),
            task_id=self.task.task_id,
            metrics=self._metrics_snapshot(),
        ).model_dump()
        return self._observation(), round(reward_value, 4), self.state_data.done, info
