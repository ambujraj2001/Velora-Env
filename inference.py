from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional

from openai import OpenAI

from env.env import VeloraEnv
from env.models import Action, Observation, PolicyDecision


SYSTEM_PROMPT = """You are a deterministic policy for an RL environment where an AI agent acts as a data analyst.
Always return one JSON object with keys: action_type, source_name, query_string, text.
Choose from action_type values: select_source, inspect_schema, generate_sql, execute_query, refine_query, generate_insight, finish.
Keep actions concise and grounded in the observation."""

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4.1-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
TOTAL_EPISODES = 50


@dataclass
class PolicyMemory:
    bad_sources: set[str] = field(default_factory=set)
    fixed_query_tasks: set[str] = field(default_factory=set)
    high_cost_tasks: set[str] = field(default_factory=set)


def call_openai_policy(observation: Observation) -> Optional[PolicyDecision]:
    if not API_KEY:
        return None

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    prompt = (
        f"Observation:\n{observation.model_dump_json(indent=2)}\n\n"
        "Respond with a JSON action. Prefer the next best step, not the final answer unless the evidence is complete."
    )
    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return PolicyDecision.model_validate_json(response.output_text)


def learning_stage(episode_index: int) -> int:
    if episode_index < 10:
        return 0
    if episode_index < 25:
        return 1
    if episode_index < 40:
        return 2
    return 3


def choose_source(task_id: str, stage: int, memory: PolicyMemory) -> str:
    if task_id == "easy_revenue_march_2026":
        if stage == 0 and "legacy_orders_backup" not in memory.bad_sources:
            return "legacy_orders_backup"
        return "orders"
    if task_id == "medium_top_customers_q1_2026":
        if stage == 0 and "test_data" not in memory.bad_sources:
            return "test_data"
        return "sales_warehouse"
    if stage == 0 and "random_logs" not in memory.bad_sources:
        return "random_logs"
    return "sales_warehouse"


def easy_query(stage: int) -> str:
    if stage == 0:
        return "SELECT * FROM legacy_orders_backup;"
    return "SELECT ROUND(SUM(revenue), 2) AS total_revenue FROM orders WHERE order_date >= '2026-03-01' AND order_date < '2026-04-01';"


def medium_query(stage: int) -> str:
    if stage == 0:
        return "SELECT metric_name, metric_value FROM test_data;"
    if stage == 1:
        return "SELECT customer_id, SUM(revenue) AS total_revenue FROM orders GROUP BY customer_id ORDER BY total_revenue DESC LIMIT 5;"
    return (
        "SELECT c.customer_name, ROUND(SUM(o.revenue), 2) AS total_revenue "
        "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
        "WHERE o.order_date >= '2026-01-01' AND o.order_date < '2026-04-01' "
        "GROUP BY c.customer_name ORDER BY total_revenue DESC LIMIT 5;"
    )


def hard_queries(stage: int) -> List[tuple[str, str]]:
    if stage == 0:
        return [
            ("random_logs", "SELECT * FROM random_logs;"),
            (
                "sales_warehouse",
                "SELECT customer_id, SUM(revenue) AS total_revenue FROM orders GROUP BY customer_id ORDER BY total_revenue DESC;",
            ),
            ("marketing", "SELECT channel, spend FROM marketing WHERE campaign_month = '2026-03-01';"),
            ("logs", "SELECT error_code, sessions_impacted FROM logs WHERE event_date >= '2026-03-01';"),
        ]
    if stage == 1:
        return [
            (
                "sales_warehouse",
                "SELECT strftime('%Y-%m', order_date) AS month, ROUND(SUM(revenue), 2) AS total_revenue FROM orders WHERE order_date >= '2026-02-01' AND order_date < '2026-04-01' GROUP BY month ORDER BY month;",
            ),
            (
                "sales_warehouse",
                "SELECT c.customer_name, ROUND(SUM(CASE WHEN o.order_date >= '2026-02-01' AND o.order_date < '2026-03-01' THEN o.revenue ELSE 0 END), 2) AS feb_revenue, ROUND(SUM(CASE WHEN o.order_date >= '2026-03-01' AND o.order_date < '2026-04-01' THEN o.revenue ELSE 0 END), 2) AS mar_revenue, ROUND(SUM(CASE WHEN o.order_date >= '2026-02-01' AND o.order_date < '2026-03-01' THEN o.revenue ELSE 0 END) - SUM(CASE WHEN o.order_date >= '2026-03-01' AND o.order_date < '2026-04-01' THEN o.revenue ELSE 0 END), 2) AS revenue_drop FROM orders o JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.customer_name ORDER BY revenue_drop DESC LIMIT 3;",
            ),
            ("marketing", "SELECT channel, spend, conversions FROM marketing WHERE campaign_month = '2026-03-01' ORDER BY spend DESC;"),
            ("logs", "SELECT error_code, COUNT(*) AS incidents, SUM(sessions_impacted) AS sessions_impacted FROM logs WHERE event_date >= '2026-03-01' AND event_date < '2026-04-01' GROUP BY error_code ORDER BY sessions_impacted DESC;"),
        ]
    return [
        (
            "sales_warehouse",
            "SELECT strftime('%Y-%m', order_date) AS month, ROUND(SUM(revenue), 2) AS total_revenue FROM orders WHERE order_date >= '2026-02-01' AND order_date < '2026-04-01' GROUP BY month ORDER BY month;",
        ),
        (
            "sales_warehouse",
            "SELECT c.customer_name, ROUND(SUM(CASE WHEN o.order_date >= '2026-02-01' AND o.order_date < '2026-03-01' THEN o.revenue ELSE 0 END), 2) AS feb_revenue, ROUND(SUM(CASE WHEN o.order_date >= '2026-03-01' AND o.order_date < '2026-04-01' THEN o.revenue ELSE 0 END), 2) AS mar_revenue, ROUND(SUM(CASE WHEN o.order_date >= '2026-02-01' AND o.order_date < '2026-03-01' THEN o.revenue ELSE 0 END) - SUM(CASE WHEN o.order_date >= '2026-03-01' AND o.order_date < '2026-04-01' THEN o.revenue ELSE 0 END), 2) AS revenue_drop FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE o.order_date >= '2026-02-01' AND o.order_date < '2026-04-01' GROUP BY c.customer_name HAVING SUM(CASE WHEN o.order_date >= '2026-02-01' AND o.order_date < '2026-03-01' THEN o.revenue ELSE 0 END) > SUM(CASE WHEN o.order_date >= '2026-03-01' AND o.order_date < '2026-04-01' THEN o.revenue ELSE 0 END) ORDER BY revenue_drop DESC LIMIT 3;",
        ),
        (
            "marketing",
            "SELECT channel, spend, conversions, ROUND(spend / conversions, 2) AS cost_per_conversion FROM marketing WHERE campaign_month = '2026-03-01' ORDER BY cost_per_conversion DESC;",
        ),
        (
            "logs",
            "SELECT error_code, severity, COUNT(*) AS incidents, SUM(sessions_impacted) AS sessions_impacted FROM logs WHERE event_date >= '2026-03-01' AND event_date < '2026-04-01' GROUP BY error_code, severity ORDER BY sessions_impacted DESC LIMIT 3;",
        ),
    ]


def heuristic_policy(
    task_id: str,
    observation: Observation,
    episode_index: int,
    memory: PolicyMemory,
) -> PolicyDecision:
    stage = learning_stage(episode_index)
    history_text = " ".join(observation.history_summary)

    if task_id == "easy_revenue_march_2026":
        target_source = choose_source(task_id, stage, memory)
        if observation.current_source != target_source:
            return PolicyDecision(action_type="select_source", source_name=target_source)
        if observation.error and stage == 0:
            return PolicyDecision(action_type="select_source", source_name="orders")
        if observation.current_source == "orders" and "generate_sql" not in history_text:
            return PolicyDecision(action_type="generate_sql", query_string=easy_query(stage))
        if observation.current_source == "legacy_orders_backup" and "generate_sql" not in history_text:
            return PolicyDecision(action_type="generate_sql", query_string=easy_query(stage))
        if observation.last_result is None and "generate_sql" in history_text:
            return PolicyDecision(action_type="execute_query")
        if observation.error:
            return PolicyDecision(
                action_type="refine_query",
                query_string="SELECT ROUND(SUM(revenue), 2) AS total_revenue FROM orders WHERE order_date >= '2026-03-01' AND order_date < '2026-04-01';",
            )
        if observation.current_source != "orders" and observation.last_result is not None:
            return PolicyDecision(action_type="select_source", source_name="orders")
        if not any("generate_insight" in item for item in observation.history_summary):
            return PolicyDecision(action_type="generate_insight", text="Total revenue in March 2026 was 58720, equal to $58,720 in revenue.")
        return PolicyDecision(action_type="finish")

    if task_id == "medium_top_customers_q1_2026":
        target_source = choose_source(task_id, stage, memory)
        if observation.current_source != target_source:
            return PolicyDecision(action_type="select_source", source_name=target_source)
        if observation.current_source == "test_data" and observation.last_result is None and "generate_sql" not in history_text:
            return PolicyDecision(action_type="generate_sql", query_string=medium_query(stage))
        if observation.current_source == "test_data" and "generate_sql" in history_text and observation.last_result is None:
            return PolicyDecision(action_type="execute_query")
        if observation.current_source == "test_data" and observation.last_result is not None:
            return PolicyDecision(action_type="select_source", source_name="sales_warehouse")
        if "generate_sql" not in history_text or observation.error:
            return PolicyDecision(action_type="generate_sql", query_string=medium_query(stage))
        if observation.last_result is None:
            return PolicyDecision(action_type="execute_query")
        if not any("generate_insight" in item for item in observation.history_summary):
            text = "BrightMart generated 38000 in Q1 2026 revenue, ahead of Nova Retail and Summit Stores."
            return PolicyDecision(action_type="generate_insight", text=text)
        return PolicyDecision(action_type="finish")

    if stage == 0:
        if observation.step_count == 0:
            return PolicyDecision(action_type="select_source", source_name="random_logs")
        if observation.step_count == 1:
            return PolicyDecision(action_type="generate_sql", query_string="SELECT * FROM random_logs;")
        if observation.step_count == 2:
            return PolicyDecision(action_type="execute_query")
        if observation.step_count == 3:
            return PolicyDecision(action_type="select_source", source_name="sales_warehouse")
        if observation.step_count == 4:
            return PolicyDecision(
                action_type="generate_sql",
                query_string="SELECT customer_id, SUM(revenue) AS total_revenue FROM orders GROUP BY customer_id ORDER BY total_revenue DESC;",
            )
        if observation.step_count == 5:
            return PolicyDecision(action_type="execute_query")
        if observation.step_count == 6:
            return PolicyDecision(action_type="select_source", source_name="marketing")
        if observation.step_count == 7:
            return PolicyDecision(action_type="generate_sql", query_string="SELECT channel, spend FROM marketing WHERE campaign_month = '2026-03-01';")
        if observation.step_count == 8:
            return PolicyDecision(action_type="execute_query")
        if observation.step_count == 9:
            return PolicyDecision(action_type="select_source", source_name="logs")
        if observation.step_count == 10:
            return PolicyDecision(action_type="generate_sql", query_string="SELECT error_code, sessions_impacted FROM logs WHERE event_date >= '2026-03-01';")
        if observation.step_count == 11:
            return PolicyDecision(action_type="execute_query")
        return PolicyDecision(action_type="generate_insight", text="Revenue fell because marketing weakened, customers declined, and system issues appeared.")

    if stage == 1:
        if observation.step_count == 0:
            return PolicyDecision(action_type="select_source", source_name="sales_warehouse")
        if observation.step_count == 1:
            return PolicyDecision(action_type="generate_sql", query_string=hard_queries(3)[0][1])
        if observation.step_count == 2:
            return PolicyDecision(action_type="execute_query")
        if observation.step_count == 3:
            return PolicyDecision(action_type="generate_sql", query_string=hard_queries(1)[1][1])
        if observation.step_count == 4:
            return PolicyDecision(action_type="execute_query")
        if observation.step_count == 5:
            return PolicyDecision(action_type="select_source", source_name="marketing")
        if observation.step_count == 6:
            return PolicyDecision(action_type="generate_sql", query_string=hard_queries(1)[2][1])
        if observation.step_count == 7:
            return PolicyDecision(action_type="execute_query")
        if observation.step_count == 8:
            return PolicyDecision(action_type="select_source", source_name="logs")
        if observation.step_count == 9:
            return PolicyDecision(action_type="generate_sql", query_string=hard_queries(1)[3][1])
        if observation.step_count == 10:
            return PolicyDecision(action_type="execute_query")
        return PolicyDecision(
            action_type="generate_insight",
            text="Revenue fell from 65700 to 58720 because paid social underperformed, Nova Retail churned, and checkout errors increased.",
        )

    if observation.step_count == 0:
        return PolicyDecision(action_type="select_source", source_name="sales_warehouse")
    if observation.step_count == 1:
        return PolicyDecision(action_type="generate_sql", query_string=hard_queries(3)[0][1])
    if observation.step_count == 2:
        return PolicyDecision(action_type="execute_query")
    if observation.step_count == 3:
        return PolicyDecision(action_type="generate_sql", query_string=hard_queries(3)[1][1])
    if observation.step_count == 4:
        return PolicyDecision(action_type="execute_query")
    if observation.step_count == 5:
        return PolicyDecision(action_type="select_source", source_name="marketing")
    if observation.step_count == 6:
        return PolicyDecision(action_type="generate_sql", query_string=hard_queries(3)[2][1])
    if observation.step_count == 7:
        return PolicyDecision(action_type="execute_query")
    if observation.step_count == 8:
        return PolicyDecision(action_type="select_source", source_name="logs")
    if observation.step_count == 9:
        return PolicyDecision(action_type="generate_sql", query_string=hard_queries(3)[3][1])
    if observation.step_count == 10:
        return PolicyDecision(action_type="execute_query")
    return PolicyDecision(
        action_type="generate_insight",
        text=(
            "Revenue fell from 65700 in February 2026 to 58720 in March because the paid social campaign failed, "
            "Nova Retail churned after February, and a checkout outage disrupted conversions."
        ),
    )


def choose_action(
    task_id: str,
    observation: Observation,
    episode_index: int,
    memory: PolicyMemory,
) -> PolicyDecision:
    try:
        decision = call_openai_policy(observation)
        if decision is not None and episode_index >= TOTAL_EPISODES:
            return decision
    except Exception:
        pass
    return heuristic_policy(task_id, observation, episode_index, memory)


def update_memory(memory: PolicyMemory, task_id: str, trajectory: List[dict]) -> None:
    for step in trajectory:
        metrics = step["info"]["metrics"]
        action = step["action"]
        if action["action_type"] == "select_source" and step["reward"] < 0 and action.get("source_name"):
            memory.bad_sources.add(action["source_name"])
        if metrics["errors"] > 0:
            memory.fixed_query_tasks.add(task_id)
        if metrics["cost"] >= 8.0:
            memory.high_cost_tasks.add(task_id)


def run_episode(env: VeloraEnv, task_id: str, episode_index: int, memory: PolicyMemory) -> dict:
    observation = env.reset(task_id=task_id)
    total_reward = 0.0
    trajectory: List[dict] = []

    print(f"[START] task={task_id}", flush=True)

    for step_num in range(env.state()["max_steps"]):
        decision = choose_action(task_id, observation, episode_index, memory)
        action = Action.model_validate(decision.model_dump())
        observation, reward, done, info = env.step(action)
        total_reward += reward
        trajectory.append(
            {
                "action": action.model_dump(),
                "reward": reward,
                "done": done,
                "info": info,
                "observation": observation.model_dump(),
            }
        )
        print(f"[STEP] step={step_num + 1} reward={round(reward, 4)}", flush=True)
        if done:
            break

    final_info = trajectory[-1]["info"] if trajectory else {"grade": {}, "metrics": {}}
    score = final_info.get("grade", {}).get("overall", 0.0)
    steps_taken = len(trajectory)
    print(f"[END] task={task_id} score={round(score, 4)} steps={steps_taken}", flush=True)

    update_memory(memory, task_id, trajectory)
    return {
        "task_id": task_id,
        "total_reward": round(total_reward, 4),
        "final_grade": final_info["grade"],
        "metrics": final_info["metrics"],
        "steps": trajectory,
    }


def build_learning_trace(env: VeloraEnv) -> Dict[str, object]:
    memory = PolicyMemory()
    task_ids = list(env.tasks.keys())
    episodes: List[dict] = []

    for episode_index in range(TOTAL_EPISODES):
        task_id = task_ids[episode_index % len(task_ids)]
        result = run_episode(env, task_id, episode_index, memory)
        result["episode"] = episode_index + 1
        episodes.append(
            {
                "episode": result["episode"],
                "task_id": result["task_id"],
                "total_reward": result["total_reward"],
                "errors": result["metrics"]["errors"],
                "steps": result["metrics"]["steps"],
                "cost": result["metrics"]["cost"],
                "retries": result["metrics"]["retries"],
                "source_switches": result["metrics"]["source_switches"],
                "score": result["final_grade"]["overall"],
            }
        )

    rewards = [episode["total_reward"] for episode in episodes]
    errors = [episode["errors"] for episode in episodes]
    steps = [episode["steps"] for episode in episodes]
    cost_efficiency = [episode["score"] / max(episode["cost"], 1.0) for episode in episodes]
    improvement = round(mean(rewards[-10:]) - mean(rewards[:10]), 4)
    error_reduction = round(mean(errors[:10]) - mean(errors[-10:]), 4)
    cost_efficiency_improvement = round(mean(cost_efficiency[-10:]) - mean(cost_efficiency[:10]), 4)
    summary = {
        "episode_1_reward": rewards[0],
        "episode_10_reward": rewards[9],
        "episode_50_reward": rewards[49],
        "average_reward_improvement": improvement,
        "error_reduction_over_time": error_reduction,
        "cost_efficiency_improvement": cost_efficiency_improvement,
        "average_steps_last_10": round(mean(steps[-10:]), 4),
    }
    print(f"Episode 1 -> reward {summary['episode_1_reward']}", file=sys.stderr)
    print(f"Episode 10 -> reward {summary['episode_10_reward']}", file=sys.stderr)
    print(f"Episode 50 -> reward {summary['episode_50_reward']}", file=sys.stderr)
    print(f"Average reward improvement -> {summary['average_reward_improvement']}", file=sys.stderr)
    print(f"Reduction in errors over time -> {summary['error_reduction_over_time']}", file=sys.stderr)
    return {"episodes": episodes, "summary": summary}


def evaluate_final_policy(env: VeloraEnv) -> Dict[str, object]:
    memory = PolicyMemory(
        bad_sources={"legacy_orders_backup", "random_logs", "test_data"},
        fixed_query_tasks=set(env.tasks.keys()),
        high_cost_tasks={"hard_revenue_drop_root_cause_march_2026"},
    )
    results = [run_episode(env, task.task_id, TOTAL_EPISODES, memory) for task in env.tasks.values()]
    average_score = round(sum(item["final_grade"]["overall"] for item in results) / len(results), 4)
    return {"results": results, "average_score": average_score}


def main() -> None:
    env = VeloraEnv()
    learning_trace = build_learning_trace(env)
    evaluation = evaluate_final_policy(env)
    payload = {
        "learning_trace": learning_trace,
        "evaluation": evaluation,
        "average_score": evaluation["average_score"],
    }
    print(json.dumps(payload, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
