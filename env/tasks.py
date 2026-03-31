from __future__ import annotations

from .models import TaskDifficulty, TaskSpec


TASKS = [
    TaskSpec(
        task_id="easy_revenue_march_2026",
        difficulty=TaskDifficulty.EASY,
        question="What was total revenue in March 2026?",
        description="Single-table aggregation over the orders dataset.",
        expected_sources=["orders"],
        expected_sql=[
            "SELECT ROUND(SUM(revenue), 2) AS total_revenue FROM orders WHERE order_date >= '2026-03-01' AND order_date < '2026-04-01';",
        ],
        ground_truth_result=[{"total_revenue": 58720.0}],
        expected_results=[[{"total_revenue": 58720.0}]],
        ground_truth_insight="Total revenue in March 2026 was $58,720.00.",
        required_insight_keywords=["58720", "march", "2026", "revenue"],
        reasoning_groups=[["58720"], ["march"], ["revenue"]],
    ),
    TaskSpec(
        task_id="medium_top_customers_q1_2026",
        difficulty=TaskDifficulty.MEDIUM,
        question="Who were the top 5 customers by revenue in Q1 2026?",
        description="Join orders with customers and rank by revenue.",
        expected_sources=["sales_warehouse"],
        expected_sql=[
            "SELECT c.customer_name, ROUND(SUM(o.revenue), 2) AS total_revenue FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE o.order_date >= '2026-01-01' AND o.order_date < '2026-04-01' GROUP BY c.customer_name ORDER BY total_revenue DESC LIMIT 5;",
        ],
        ground_truth_result=[
            {"customer_name": "BrightMart", "total_revenue": 38000.0},
            {"customer_name": "Nova Retail", "total_revenue": 27000.0},
            {"customer_name": "Summit Stores", "total_revenue": 25700.0},
            {"customer_name": "Enterprise Labs", "total_revenue": 21900.0},
            {"customer_name": "Blue Market", "total_revenue": 17200.0},
        ],
        expected_results=[[
            {"customer_name": "BrightMart", "total_revenue": 38000.0},
            {"customer_name": "Nova Retail", "total_revenue": 27000.0},
            {"customer_name": "Summit Stores", "total_revenue": 25700.0},
            {"customer_name": "Enterprise Labs", "total_revenue": 21900.0},
            {"customer_name": "Blue Market", "total_revenue": 17200.0},
        ]],
        ground_truth_insight="BrightMart led Q1 2026 revenue with $38,000.00, followed by Nova Retail and Summit Stores.",
        required_insight_keywords=["brightmart", "38000", "nova retail", "summit stores"],
        reasoning_groups=[["brightmart"], ["nova retail"], ["summit stores"]],
    ),
    TaskSpec(
        task_id="hard_revenue_drop_root_cause_march_2026",
        difficulty=TaskDifficulty.HARD,
        question="Build a root-cause explanation for the March 2026 revenue drop using customer behavior, marketing performance, and operational reliability data.",
        description=(
            "Hard multi-factor diagnosis. The agent must connect the revenue drop to a failed paid social campaign, "
            "customer churn led by Nova Retail, and a checkout outage visible in incident logs."
        ),
        expected_sources=["sales_warehouse", "marketing", "logs"],
        expected_sql=[
            "SELECT strftime('%Y-%m', order_date) AS month, ROUND(SUM(revenue), 2) AS total_revenue FROM orders WHERE order_date >= '2026-02-01' AND order_date < '2026-04-01' GROUP BY month ORDER BY month;",
            "SELECT c.customer_name, ROUND(SUM(CASE WHEN o.order_date >= '2026-02-01' AND o.order_date < '2026-03-01' THEN o.revenue ELSE 0 END), 2) AS feb_revenue, ROUND(SUM(CASE WHEN o.order_date >= '2026-03-01' AND o.order_date < '2026-04-01' THEN o.revenue ELSE 0 END), 2) AS mar_revenue, ROUND(SUM(CASE WHEN o.order_date >= '2026-02-01' AND o.order_date < '2026-03-01' THEN o.revenue ELSE 0 END) - SUM(CASE WHEN o.order_date >= '2026-03-01' AND o.order_date < '2026-04-01' THEN o.revenue ELSE 0 END), 2) AS revenue_drop FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE o.order_date >= '2026-02-01' AND o.order_date < '2026-04-01' GROUP BY c.customer_name HAVING SUM(CASE WHEN o.order_date >= '2026-02-01' AND o.order_date < '2026-03-01' THEN o.revenue ELSE 0 END) > SUM(CASE WHEN o.order_date >= '2026-03-01' AND o.order_date < '2026-04-01' THEN o.revenue ELSE 0 END) ORDER BY revenue_drop DESC LIMIT 3;",
            "SELECT channel, spend, conversions, ROUND(spend / conversions, 2) AS cost_per_conversion FROM marketing WHERE campaign_month = '2026-03-01' ORDER BY cost_per_conversion DESC;",
            "SELECT error_code, severity, COUNT(*) AS incidents, SUM(sessions_impacted) AS sessions_impacted FROM logs WHERE event_date >= '2026-03-01' AND event_date < '2026-04-01' GROUP BY error_code, severity ORDER BY sessions_impacted DESC LIMIT 3;",
        ],
        ground_truth_result=[
            {"month": "2026-02", "total_revenue": 65700.0},
            {"month": "2026-03", "total_revenue": 58720.0},
        ],
        expected_results=[
            [
                {"month": "2026-02", "total_revenue": 65700.0},
                {"month": "2026-03", "total_revenue": 58720.0},
            ],
            [
                {"customer_name": "Nova Retail", "feb_revenue": 15000.0, "mar_revenue": 0.0, "revenue_drop": 15000.0},
                {"customer_name": "Blue Market", "feb_revenue": 8700.0, "mar_revenue": 0.0, "revenue_drop": 8700.0},
                {"customer_name": "Fresh Basket", "feb_revenue": 9700.0, "mar_revenue": 6800.0, "revenue_drop": 2900.0},
            ],
            [
                {"channel": "Paid Social", "spend": 20000.0, "conversions": 45, "cost_per_conversion": 444.44},
                {"channel": "Paid Search", "spend": 17000.0, "conversions": 170, "cost_per_conversion": 100.0},
                {"channel": "Email", "spend": 5800.0, "conversions": 158, "cost_per_conversion": 36.71},
            ],
            [
                {"error_code": "CHK-OUTAGE", "severity": "critical", "incidents": 1, "sessions_impacted": 620},
                {"error_code": "CHK-503", "severity": "error", "incidents": 2, "sessions_impacted": 400},
                {"error_code": "CHK-408", "severity": "warning", "incidents": 1, "sessions_impacted": 90},
            ],
        ],
        ground_truth_insight=(
            "Revenue fell from $65,700.00 in February 2026 to $58,720.00 in March 2026 because the paid social campaign failed, "
            "Nova Retail churned after February, and a March checkout outage blocked conversions."
        ),
        required_insight_keywords=[
            "65700",
            "58720",
            "paid social",
            "campaign",
            "nova retail",
            "outage",
            "checkout",
            "churn",
        ],
        reasoning_groups=[
            ["paid social", "campaign failure", "marketing failure"],
            ["nova retail", "churn", "customer churn"],
            ["outage", "checkout outage", "system outage", "chk-outage"],
        ],
        max_steps=12,
    ),
]


def get_task(task_id: str | None = None, index: int = 0) -> TaskSpec:
    if task_id is not None:
        for task in TASKS:
            if task.task_id == task_id:
                return task
        raise ValueError(f"Unknown task_id: {task_id}")
    return TASKS[index % len(TASKS)]
