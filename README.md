---
title: Velora Env
emoji: "📊"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# Velora Env

Velora Env is a training environment where AI agents learn to behave like real data analysts by making sequential decisions under uncertainty, cost constraints, and noisy data conditions.

Velora Env is an OpenEnv environment for training and evaluating data-analysis agents in a realistic enterprise setting. The agent is not asked for a one-shot answer. It must navigate a noisy data ecosystem, distinguish trusted sources from distractors, trade off SQL cost against evidence quality, recover from mistakes, and construct a grounded business explanation over multiple steps.

This project is designed for the Meta PyTorch OpenEnv Hackathon, but the environment is framed as a reusable benchmark for intelligent data agents: source routing under uncertainty, iterative query repair, and evidence-based root-cause analysis.

## Environment Diagram

```text
Business Question
      |
      v
Available Sources
(trusted + distractors)
      |
      v
Select Source -> Inspect Schema -> Draft SQL -> Execute Query
      |                 |               |              |
      |                 |               |              v
      |                 |               |         Result / Error
      |                 |               |              |
      |                 |               <---- Refine Query
      |                 |
      v                 v
Cost Tracking      Error Tracking
      \                 /
       \               /
        v             v
      Generate Insight
             |
             v
           Finish
             |
             v
Deterministic Grade + Dense Reward + Learning Metrics
```

## Environment Motivation

Real analysts do not operate over clean, single-table benchmarks. They move through warehouse views, logs, backups, staging exports, and half-trusted internal datasets. Good agents must learn more than SQL syntax:

- when to distrust a source that looks plausible
- when to pay more query cost for higher-fidelity evidence
- how to recover after a bad query or a misleading first move
- how to assemble a multi-cause explanation from fragmented signals

Velora Env turns that workflow into a deterministic RL environment with dense, interpretable rewards.

## Environment API

- `reset()` returns the initial typed `Observation`
- `step(action)` returns `(observation, reward, done, info)`
- `state()` returns the full internal environment state

The environment preserves strict OpenEnv semantics and uses typed Pydantic models for `Observation`, `Action`, and `Reward`.

## Action Space

- `select_source(source_name)`
- `inspect_schema()`
- `generate_sql(query_string)`
- `execute_query()`
- `refine_query(query_string)`
- `generate_insight(text)`
- `finish()`

The action space is intentionally analyst-like rather than chatbot-like. Agents must choose when to inspect, when to query, when to retry, and when enough evidence exists to conclude.

## Observation Space

Each observation contains:

- `question`
- `available_sources`
- `current_source`
- `schema`
- `last_result`
- `error`
- `history_summary`
- `metrics_summary`
- `step_count`
- `max_steps`

`metrics_summary` exposes the learning-relevant control signals: source switches, SQL errors, retries, cumulative cost, and successful recoveries.

## Data Sources

Trusted sources:

- `orders`: raw transactional data
- `sales_warehouse`: curated warehouse source combining `orders` and `customers`
- `marketing`: monthly campaign performance
- `logs`: production incident data

Distractor sources:

- `legacy_orders_backup`: stale order export with partial March coverage
- `random_logs`: sandbox and QA incidents that resemble production failures
- `test_data`: synthetic fixtures with plausible but wrong business metrics

These distractors are not cosmetic. They are built to induce realistic failure modes in source selection.

## Tasks

1. Easy: compute total revenue in March 2026 from the correct production orders source.
2. Medium: identify the top 5 customers by Q1 2026 revenue using the warehouse join path.
3. Hard: produce a root-cause explanation for the March 2026 revenue drop by integrating:
   paid social campaign failure,
   customer churn led by Nova Retail,
   checkout outage evidence from logs.

The hard task is deliberately multi-factor. An agent can no longer succeed with a single query or a vague summary.

## Reward Design

Reward shaping is dense and trajectory-aware:

- positive reward for selecting relevant sources
- extra reward for correcting a bad source choice
- reward for valid SQL, correct evidence, and grounded insight generation
- reward for successful error recovery after a failed query
- penalties for distractor selection, repeated mistakes, loops, and unnecessarily expensive scans

Query cost is deterministic and depends on:

- `SELECT *`
- join count
- missing `WHERE`
- missing `LIMIT` on larger sources
- source scan size

The environment also simulates fidelity tradeoffs: cheap exploratory queries can return partial evidence, while more expensive queries can recover full, task-complete evidence. This creates a meaningful cost-versus-accuracy decision boundary.

## Deterministic Grading

Each task is graded from `0.0` to `1.0` using:

- query correctness
- result accuracy across all required evidence sets
- insight quality from evidence keywords and reasoning-group coverage
- efficiency from steps, retries, source switches, SQL errors, partial scans, and total cost

The hard-task grader checks for all three causal factors, not just a generic explanation.

## Learning Behavior

The baseline `inference.py` now runs a deterministic 50-episode learning trace before final evaluation. It logs:

- total reward per episode
- error count
- steps used
- cumulative query cost
- retries
- source switches

The baseline demonstrates policy improvement over time by learning to:

- avoid distractor sources
- stop repeating broken SQL patterns
- spend query cost only when higher-fidelity evidence is needed
- recover from early poor strategies and converge to grounded workflows

Current deterministic learning-trace summary:

- Episode 1 reward: `-8.5`
- Episode 10 reward: `14.876`
- Episode 50 reward: `20.0`
- Average reward improvement: `18.046`
- Error reduction over time: `0.5`
- Cost-efficiency improvement: `0.1969`

## Emergent Behaviors

Velora Env is designed to produce recognizable agent behaviors rather than fixed script execution:

- agents learn to stop trusting stale backups and synthetic fixtures
- agents learn that cheap scans are good for exploration but insufficient for final explanations
- agents learn to pay higher SQL cost only on the hard task when richer evidence is required
- agents learn to reduce retries and repeated SQL failures over time

## What Makes This Environment Novel

- Source selection under uncertainty: several sources are plausible but wrong.
- Cost-aware evidence gathering: low-cost queries can be insufficient even when syntactically valid.
- Iterative error recovery: query repair is part of the task, not an edge case.
- Multi-cause business reasoning: the hard task requires combining churn, campaign failure, and outage evidence into one explanation.

This makes the environment feel closer to analyst training than to benchmark QA.

## Project Structure

```text
velora-env/
├── data/
│   ├── customers.csv
│   ├── legacy_orders_backup.csv
│   ├── logs.json
│   ├── marketing.csv
│   ├── orders.csv
│   ├── random_logs.json
│   └── test_data.csv
├── env/
│   ├── env.py
│   ├── grader.py
│   ├── models.py
│   ├── reward.py
│   └── tasks.py
├── server/
│   └── app.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## Setup

```bash
cd velora-env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Inference

```bash
cd velora-env
python3 inference.py
```

Environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `OPENAI_API_KEY` as an optional local fallback

The script uses the OpenAI client when credentials are present and otherwise falls back to the deterministic baseline policy so scores remain reproducible.

## Hugging Face Space Deployment

The project exposes a FastAPI application in [`server/app.py`](/Users/ambujraj/Documents/Personal%20Projects/hacktatohn/velora-env/server/app.py). Main endpoints:

- `GET /`
- `GET /health`
- `GET /metadata`
- `POST /reset`
- `POST /step`
- `GET /state`

Run locally:

```bash
cd velora-env
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Validation

Validated locally with:

- `openenv validate .`
- `python3 inference.py`
- `docker build -t velora-env .`
- Docker health and reset endpoint checks

## Baseline Scores

Current final evaluation from `inference.py`:

- easy: `0.9667`
- medium: `0.9667`
- hard: `0.9043`
- average: `0.9459`

These scores are deterministic under the default baseline policy.
