# Velora Env Submission Steps

## 1. Prepare the Project

Make sure you are inside the project root:

```bash
cd "/Users/ambujraj/Documents/Personal Projects/hacktatohn/velora-env"
```

Confirm the required files exist:

- `inference.py`
- `openenv.yaml`
- `Dockerfile`
- `pyproject.toml`
- `uv.lock`
- `server/app.py`
- `README.md`

## 2. Create a Local Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Run Local Validation

Run syntax checks:

```bash
python3 -m py_compile env/*.py inference.py server/app.py
```

Run the baseline inference:

```bash
python3 inference.py
```

You should see the learning trace summary including:

- `Episode 1 -> reward ...`
- `Episode 10 -> reward ...`
- `Episode 50 -> reward ...`

Run OpenEnv validation:

```bash
openenv validate .
```

Expected result:

```text
[OK] : Ready for multi-mode deployment
```

## 4. Test Docker Locally

Build the image:

```bash
docker build -t velora-env .
```

Run the container:

```bash
docker run --rm -p 7860:7860 velora-env
```

In another terminal, verify the app:

```bash
curl http://127.0.0.1:7860/health
curl -X POST http://127.0.0.1:7860/reset -H "content-type: application/json" -d '{"task_id":"easy_revenue_march_2026"}'
```

Stop the container with `Ctrl+C`.

## 5. Create a Hugging Face Space

Go to Hugging Face and create a new Space.

Recommended settings:

- Space type: `Docker`
- Visibility: your choice
- Tag the Space with `openenv` if the UI allows it

## 6. Push the Project to the Space Repo

Initialize git if needed:

```bash
git init
git add .
git commit -m "Prepare Velora Env submission"
```

Connect the Hugging Face Space remote:

```bash
git remote add origin <YOUR_HF_SPACE_GIT_URL>
git branch -M main
git push -u origin main
```

If the remote already exists, just push:

```bash
git push origin main
```

## 7. Configure Space Secrets / Variables

In the Hugging Face Space settings, add:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional local fallback only:

- `OPENAI_API_KEY`

Recommended values:

- `API_BASE_URL=https://router.huggingface.co/v1`
- `MODEL_NAME=<your chosen model>`
- `HF_TOKEN=<your Hugging Face token>`

## 8. Wait for Deployment

After push, wait for the Space build to complete.

Your Space must return `200` and respond on:

- `GET /`
- `GET /health`
- `POST /reset`

## 9. Verify the Live Space

Replace `<SPACE_URL>` with your Space URL:

```bash
curl <SPACE_URL>/health
curl -X POST <SPACE_URL>/reset -H "content-type: application/json" -d '{"task_id":"easy_revenue_march_2026"}'
```

You should get valid JSON responses.

## 10. Run the Official Validation Script

Install prerequisites if needed:

```bash
pip install openenv-core
```

Run the validator against your live Space:

```bash
curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh | bash -s -- <SPACE_URL> .
```

Or if you have the validator locally:

```bash
./validate-submission.sh <SPACE_URL> .
```

## 11. Final Pre-Submission Checklist

Confirm all of these:

- HF Space deploys successfully
- `GET /health` returns `200`
- `POST /reset` works on the live Space
- `openenv validate .` passes
- `docker build -t velora-env .` passes
- `python3 inference.py` completes without error
- `inference.py` is in the root
- `openenv.yaml` is in the root
- there are at least 3 tasks with deterministic grading
- README explains the environment clearly

## 12. Submit

Submit the Hugging Face Space URL and the repo requested by the hackathon form.

## Recommended Submission Notes

When describing the project, emphasize:

- sequential analyst behavior, not one-shot QA
- distractor-source reasoning
- cost-aware evidence gathering
- iterative SQL error recovery
- 50-episode learning trace with improving reward and fewer mistakes

## If Something Fails

If `openenv validate .` fails:

- check `pyproject.toml`
- check `uv.lock`
- check `server/app.py`
- rerun `openenv validate .`

If the Space fails to deploy:

- check Space build logs
- confirm it is a Docker Space
- confirm `Dockerfile` starts `uvicorn server.app:app`

If inference fails on the Space:

- confirm `HF_TOKEN`, `MODEL_NAME`, and `API_BASE_URL` are set
- rerun locally with the same variables
