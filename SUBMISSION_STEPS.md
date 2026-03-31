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

## 7. Current Status

These parts are already done:

- project is pushed to GitHub: `https://github.com/ambujraj2001/Velora-Env.git`
- local `openenv validate .` passes
- local `python3 inference.py` runs successfully
- local Docker build works

From here onward, your work is only Hugging Face Space deployment and final submission.

## 8. Create the Hugging Face Space

Go to Hugging Face and create a new Space.

Use:

- Space type: `Docker`
- Space name: your choice
- Visibility: your choice

After the Space is created, copy its git URL.

## 9. Push This Code to the Hugging Face Space

Add the Space as a second remote. Keep GitHub as `origin`.

Example:

```bash
git remote add hf <YOUR_HF_SPACE_GIT_URL>
git push -u hf main
```

If the `hf` remote already exists:

```bash
git push hf main
```

Do not remove your GitHub remote.

## 10. Configure Hugging Face Space Variables

In the Space settings, add these variables/secrets:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional fallback:

- `OPENAI_API_KEY`

Recommended:

- `API_BASE_URL=https://router.huggingface.co/v1`
- `MODEL_NAME=<your chosen model>`
- `HF_TOKEN=<your Hugging Face token>`

## 11. Wait for the Space Build to Finish

After pushing to the Space remote, wait for the Hugging Face build logs to finish.

The Space should come up successfully and expose:

- `GET /`
- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`

## 12. Verify the Live Space Manually

Replace `<SPACE_URL>` with your real Space URL:

```bash
curl <SPACE_URL>/health
curl -X POST <SPACE_URL>/reset -H "content-type: application/json" -d '{"task_id":"easy_revenue_march_2026"}'
```

Expected:

- `/health` returns `{"status":"ok"}`
- `/reset` returns a valid observation JSON payload

## 13. Run the Official Submission Validator

If you have the validator script locally:

```bash
./validate-submission.sh <SPACE_URL> .
```

Or run the hosted version:

```bash
curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh | bash -s -- <SPACE_URL> .
```

This is the main final check before submission.

## 14. Final Checklist Before Submitting

Confirm all of these are true:

- Hugging Face Space builds successfully
- Space returns `200`
- `GET /health` works
- `POST /reset` works
- `openenv validate .` passes locally
- `python3 inference.py` completes locally
- Docker build works locally
- `inference.py` is at the project root
- `openenv.yaml` is at the project root
- README is present and complete
- at least 3 deterministic tasks are available

## 15. Submit

Submit:

- your Hugging Face Space URL
- your GitHub repo URL: `https://github.com/ambujraj2001/Velora-Env.git`

## 16. What To Say in the Submission

Keep the description focused on the strongest differentiators:

- sequential analyst training environment, not one-shot QA
- distractor-source reasoning under uncertainty
- cost-aware SQL evidence gathering
- iterative SQL error recovery
- multi-factor hard task with churn, campaign failure, and outage reasoning
- 50-episode deterministic learning trace showing reward improvement

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
