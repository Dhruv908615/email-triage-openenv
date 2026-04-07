# Email Triage OpenEnv

A simple real-world OpenEnv environment where an AI agent learns to triage incoming emails.

## Features
- Standard environment API:
  - `reset()`
  - `state()`
  - `step()`
- 3 task difficulty levels:
  - easy
  - medium
  - hard
- Reward-based grading from 0.0 to 1.0
- FastAPI app for testing
- Rule-based baseline inference

## Project Structure

```text
email-triage-openenv/
├── app.py
├── env.py
├── models.py
├── tasks.py
├── graders.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── README.md
└── data/
    └── sample_tasks.json