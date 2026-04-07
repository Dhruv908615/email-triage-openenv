from fastapi import FastAPI
from env import EmailTriageEnv
from models import AgentAction

app = FastAPI(title="Email Triage OpenEnv")

env = EmailTriageEnv()


@app.get("/")
def home():
    return {"message": "Email Triage OpenEnv is running"}


@app.post("/reset")
def reset_env():
    state = env.reset()
    return state.model_dump()


@app.get("/state")
def get_state():
    state = env.state()
    if state is None:
        return {"message": "No active task. Call /reset first."}
    return state.model_dump()


@app.post("/step")
def step_env(action: AgentAction):
    result = env.step(action)
    return result.model_dump()