from pydantic import BaseModel
from typing import Optional


# Represents the email input (state)
class EmailState(BaseModel):
    subject: str
    body: str


# Represents the agent's action (what AI decides)
class AgentAction(BaseModel):
    priority: Optional[str] = None
    category: Optional[str] = None
    team: Optional[str] = None
    response: Optional[str] = None


# Represents the result after a step
class StepResult(BaseModel):
    reward: float
    done: bool
    feedback: str