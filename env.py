import json
import random

from models import EmailState, AgentAction, StepResult
from graders import grade_action


class EmailTriageEnv:
    def __init__(self, task_file="data/sample_tasks.json"):
        self.task_file = task_file
        self.tasks = self.load_tasks()
        self.current_task = None
        self.done = False

    def load_tasks(self):
        with open(self.task_file, "r", encoding="utf-8") as file:
            return json.load(file)

    def reset(self):
        self.current_task = random.choice(self.tasks)
        self.done = False

        return EmailState(
            subject=self.current_task["subject"],
            body=self.current_task["body"]
        )

    def state(self):
        if not self.current_task:
            return None

        return EmailState(
            subject=self.current_task["subject"],
            body=self.current_task["body"]
        )

    def step(self, action: AgentAction):
        if not self.current_task:
            return StepResult(
                reward=0.0,
                done=True,
                feedback="Environment not initialized. Call reset() first."
            )

        if self.done:
            return StepResult(
                reward=0.0,
                done=True,
                feedback="Episode already finished. Call reset() for a new task."
            )

        reward, feedback = grade_action(action, self.current_task)
        self.done = True

        return StepResult(
            reward=reward,
            done=True,
            feedback=feedback
        )