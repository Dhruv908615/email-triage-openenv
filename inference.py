import os
import json

from openai import OpenAI

from env import EmailTriageEnv
from models import AgentAction


def llm_agent(subject: str, body: str) -> AgentAction:
    api_base_url = os.environ["API_BASE_URL"]
    api_key = os.environ["API_KEY"]
    model_name = os.environ.get("MODEL_NAME", "openai/gpt-4o-mini")

    client = OpenAI(
        base_url=api_base_url,
        api_key=api_key,
    )

    prompt = f"""
You are an email triage assistant.

Read the email and return ONLY valid JSON with these keys:
priority, category, team, response

Allowed values:
- priority: low, medium, high
- category: billing, account, technical, shipping
- team: billing, support, tech

Email Subject: {subject}
Email Body: {body}
""".strip()

    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON. Do not add markdown or explanation."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    content = response.choices[0].message.content.strip()
    data = json.loads(content)

    return AgentAction(
        priority=data.get("priority"),
        category=data.get("category"),
        team=data.get("team"),
        response=data.get("response"),
    )


def main():
    print("[START] Email Triage Inference")

    env = EmailTriageEnv()

    for i in range(3):
        print(f"\n[STEP] Episode {i + 1}")

        state = env.reset()

        print(f"[STATE] Subject: {state.subject}")
        print(f"[STATE] Body: {state.body}")

        action = llm_agent(state.subject, state.body)

        print("[ACTION]")
        print(f"priority={action.priority}")
        print(f"category={action.category}")
        print(f"team={action.team}")
        print(f"response={action.response}")

        result = env.step(action)

        print("[RESULT]")
        print(f"reward={result.reward}")
        print(f"done={result.done}")
        print(f"feedback={result.feedback}")

    print("\n[END] Inference completed")


if __name__ == "__main__":
    main()