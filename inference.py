from env import EmailTriageEnv
from models import AgentAction


def simple_rule_based_agent(subject: str, body: str) -> AgentAction:
    text = f"{subject} {body}".lower()

    priority = "low"
    category = "account"
    team = "support"
    response = "Thank you for contacting us. We are looking into your issue."

    # Priority rules
    if (
        "urgent" in text
        or "immediate" in text
        or "charged twice" in text
        or "charged" in text
        or "crash" in text
        or "crashes" in text
        or "locked" in text
        or "as soon as possible" in text
    ):
        priority = "high"
    elif (
        "help" in text
        or "unable" in text
        or "not delivered" in text
        or "tracking" in text
    ):
        priority = "medium"

    # Category + team + response rules
    # Keep the order carefully arranged

    # 1. Technical issues first
    if "crash" in text or "crashes" in text or "app" in text or "checkout" in text:
        category = "technical"
        team = "tech"
        response = (
            "Thank you for reporting the app crash during checkout. "
            "Our technical team will investigate this checkout issue."
        )

    # 2. Account issues
    elif "password" in text or "login" in text or "account" in text or "locked" in text:
        category = "account"
        team = "support"
        response = (
            "We understand your account issue. "
            "Our support team will help with account access, password reset, and login recovery."
        )

    # 3. Billing issues before shipping
    elif "charged twice" in text or "charged" in text or "refund" in text or "payment" in text:
        category = "billing"
        team = "billing"
        response = (
            "Thank you for contacting billing support. "
            "We will review the charged twice payment issue and help with refund resolution."
        )

    elif "invoice" in text:
        category = "billing"
        team = "billing"
        response = (
            "Thank you for contacting billing. "
            "We will share your invoice copy and help with your billing request."
        )

    # 4. Shipping issues after billing
    elif "order" in text or "delivery" in text or "tracking" in text:
        category = "shipping"
        team = "support"
        response = (
            "We are sorry about the order delivery delay. "
            "Our support team will check your tracking update and assist you."
        )

    return AgentAction(
        priority=priority,
        category=category,
        team=team,
        response=response
    )


def main():
    print("[START] Email Triage Inference")

    env = EmailTriageEnv()

    for i in range(3):
        print(f"\n[STEP] Episode {i + 1}")

        state = env.reset()

        print(f"[STATE] Subject: {state.subject}")
        print(f"[STATE] Body: {state.body}")

        action = simple_rule_based_agent(state.subject, state.body)

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