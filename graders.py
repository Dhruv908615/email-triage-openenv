from models import AgentAction


def bounded_score(score: float) -> float:
    """
    Validator wants score strictly between 0 and 1.
    """
    if score <= 0.0:
        return 0.05
    if score >= 1.0:
        return 0.95
    return round(score, 4)


def grade_priority_only(agent_action: AgentAction, expected_task: dict):
    score = 0.2
    feedback = []

    if agent_action.priority and agent_action.priority.lower() == expected_task["expected_priority"].lower():
        score += 0.6
        feedback.append("Priority correct")
    else:
        feedback.append("Priority incorrect")

    if agent_action.response:
        score += 0.1
        feedback.append("Response present")
    else:
        feedback.append("Response missing")

    return bounded_score(score), "; ".join(feedback)


def grade_routing(agent_action: AgentAction, expected_task: dict):
    score = 0.1
    feedback = []

    if agent_action.priority and agent_action.priority.lower() == expected_task["expected_priority"].lower():
        score += 0.25
        feedback.append("Priority correct")
    else:
        feedback.append("Priority incorrect")

    if agent_action.category and agent_action.category.lower() == expected_task["expected_category"].lower():
        score += 0.3
        feedback.append("Category correct")
    else:
        feedback.append("Category incorrect")

    if agent_action.team and agent_action.team.lower() == expected_task["expected_team"].lower():
        score += 0.25
        feedback.append("Team correct")
    else:
        feedback.append("Team incorrect")

    return bounded_score(score), "; ".join(feedback)


def grade_full_triage(agent_action: AgentAction, expected_task: dict):
    score = 0.1
    feedback = []

    if agent_action.priority and agent_action.priority.lower() == expected_task["expected_priority"].lower():
        score += 0.2
        feedback.append("Priority correct")
    else:
        feedback.append("Priority incorrect")

    if agent_action.category and agent_action.category.lower() == expected_task["expected_category"].lower():
        score += 0.2
        feedback.append("Category correct")
    else:
        feedback.append("Category incorrect")

    if agent_action.team and agent_action.team.lower() == expected_task["expected_team"].lower():
        score += 0.2
        feedback.append("Team correct")
    else:
        feedback.append("Team incorrect")

    response_keywords = expected_task.get("expected_reply_keywords", [])
    matched = 0

    if agent_action.response:
        response_text = agent_action.response.lower()
        for keyword in response_keywords:
            if keyword.lower() in response_text:
                matched += 1

        if matched >= 2:
            score += 0.2
            feedback.append("Response acceptable")
        elif matched == 1:
            score += 0.1
            feedback.append("Response partially acceptable")
        else:
            feedback.append("Response weak")
    else:
        feedback.append("Response missing")

    return bounded_score(score), "; ".join(feedback)


def grade_action(agent_action: AgentAction, expected_task: dict):
    task_type = expected_task.get("task_type", "full_triage")

    if task_type == "priority_only":
        return grade_priority_only(agent_action, expected_task)

    if task_type == "routing":
        return grade_routing(agent_action, expected_task)

    return grade_full_triage(agent_action, expected_task)