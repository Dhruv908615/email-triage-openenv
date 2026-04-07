from models import AgentAction


def grade_action(agent_action: AgentAction, expected_task: dict):
    reward = 0.0
    feedback = []

    # Check priority
    if agent_action.priority and agent_action.priority.lower() == expected_task["expected_priority"].lower():
        reward += 0.30
        feedback.append("Priority correct")
    else:
        feedback.append("Priority incorrect")

    # Check category
    if agent_action.category and agent_action.category.lower() == expected_task["expected_category"].lower():
        reward += 0.25
        feedback.append("Category correct")
    else:
        feedback.append("Category incorrect")

    # Check team
    if agent_action.team and agent_action.team.lower() == expected_task["expected_team"].lower():
        reward += 0.25
        feedback.append("Team correct")
    else:
        feedback.append("Team incorrect")

    # Check response quality using keywords
    response_score = 0.0
    if agent_action.response:
        response_text = agent_action.response.lower()
        keywords = expected_task["expected_reply_keywords"]

        matched_keywords = 0
        for keyword in keywords:
            if keyword.lower() in response_text:
                matched_keywords += 1

        if matched_keywords >= 2:
            response_score = 0.20
            feedback.append("Response acceptable")
        else:
            feedback.append("Response too weak")
    else:
        feedback.append("Response missing")

    reward += response_score

    # Keep reward between 0 and 1
    reward = min(max(reward, 0.0), 1.0)

    return reward, "; ".join(feedback)