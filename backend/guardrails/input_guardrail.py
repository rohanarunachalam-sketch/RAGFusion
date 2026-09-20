# ============================================================
# INPUT GUARDRAIL
# ============================================================

MAX_QUESTION_LENGTH = 1000


# Common prompt-injection patterns.
# This is intentionally a lightweight first layer.
BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore the instructions above",
    "forget previous instructions",
    "forget all previous instructions",
    "disregard previous instructions",
    "disregard all previous instructions",
    "system prompt",
    "reveal your prompt",
    "show me your prompt",
    "developer message",
    "reveal hidden instructions",
]


def validate_question(question: str):
    """
    Validate a user question before sending it
    into the RAG pipeline.

    Returns:

        {
            "allowed": True/False,
            "message": "..."
        }
    """

    # --------------------------------------------------------
    # TYPE CHECK
    # --------------------------------------------------------

    if not isinstance(question, str):

        return {
            "allowed": False,
            "message": "Invalid question."
        }


    # --------------------------------------------------------
    # EMPTY QUESTION
    # --------------------------------------------------------

    question = question.strip()

    if not question:

        return {
            "allowed": False,
            "message": "Please enter a question."
        }


    # --------------------------------------------------------
    # LENGTH CHECK
    # --------------------------------------------------------

    if len(question) > MAX_QUESTION_LENGTH:

        return {
            "allowed": False,
            "message": (
                f"Question is too long. "
                f"Please keep it under "
                f"{MAX_QUESTION_LENGTH} characters."
            )
        }


    # --------------------------------------------------------
    # PROMPT INJECTION CHECK
    # --------------------------------------------------------

    question_lower = question.lower()

    for pattern in BLOCKED_PATTERNS:

        if pattern in question_lower:

            return {
                "allowed": False,
                "message": (
                    "This request contains an instruction "
                    "that cannot be processed."
                )
            }


    # --------------------------------------------------------
    # QUESTION IS SAFE
    # --------------------------------------------------------

    return {
        "allowed": True,
        "message": "Question passed input validation."
    }