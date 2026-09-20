# ============================================================
# OUTPUT GUARDRAIL
# ============================================================

MAX_ANSWER_LENGTH = 8000


def validate_answer(answer: str):
    """
    Validate the generated answer before displaying it
    to the user.
    """

    # --------------------------------------------------------
    # TYPE CHECK
    # --------------------------------------------------------

    if not isinstance(answer, str):

        return {
            "allowed": False,
            "message": "Invalid model response.",
            "answer": ""
        }


    answer = answer.strip()


    # --------------------------------------------------------
    # EMPTY RESPONSE
    # --------------------------------------------------------

    if not answer:

        return {
            "allowed": False,
            "message": (
                "I couldn't generate an answer "
                "from the available document."
            ),
            "answer": ""
        }


    # --------------------------------------------------------
    # LENGTH CHECK
    # --------------------------------------------------------

    if len(answer) > MAX_ANSWER_LENGTH:

        answer = answer[:MAX_ANSWER_LENGTH].rstrip()

        answer += "\n\n[Answer truncated.]"


    # --------------------------------------------------------
    # BASIC OUTPUT CHECK
    # --------------------------------------------------------

    suspicious_patterns = [
        "BEGIN SYSTEM PROMPT",
        "SYSTEM MESSAGE:",
        "DEVELOPER MESSAGE:",
        "INTERNAL INSTRUCTIONS:",
    ]

    answer_upper = answer.upper()

    for pattern in suspicious_patterns:

        if pattern in answer_upper:

            return {
                "allowed": False,
                "message": (
                    "The generated response was "
                    "blocked by the output safety check."
                ),
                "answer": ""
            }


    # --------------------------------------------------------
    # SAFE RESPONSE
    # --------------------------------------------------------

    return {
        "allowed": True,
        "message": "Answer passed output validation.",
        "answer": answer
    }