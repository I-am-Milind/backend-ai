def verify(answer: str, sources: list) -> bool:
    if not sources:
        return False

    red_flags = [
        "I think", "probably", "might be",
        "guess", "not sure"
    ]

    for f in red_flags:
        if f.lower() in answer.lower():
            return False

    return True
