def confidence_score(sources: list, verified: bool) -> float:
    score = 0.4

    if sources:
        score += 0.3
    if len(sources) >= 2:
        score += 0.2
    if verified:
        score += 0.1

    return round(min(score, 0.99), 2)
