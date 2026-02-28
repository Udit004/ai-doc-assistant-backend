def lexical_overlap_score(answer: str, reference: str) -> float:
    answer_tokens = {token.lower() for token in answer.split() if token.strip()}
    reference_tokens = {token.lower() for token in reference.split() if token.strip()}

    if not answer_tokens or not reference_tokens:
        return 0.0

    intersection = answer_tokens.intersection(reference_tokens)
    union = answer_tokens.union(reference_tokens)
    return len(intersection) / len(union)

