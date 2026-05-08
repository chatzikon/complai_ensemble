from typing import Any


VALID_SCORES = {0, 1, 2}
MAX_SCORE_PER_QUESTION = 2


def validate_score(score: Any) -> bool:
    return score in VALID_SCORES


def _summarize(
    scores: list[int],
    num_questions: int,
    na_count: int = 0,
) -> dict[str, Any]:
    if not scores:
        return {
            "score": 0,
            "max_score": 0,
            "normalized_score": None,
            "num_questions": num_questions,
            "not_applicable_questions": na_count,
        }

    total = sum(scores)
    max_score = MAX_SCORE_PER_QUESTION * len(scores)

    return {
        "score": total,
        "max_score": max_score,
        "normalized_score": total / max_score,
        "num_questions": num_questions,
        "not_applicable_questions": na_count,
    }


def altai_qualitative_scorer(
    samples: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    prediction_by_id = {
        prediction["question_id"]: prediction
        for prediction in predictions
    }

    question_results: list[dict[str, Any]] = []

    principle_scores: dict[str, list[int]] = {}
    requirement_scores: dict[str, list[int]] = {}

    principle_question_counts: dict[str, int] = {}
    requirement_question_counts: dict[str, int] = {}

    principle_na_counts: dict[str, int] = {}
    requirement_na_counts: dict[str, int] = {}

    overall_question_count = 0
    overall_na_count = 0

    for sample in samples:
        question_id = sample["question_id"]
        prediction = prediction_by_id.get(question_id, {})
        score = prediction.get("score")

        overall_question_count += 1

        principle_id = sample["principle_id"]
        requirement_id = sample["requirement_id"]

        principle_scores.setdefault(principle_id, [])
        requirement_scores.setdefault(requirement_id, [])

        principle_question_counts[principle_id] = (
            principle_question_counts.get(principle_id, 0) + 1
        )
        requirement_question_counts[requirement_id] = (
            requirement_question_counts.get(requirement_id, 0) + 1
        )

        if score is None:
            overall_na_count += 1
            principle_na_counts[principle_id] = (
                principle_na_counts.get(principle_id, 0) + 1
            )
            requirement_na_counts[requirement_id] = (
                requirement_na_counts.get(requirement_id, 0) + 1
            )

        elif validate_score(score):
            principle_scores[principle_id].append(score)
            requirement_scores[requirement_id].append(score)

        question_results.append(
            {
                **sample,
                "score": score,
                "max_score": MAX_SCORE_PER_QUESTION,
                "normalized_score": (
                    score / MAX_SCORE_PER_QUESTION
                    if validate_score(score)
                    else None
                ),
                "justification": prediction.get("justification", ""),
            }
        )

    valid_scores = [
        result["score"]
        for result in question_results
        if validate_score(result["score"])
    ]

    return {
        "overall": _summarize(
            valid_scores,
            overall_question_count,
            overall_na_count,
        ),
        "by_principle": {
            principle_id: _summarize(
                principle_scores.get(principle_id, []),
                principle_question_counts[principle_id],
                principle_na_counts.get(principle_id, 0),
            )
            for principle_id in principle_question_counts
        },
        "by_requirement": {
            requirement_id: _summarize(
                requirement_scores.get(requirement_id, []),
                requirement_question_counts[requirement_id],
                requirement_na_counts.get(requirement_id, 0),
            )
            for requirement_id in requirement_question_counts
        },
        "question_results": question_results,
    }