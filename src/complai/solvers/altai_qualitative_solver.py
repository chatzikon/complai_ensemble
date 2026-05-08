from typing import Any


class AltaiQualitativeSolver:
    def __init__(self, responses: dict[str, dict[str, Any]]):
        self.responses = responses

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        question_id = sample["question_id"]
        response = self.responses.get(question_id, {})

        return {
            "question_id": question_id,
            "score": response.get("score"),
            "justification": response.get("justification", ""),
        }