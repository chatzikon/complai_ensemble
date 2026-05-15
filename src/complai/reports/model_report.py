from __future__ import annotations

from collections import defaultdict
from typing import Any


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_requirement_scores(
    metric_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregation hierarchy:

    normalized metric scores
        -> technical requirement score
        -> ethical principle score
        -> overall score
    """

    requirement_to_metrics: dict[str, list[float]] = defaultdict(list)
    requirement_to_principle: dict[str, str] = {}
    requirement_metadata: dict[str, dict[str, Any]] = {}

    for result in metric_results:
        normalized_score = result.get("normalized_score")

        if normalized_score is None:
            continue

        requirement_id = result["requirement_id"]
        principle_id = result["principle_id"]

        requirement_to_metrics[requirement_id].append(normalized_score)
        requirement_to_principle[requirement_id] = principle_id

        requirement_metadata[requirement_id] = {
            "requirement_id": requirement_id,
            "requirement_name": result.get("requirement_name"),
            "principle_id": principle_id,
            "principle_name": result.get("principle_name"),
        }

    by_requirement: dict[str, Any] = {}

    for requirement_id, scores in requirement_to_metrics.items():
        by_requirement[requirement_id] = {
            **requirement_metadata[requirement_id],
            "score": mean(scores),
            "num_metrics": len(scores),
            "metric_scores": scores,
        }

    principle_to_requirements: dict[str, list[float]] = defaultdict(list)
    principle_metadata: dict[str, dict[str, Any]] = {}

    for requirement in by_requirement.values():
        principle_id = requirement["principle_id"]

        if requirement["score"] is not None:
            principle_to_requirements[principle_id].append(requirement["score"])

        principle_metadata[principle_id] = {
            "principle_id": principle_id,
            "principle_name": requirement.get("principle_name"),
        }

    by_principle: dict[str, Any] = {}

    for principle_id, requirement_scores in principle_to_requirements.items():
        by_principle[principle_id] = {
            **principle_metadata[principle_id],
            "score": mean(requirement_scores),
            "num_requirements": len(requirement_scores),
            "requirement_scores": requirement_scores,
        }

    overall_scores = [
        principle["score"]
        for principle in by_principle.values()
        if principle["score"] is not None
    ]

    return {
        "overall": {
            "score": mean(overall_scores),
            "num_principles": len(overall_scores),
        },
        "by_principle": by_principle,
        "by_requirement": by_requirement,
        "metric_results": metric_results,
    }