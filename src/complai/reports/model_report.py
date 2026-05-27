from __future__ import annotations

from collections import defaultdict
from typing import Any

REQUIREMENT_METADATA = {
    # EP2 — Technical Robustness and Safety
    "Robustness and Predictability": {
        "requirement_id": "TRS-1",
        "requirement_name": "Robustness and Predictability",
        "principle_id": "EP2",
        "principle_name": "Technical Robustness and Safety",
    },
    "Cyberattack Resilience": {
        "requirement_id": "TRS-2",
        "requirement_name": "Cyberattack Resilience",
        "principle_id": "EP2",
        "principle_name": "Technical Robustness and Safety",
    },

    # EP3 — Privacy and Data Governance
    "Training Data Suitability": {
        "requirement_id": "PDG-1",
        "requirement_name": "Training Data Suitability",
        "principle_id": "EP3",
        "principle_name": "Privacy and Data Governance",
    },
    "No Copyright Infringement": {
        "requirement_id": "PDG-2",
        "requirement_name": "No Copyright Infringement",
        "principle_id": "EP3",
        "principle_name": "Privacy and Data Governance",
    },
    "User Privacy Protection": {
        "requirement_id": "PDG-3",
        "requirement_name": "User Privacy Protection",
        "principle_id": "EP3",
        "principle_name": "Privacy and Data Governance",
    },

    # EP4 — Transparency
    "Capabilities, Performance, and Limitations": {
        "requirement_id": "TRA-1",
        "requirement_name": "Capabilities, Performance, and Limitations",
        "principle_id": "EP4",
        "principle_name": "Transparency",
    },
    "Interpretability": {
        "requirement_id": "TRA-2",
        "requirement_name": "Interpretability",
        "principle_id": "EP4",
        "principle_name": "Transparency",
    },
    "Disclosure of AI Presence": {
        "requirement_id": "TRA-3",
        "requirement_name": "Disclosure of AI Presence",
        "principle_id": "EP4",
        "principle_name": "Transparency",
    },
    "Traceability": {
        "requirement_id": "TRA-4",
        "requirement_name": "Traceability",
        "principle_id": "EP4",
        "principle_name": "Transparency",
    },

    # EP5 — Diversity, Non-discrimination and Fairness
    "Fairness -- Absence of Discrimination": {
        "requirement_id": "DNF-1",
        "requirement_name": "Fairness — Absence of Discrimination",
        "principle_id": "EP5",
        "principle_name": "Diversity, Non-discrimination and Fairness",
    },
    "Representation -- Absence of Bias": {
        "requirement_id": "DNF-2",
        "requirement_name": "Representation — Absence of Bias",
        "principle_id": "EP5",
        "principle_name": "Diversity, Non-discrimination and Fairness",
    },

    # EP6 — Societal and Environmental Well-being
    "Environmental Impact": {
        "requirement_id": "SEW-1",
        "requirement_name": "Environmental Impact",
        "principle_id": "EP6",
        "principle_name": "Societal and Environmental Well-being",
    },
    "Harmful Content and Toxicity": {
        "requirement_id": "SEW-2",
        "requirement_name": "Harmful Content and Toxicity",
        "principle_id": "EP6",
        "principle_name": "Societal and Environmental Well-being",
    },
}

def enrich_metric_results(
    metric_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched_results: list[dict[str, Any]] = []

    for result in metric_results:
        requirement_name = (
            result.get("technical_requirement")
            or result.get("requirement_name")
            or result.get("requirement_id")
        )

        if requirement_name is None:
            continue

        metadata = REQUIREMENT_METADATA.get(requirement_name)

        if metadata is None:
            # Keep it visible instead of silently dropping it
            enriched_results.append(
                {
                    **result,
                    "requirement_id": requirement_name,
                    "requirement_name": requirement_name,
                    "principle_id": "Unknown",
                    "principle_name": "Unknown",
                }
            )
            continue

        enriched_results.append(
            {
                **result,
                "requirement_id": metadata["requirement_id"],
                "requirement_name": metadata["requirement_name"],
                "principle_id": metadata["principle_id"],
                "principle_name": metadata["principle_name"],
            }
        )

    return enriched_results


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_requirement_scores(
    metric_results: list[dict[str, Any]],
) -> dict[str, Any]:
    enriched_results = enrich_metric_results(metric_results)

    benchmark_to_metrics: dict[str, list[float]] = defaultdict(list)
    benchmark_metadata: dict[str, dict[str, Any]] = {}

    for result in enriched_results:
        task = result["task"]
        requirement_id = result["requirement_id"]
        benchmark_key = f"{task}::{requirement_id}"

        benchmark_to_metrics[benchmark_key].append(result["normalized_score"])

        benchmark_metadata[benchmark_key] = {
            "benchmark_id": benchmark_key,
            "benchmark_name": task,
            "requirement_id": requirement_id,
            "requirement_name": result["requirement_name"],
            "principle_id": result["principle_id"],
            "principle_name": result["principle_name"],
        }

    by_benchmark: dict[str, Any] = {}

    for benchmark_id, scores in benchmark_to_metrics.items():
        by_benchmark[benchmark_id] = {
            **benchmark_metadata[benchmark_id],
            "score": mean(scores),
            "num_metrics": len(scores),
        }

    requirement_to_benchmarks: dict[str, list[float]] = defaultdict(list)
    requirement_metadata: dict[str, dict[str, Any]] = {}

    for benchmark in by_benchmark.values():
        requirement_id = benchmark["requirement_id"]

        if benchmark["score"] is not None:
            requirement_to_benchmarks[requirement_id].append(benchmark["score"])

        requirement_metadata[requirement_id] = {
            "requirement_id": requirement_id,
            "requirement_name": benchmark["requirement_name"],
            "principle_id": benchmark["principle_id"],
            "principle_name": benchmark["principle_name"],
        }

    by_requirement: dict[str, Any] = {}

    for requirement_id, scores in requirement_to_benchmarks.items():
        by_requirement[requirement_id] = {
            **requirement_metadata[requirement_id],
            "score": mean(scores),
            "num_benchmarks": len(scores),
            "benchmark_scores": scores,
        }

    principle_to_requirements: dict[str, list[float]] = defaultdict(list)
    principle_metadata: dict[str, dict[str, Any]] = {}

    for requirement in by_requirement.values():
        principle_id = requirement["principle_id"]

        if requirement["score"] is not None:
            principle_to_requirements[principle_id].append(requirement["score"])

        principle_metadata[principle_id] = {
            "principle_id": principle_id,
            "principle_name": requirement["principle_name"],
        }

    by_principle: dict[str, Any] = {}

    for principle_id, scores in principle_to_requirements.items():
        by_principle[principle_id] = {
            **principle_metadata[principle_id],
            "score": mean(scores),
            "num_requirements": len(scores),
            "requirement_scores": scores,
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
        "by_benchmark": by_benchmark,
        "metric_results": enriched_results,
    }

def build_model_report_from_saved_results(
    saved_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    quantitative_metric_results: list[dict[str, Any]] = []
    qualitative_reports: list[dict[str, Any]] = []

    for saved in saved_results:
        evaluation_name = saved.get("evaluation_name", "")
        result = saved.get("result", {})

        if evaluation_name.startswith("quantitative_"):
            quantitative_metric_results.extend(
                result.get("metric_results", [])
            )

        elif evaluation_name == "altai_qualitative":
            qualitative_reports.append(result)

    quantitative_report = aggregate_requirement_scores(
        quantitative_metric_results
    )

    # MVP: start from quantitative report
    report = quantitative_report

    # Add qualitative ALTAI requirement/principle scores into same report
    for qualitative in qualitative_reports:
        for requirement_id, requirement in qualitative.get("by_requirement", {}).items():


            report["by_requirement"][requirement_id] = {
                "requirement_id": requirement_id,
                "requirement_name": requirement_id,
                "principle_id": _infer_altai_principle_id(requirement_id),
                "principle_name": _infer_altai_principle_name(requirement_id),
                "score": requirement.get("normalized_score"),
                "num_questions": requirement.get("num_questions"),
                "not_applicable_questions": requirement.get(
                    "not_applicable_questions", 0
                ),
            }

        for principle_id, principle in qualitative.get("by_principle", {}).items():
            report["by_principle"][principle_id] = {
                "principle_id": principle_id,
                "principle_name": _infer_altai_principle_name_from_id(principle_id),
                "score": principle.get("normalized_score"),
                "num_questions": principle.get("num_questions"),
                "not_applicable_questions": principle.get(
                    "not_applicable_questions", 0
                ),
            }

    all_scores = [
        principle["score"]
        for principle in report["by_principle"].values()
        if principle.get("score") is not None
    ]

    report["overall"] = {
        "score": mean(all_scores),
        "num_principles": len(all_scores),
    }

    return report


def _infer_altai_principle_id(requirement_id: str) -> str:
    if requirement_id.startswith("HAO"):
        return "EP1"
    if requirement_id.startswith("ACC"):
        return "EP7"
    return "Unknown"


def _infer_altai_principle_name(requirement_id: str) -> str:
    if requirement_id.startswith("HAO"):
        return "Human Agency and Oversight"
    if requirement_id.startswith("ACC"):
        return "Accountability"
    return "Unknown"


def _infer_altai_principle_name_from_id(principle_id: str) -> str:
    if principle_id == "EP1":
        return "Human Agency and Oversight"
    if principle_id == "EP7":
        return "Accountability"
    return "Unknown"