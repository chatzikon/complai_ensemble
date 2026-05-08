from pathlib import Path
from typing import Any

import yaml


def load_altai_questions(questionnaire_path: str | Path) -> dict[str, Any]:
    questionnaire_path = Path(questionnaire_path)

    with questionnaire_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_altai_samples(questionnaire: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    for principle in questionnaire.get("principles", []):
        for requirement in principle.get("technical_requirements", []):
            for question in requirement.get("questions", []):
                samples.append(
                    {
                        "principle_id": principle["principle_id"],
                        "principle_name": principle["principle_name"],
                        "requirement_id": requirement["requirement_id"],
                        "requirement_name": requirement["requirement_name"],
                        "question_id": question["question_id"],
                        "question": question["question"].strip(),
                        "supporting_indicators": question.get(
                            "supporting_indicators", []
                        ),
                        # "expected_evidence": question.get(
                        #     "expected_evidence", []
                        # ),
                        "scoring_guidelines": question.get(
                            "scoring_guidelines", {}
                        ),
                    }
                )

    return samples