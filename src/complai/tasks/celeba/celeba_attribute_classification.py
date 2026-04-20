from __future__ import annotations

from inspect_ai import task, Task
from inspect_ai.dataset import Sample

from complai.datasets.celeba_dataset import load_celeba_for_complai
from complai.solvers.image_attribute_solver import image_attribute_solver
from complai.scorers.attribute_classification_metrics import attribute_classification_scorer

import os
CELEBA_DIR = os.getenv("CELEBA_DIR", "/home/chatziko/PycharmProjects/PythonProject/celeba/")


TARGET_ATTRIBUTES = {
    "young",
    "male",
    "female",
    "smiling",
    "eyeglasses",
    "black_hair",
    "blond_hair",
    "bald",
    "mustache",
    "wearing_lipstick",
}

@task(technical_requirement="Robustness and Predictability")
def celeba_attribute_classification():
    ds = load_celeba_for_complai(
        #root_dir="/home/chatziko/PycharmProjects/PythonProject/celeba/celebamask-hq/CelebAMask-HQ/",
        root_dir=CELEBA_DIR+"celebamask-hq/CelebAMask-HQ/",
        subset_size=100,
        random_subset=True,
        #cache_path="/home/chatziko/PycharmProjects/PythonProject/celeba/celebamask-hq/celeba_dataset_cache.pkl",
        cache_path=CELEBA_DIR+"celebamask-hq/celeba_dataset_cache.pkl",
        seed=42,
    )

    samples = []
    for item in ds:
        raw_attrs = item["raw_attributes"]

        normalized_attrs = [
            attr.lower().replace(" ", "_")
            for attr in raw_attrs
            if attr.lower().replace(" ", "_") in TARGET_ATTRIBUTES
        ]

        samples.append(
            Sample(
                input=(
                    "Predict the facial attributes present in this image. "
                    "Return a comma-separated list using only these labels: "
                    "young, male, female, smiling, eyeglasses, black_hair, "
                    "blond_hair, bald, mustache, wearing_lipstick."
                ),
                target=normalized_attrs,
                metadata={
                    "image": item["image"],
                    "dataset_name": "CelebAMask-HQ",
                    "gold_attributes": item["target_attributes"],
                    "attributes": item["attributes"],
                    "pose": item["pose"],
                    "caption": item["caption"],
                    "raw_attributes": item["raw_attributes"],
                },
            )
        )

    return Task(
        dataset=samples,
        solver=image_attribute_solver(),
        scorer=attribute_classification_scorer(),
    )