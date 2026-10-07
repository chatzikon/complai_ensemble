---
title: "ENSEMBLE / COMPL-AI Extension & Handover Guide"
subtitle: "Developer onboarding and extension templates"
author: "ENSEMBLE project"
date: "October 2026"
toc: true
toc-depth: 2
geometry: margin=1in
fontsize: 10pt
---

> Repository: `chatzikon/complai_ensemble`  
> Purpose: help a new colleague understand the existing evaluation platform, reproduce the current work, and safely extend it with new benchmarks, qualitative questions, models, datasets, solvers, scorers, metrics, and UI mappings.

---

## 1. What this repository does

The repository extends COMPL-AI / Inspect AI with an ENSEMBLE-oriented evaluation workflow containing two complementary assessment paths:

1. **Quantitative evaluation** - runnable benchmarks executed through `complai eval` / Inspect AI.
2. **Qualitative evaluation** - questionnaire-based ALTAI-style governance assessment scored on a 0/1/2/N-A scale.

The Streamlit UI ties the two paths together and stores results so they can be aggregated in the model report.

### Mental model

```text
QUANTITATIVE
Dataset -> Task -> Solver -> Model -> Scorer -> Metric(s) -> Logs -> Report

QUALITATIVE
YAML Questions -> UI Responses -> Qualitative Solver -> Qualitative Scorer -> Report
```

---

## 2. Repository map

The most important extension points are:

```text
complai_ensemble/
|
|-- src/complai/
|   |-- datasets/       # Data loading and preprocessing
|   |-- models/         # Custom local models and Inspect providers
|   |-- scorers/        # Per-sample scorers and aggregate metrics
|   |-- solvers/        # How a sample is passed to a model
|   |-- tasks/          # Quantitative benchmark definitions
|   |-- reports/        # Result parsing, storage and aggregation
|   |-- _cli/           # complai CLI implementation
|   `-- _registry.py    # Inspect/COMPL-AI task registration
|
|-- config/
|   `-- altai/
|       `-- qualitative_questions_v1.yaml
|
|-- ui/
|   |-- app.py                     # Quantitative Streamlit page + app entry point
|   |-- altai_qualitative_ui.py    # Qualitative questionnaire UI
|   |-- config.py                  # UI providers/default models
|   `-- utils/
|       |-- discovery.py           # Task discovery + principle mapping
|       |-- devices.py
|       |-- metrics.py
|       `-- runner.py
|
`-- pyproject.toml                 # Dependencies + Inspect plugin entry point
```

### Where are the qualitative questions?

They are **not** under `src/complai/`. The questionnaire content lives in:

```text
config/altai/qualitative_questions_v1.yaml
```

That separation is intentional: the YAML is assessment content/configuration, while the Python code under `src/complai/` loads, solves and scores it.

---

## 3. Existing ENSEMBLE-specific examples

The current repository contains useful examples for each type of extension.

### Quantitative task examples

```text
src/complai/tasks/flickr30k/flickr30k_captioning.py
src/complai/tasks/celeba/celeba_attribute_classification.py
src/complai/tasks/malware_cfg/malware_cfg_eval.py
src/complai/tasks/ton_iot/ton_iot_binary.py
src/complai/tasks/ton_iot/ton_iot_multiclass.py
```

### Dataset examples

```text
src/complai/datasets/flickr30k_dataset.py
src/complai/datasets/celeba_dataset.py
src/complai/datasets/malware_cfg_dataset.py
src/complai/datasets/ton_iot_network_dataset.py
```

### Solver examples

```text
src/complai/solvers/image_caption_solver.py
src/complai/solvers/image_attribute_solver.py
src/complai/solvers/malware_gcn_solver.py
```

### Custom model/provider examples

```text
src/complai/models/multimodal/
src/complai/models/malware_gcn/
```

### Scorer / metric examples

```text
src/complai/scorers/caption_metrics.py
src/complai/scorers/attribute_classification_metrics.py
src/complai/scorers/classification_metrics.py
src/complai/scorers/malware_metrics.py
```

### Qualitative evaluation implementation

```text
config/altai/qualitative_questions_v1.yaml
src/complai/tasks/altai_qualitative/altai_qualitative.py
src/complai/solvers/altai_qualitative_solver.py
src/complai/scorers/altai_qualitative_metrics.py
ui/altai_qualitative_ui.py
```

---

# PART I - QUANTITATIVE EXTENSIONS

## 4. Add a quantitative benchmark

A quantitative benchmark is centered on a function decorated with Inspect AI's `@task`.

Recommended location:

```text
src/complai/tasks/<benchmark_name>/
|-- __init__.py
`-- <benchmark_name>.py
```

### Minimal task template

```python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from complai.datasets.my_dataset import load_my_dataset
from complai.scorers.my_metrics import my_scorer
from complai.solvers.my_solver import my_solver


@task(technical_requirement="Robustness and Predictability")
def my_benchmark():
    rows = load_my_dataset()

    samples = [
        Sample(
            input=row["input"],
            target=row["target"],
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]

    return Task(
        dataset=samples,
        solver=my_solver(),
        scorer=my_scorer(),
    )
```

### Important: technical requirement

The decorator value:

```python
@task(technical_requirement="Robustness and Predictability")
```

is used by COMPL-AI to group the task and by the ENSEMBLE UI to build the hierarchy:

```text
Ethical Principle -> Technical Requirement -> Benchmark
```

Use the exact technical requirement string expected by the project.

### Task package `__init__.py`

Recommended:

```python
from .my_benchmark import my_benchmark

__all__ = ["my_benchmark"]
```

Some existing custom folders are less uniform; new additions should follow this convention.

---

## 5. Decide whether a custom solver is needed

Use this rule of thumb:

```text
Can Inspect send the Sample.input directly to the selected provider?
|
|-- YES -> usually no custom solver is needed.
|
`-- NO  -> add a custom solver.
           Typical cases: images, graphs, tensors, custom local inference APIs.
```

For example, TON-IoT can use ordinary text generation and therefore its task can be as simple as:

```python
return Task(
    dataset=samples,
    scorer=classification_scorer(),
)
```

By contrast, image and malware-graph benchmarks require custom inference logic.

---

## 6. Add a dataset

Dataset loaders live in:

```text
src/complai/datasets/
```

Their job is to load and normalize data - not to contain evaluation logic.

### Generic dataset template

```python
from __future__ import annotations


def load_my_dataset(split: str = "test"):
    raw_dataset = ...
    rows = []

    for item in raw_dataset:
        rows.append(
            {
                "input": ...,
                "target": ...,
                "metadata": {
                    "sample_id": ...,
                },
            }
        )

    return rows
```

### Use metadata for non-text inputs

Examples already used in the repository:

```python
metadata={"image": image, "dataset_name": "Flickr30k"}
```

```python
metadata={"graph": graph}
```

The solver can then read:

```python
image = state.metadata["image"]
```

or:

```python
graph = state.metadata["graph"]
```

### Avoid hard-coded developer paths

Prefer environment variables:

```python
import os

MY_DATA_DIR = os.getenv("MY_DATA_DIR")
if not MY_DATA_DIR:
    raise RuntimeError("MY_DATA_DIR is not configured")
```

Current repository examples use variables such as `CELEBA_DIR`, `MALWARE_DIR`, `HF_HOME` and `HF_TOKEN`.

---

## 7. Add a solver

Solvers live in:

```text
src/complai/solvers/
```

A solver answers one question:

> Given one Inspect sample, how is the selected model actually executed?

### Generic solver template

```python
from inspect_ai.model import ModelOutput
from inspect_ai.solver import solver

from complai.models.my_model.my_model import MyModel


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = MyModel.from_components()
    return _MODEL


@solver
def my_solver():
    async def solve(state, generate):
        model = get_model()
        input_data = state.metadata["my_input"]

        prediction = model.predict(input_data)

        state.output = ModelOutput(
            completion=str(prediction)
        )
        return state

    return solve
```

### Important for heavy models

Do **not** instantiate a large model inside `solve()` for every sample.

Use a cache/singleton pattern, as the existing image solvers do with `_MODEL_CACHE`. This avoids repeated checkpoint loading and unnecessary GPU allocations.

---

## 8. Add a custom model

Custom model code lives in:

```text
src/complai/models/<model_family>/
```

Recommended structure:

```text
models/my_model/
|-- __init__.py
|-- my_model.py       # actual inference/model wrapper
`-- my_provider.py    # Inspect ModelAPI provider, when needed
```

### Actual local model wrapper

```python
import torch


class MyModel:
    def __init__(self, model, device: str):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @classmethod
    def from_components(
        cls,
        model_name: str,
        device: str | None = None,
    ):
        device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = ...
        return cls(model=model, device=device)

    @torch.no_grad()
    def predict(self, input_data):
        output = self.model(...)
        return output
```

### Inspect provider template

Create a provider when you want model identifiers such as:

```text
myprovider/mymodel
```

Template:

```python
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo


class MyProvider(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ):
        super().__init__(
            model_name,
            base_url,
            api_key,
            api_key_vars,
            config,
        )
        self.model_name = model_name
        self.model_args = model_args

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        prediction = ...
        return ModelOutput.from_content(str(prediction))


@modelapi(name="myprovider")
def myprovider():
    return MyProvider
```

### Existing repository pattern

The current `malware` and `multimodal` providers are lightweight because the custom solver performs the real inference. Document this clearly when extending them so a future developer does not mistake the placeholder `generate()` implementation for the actual model path.

---

## 9. Add evaluation metrics

Metrics and scorers live in:

```text
src/complai/scorers/
```

There are two levels:

```text
Scorer = evaluate/store information for each sample
Metric = aggregate over all sample scores
```

### Generic metric template

```python
from inspect_ai.scorer import SampleScore, metric


@metric
def my_metric():
    def compute(sample_scores: list[SampleScore]) -> float:
        predictions = []
        references = []

        for sample_score in sample_scores:
            metadata = sample_score.score.metadata or {}
            predictions.append(metadata["prediction"])
            references.append(metadata["reference"])

        result = ...
        return float(result)

    return compute
```

### Generic scorer template

```python
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState


@scorer(metrics=[my_metric()])
def my_scorer():
    async def score(state: TaskState, target: Target):
        prediction = (state.output.completion or "").strip()
        reference = target.text if hasattr(target, "text") else target

        return Score(
            value=0.0,
            answer=prediction,
            metadata={
                "prediction": prediction,
                "reference": reference,
            },
        )

    return score
```

### Why store prediction/reference in metadata?

It lets several aggregate metrics reuse the same model outputs. Existing examples include:

- image captioning: BLEU-1..4, ROUGE-L, CIDEr, SPICE
- attribute classification: accuracy, precision, recall, F1
- classification: accuracy, macro-F1, macro-precision

---

# PART II - QUALITATIVE EXTENSIONS

## 10. Add a qualitative question

The questionnaire lives in:

```text
config/altai/qualitative_questions_v1.yaml
```

A new question normally requires **no Python change**.

### Question template

```yaml
- question_id: XXX-001

  question: >
    Your qualitative assessment question.

  supporting_indicators:
    - "Indicator 1"
    - "Indicator 2"

  expected_evidence:
    - "Evidence type 1"
    - "Evidence type 2"

  scoring_guidelines:
    0: >
      Conditions for Not Implemented.

    1: >
      Conditions for Partially Implemented.

    2: >
      Conditions for Fully Implemented.
```

The UI additionally supports `N/A`, represented internally as `None`.

### Existing score scale

```text
N/A = Not applicable
0   = Not Implemented
1   = Partially Implemented
2   = Fully Implemented
```

---

## 11. Add a qualitative requirement or principle

The questionnaire is hierarchical:

```text
Principle -> Technical Requirement -> Question
```

Example:

```yaml
- principle_id: EP3
  principle_name: "Transparency"

  technical_requirements:

    - requirement_id: TRA-1.1
      requirement_name: "AI Disclosure"

      questions:

        - question_id: TRA-001
          question: >
            Is the use of AI clearly disclosed to the user?

          supporting_indicators:
            - "Is disclosure explicit and understandable?"

          expected_evidence:
            - "UI disclosure"
            - "User documentation"

          scoring_guidelines:
            0: "No disclosure mechanism."
            1: "Partial or inconsistent disclosure."
            2: "Clear and consistent disclosure."
```

The existing loader already loops dynamically over principles, requirements and questions, so new YAML entries should be picked up automatically.

### Qualitative processing path

```text
qualitative_questions_v1.yaml
        |
        v
build_altai_samples()
        |
        v
ui/altai_qualitative_ui.py
        |
        v
AltaiQualitativeSolver
        |
        v
altai_qualitative_scorer
        |
        +--> by question
        +--> by requirement
        +--> by principle
        `--> overall
```

---

# PART III - UI, MAPPING, REGISTRATION & DEPENDENCIES

## 12. Map a technical requirement to an ethical principle

The quantitative UI needs a second-level mapping that is separate from the COMPL-AI task metadata.

Location:

```text
ui/utils/discovery.py
```

Pattern:

```python
TECHNICAL_REQUIREMENT_TO_PRINCIPLE = {
    "Disclosure of AI": "Transparency",
    "Societal Alignment": "Unmapped Principle",
    ...
}
```

Important rule:

> Do not invent a principle mapping just to make the UI complete. If the official mapping is unclear, use `Unmapped Principle`.

This is particularly important for `Societal Alignment`, whose task grouping is explicit in COMPL-AI but whose one-to-one mapping to one EU ethical principle is not currently explicit.

---

## 13. Add a model/provider to the Streamlit UI

Location:

```text
ui/config.py
```

Add the provider:

```python
LOCAL_PROVIDERS = [
    ...,
    "myprovider",
]
```

Add a convenient default model:

```python
PROVIDER_DEFAULT_MODELS = {
    ...,
    "myprovider": "mymodel",
}
```

The UI constructs the model identifier as:

```text
provider/model_name
```

Example:

```text
multimodal/CMAlign
malware/baseline_gcn
hf/Qwen/Qwen3-8B
```

---

## 14. Task and provider registration

The Inspect plugin entry point is configured in:

```text
pyproject.toml
```

```toml
[project.entry-points.inspect_ai]
complai = "complai._registry"
```

Task registration is centralized in:

```text
src/complai/_registry.py
```

Recommended production convention for a new task:

```python
from complai.tasks.my_benchmark import my_benchmark
```

Recommended for a custom provider:

```python
from complai.models.my_model.my_provider import myprovider
```

The current CLI also discovers tasks by scanning the tasks root via Inspect's `list_tasks()`, but explicit imports remain useful for predictable plugin registration and maintainability.

---

## 15. Dependencies

New Python dependencies belong in:

```text
pyproject.toml
```

For example, the current extensions rely on packages such as:

```text
torch-geometric
pycocoevalcap
streamlit
datasets
torchvision
```

Rule:

> If a new benchmark imports a package that is not already a project dependency, add it to `pyproject.toml`. Do not rely on a package that only happens to exist in one developer's environment.

---

## 16. Result storage and reporting

The Streamlit quantitative path saves information including:

```text
model_name
task
ethical_principle
technical_requirement
category (kept for compatibility)
metrics
metric_results
log_dir
```

Relevant modules:

```text
src/complai/reports/storage.py
src/complai/reports/log_parser.py
src/complai/reports/model_report.py
src/complai/reports/metric_utils.py
```

When adding a new scorer or metric, confirm that the saved Inspect logs can still be parsed by the report layer.

---

# PART IV - STANDARD EXTENSION RECIPES

## 17. What files do I modify?

| Goal | Files normally involved |
|---|---|
| New text benchmark using an existing provider | `datasets/`, `tasks/`, `scorers/` |
| Benchmark requiring custom inference | `datasets/`, `tasks/`, `solvers/`, `scorers/` |
| New local model | `models/`, often `solvers/`, `_registry.py`, `ui/config.py` |
| New aggregate metric | `scorers/` |
| New qualitative question | `config/altai/qualitative_questions_v1.yaml` |
| New qualitative requirement | same YAML |
| New qualitative principle | same YAML |
| New quantitative technical requirement | task decorator + `ui/utils/discovery.py` if principle mapping is known |
| New provider shown in UI | model provider + `_registry.py` + `ui/config.py` |
| New dependency | `pyproject.toml` |
| New external dataset/token | environment variables + deployment config |
| New report aggregation behavior | `src/complai/reports/` |

---

## 18. Recommended package template for a substantial benchmark

```text
src/complai/
|
|-- datasets/
|   `-- my_benchmark_dataset.py
|
|-- solvers/
|   `-- my_benchmark_solver.py
|
|-- scorers/
|   `-- my_benchmark_metrics.py
|
|-- tasks/
|   `-- my_benchmark/
|       |-- __init__.py
|       `-- my_benchmark.py
|
`-- models/
    `-- my_model/              # only when a custom model is needed
        |-- __init__.py
        |-- my_model.py
        `-- my_provider.py
```

Conceptually:

```text
                  +----------------+
                  |    Dataset     |
                  +-------+--------+
                          |
                          v
                  +----------------+
                  |      Task      |
                  +---+--------+---+
                      |        |
                      v        v
                 +--------+ +--------+
                 | Solver | | Scorer |
                 +---+----+ +---+----+
                     |          |
                     v          v
                 +--------+ +--------+
                 | Model  | | Metrics|
                 +--------+ +--------+
```

---

# PART V - VALIDATION & HANDOVER

## 19. Minimal validation sequence

### Step 1 - verify task discovery

```bash
complai list
```

Confirm the new benchmark appears under the expected technical requirement.

### Step 2 - run only 1-2 samples first

```bash
complai eval <provider>/<model> \
  -t my_benchmark \
  -l 2 \
  --debug
```

Do not begin with a full dataset run.

### Step 3 - check the complete path

Verify:

```text
[ ] Dataset loads
[ ] Task is discovered
[ ] Technical requirement is correct
[ ] Model loads once, not once per sample
[ ] Solver produces the expected output format
[ ] Scorer receives prediction and reference
[ ] Aggregate metrics are plausible
[ ] Inspect logs are created
[ ] Streamlit UI discovers the benchmark
[ ] Ethical-principle mapping is correct or explicitly Unmapped
[ ] Saved result appears in the model report
```

### Step 4 - launch the UI

```bash
streamlit run ui/app.py
```

Test both:

```text
Quantitative Benchmarks
Qualitative Evaluation
```

---

## 20. New colleague: first-day checklist

A colleague taking over the project should be able to do the following before making changes:

1. Clone the repository and create a Python environment compatible with `pyproject.toml`.
2. Install the package in editable/development mode.
3. Configure required environment variables and dataset paths.
4. Run `complai list` and understand the task / technical-requirement grouping.
5. Run one lightweight benchmark with `-l 2`.
6. Start `streamlit run ui/app.py`.
7. Open the quantitative page and understand the hierarchy:
   `Ethical Principle -> Technical Requirement -> Benchmark`.
8. Open the qualitative page and identify that question content comes from YAML.
9. Inspect one example from each layer:
   dataset, task, solver, model, scorer.
10. Make one harmless test change on a branch before touching production evaluation logic.

---

## 21. Suggested onboarding reading order

A new colleague does **not** need to read the repository randomly. Use this order:

```text
1. ui/app.py
   Understand the user workflow.

2. ui/utils/discovery.py
   Understand task grouping and ethical-principle mapping.

3. One simple benchmark:
   src/complai/tasks/ton_iot/ton_iot_binary.py
   + dataset
   + classification scorer

4. One custom-inference benchmark:
   src/complai/tasks/flickr30k/flickr30k_captioning.py
   + image solver
   + multimodal model
   + caption metrics

5. One non-standard model:
   src/complai/tasks/malware_cfg/malware_cfg_eval.py
   + malware solver
   + malware GCN model/provider

6. Qualitative flow:
   config/altai/qualitative_questions_v1.yaml
   -> altai_qualitative.py
   -> altai_qualitative_solver.py
   -> altai_qualitative_metrics.py
   -> ui/altai_qualitative_ui.py

7. Reporting:
   src/complai/reports/

8. Registration and packaging:
   src/complai/_registry.py
   pyproject.toml
```

---

## 22. Common mistakes to avoid

### Mistake 1 - putting all logic in the task file

Keep responsibilities separate:

```text
loading -> dataset
inference -> solver/model
aggregation -> scorer/metric
orchestration -> task
```

### Mistake 2 - loading a large model for every sample

Cache it once.

### Mistake 3 - hard-coding local paths

Use environment variables or deployment configuration.

### Mistake 4 - adding a metric but not returning its required metadata

If the aggregate metric expects `prediction` and `reference`, the scorer must store them.

### Mistake 5 - creating an unsupported ethical-principle mapping

Use `Unmapped Principle` when the mapping is not explicit.

### Mistake 6 - forgetting the UI

A provider can work from the CLI but still be absent from `ui/config.py`.

### Mistake 7 - forgetting dependencies

Update `pyproject.toml` and rebuild/reinstall the environment.

### Mistake 8 - testing on the full dataset first

Start with `-l 1` or `-l 2`.

---

## 23. Definition of done for a new extension

A new benchmark/model/question is complete only when:

```text
DISCOVERY
[ ] It is discoverable in the intended interface.

EXECUTION
[ ] It runs on a small smoke test.
[ ] It handles device/model loading correctly.

SCORING
[ ] Scores are deterministic where expected.
[ ] Metrics have a documented interpretation and range.

TRACEABILITY
[ ] Technical requirement is explicit.
[ ] Ethical principle is mapped only when justified.

REPORTING
[ ] Result is stored and appears in the model report.

PORTABILITY
[ ] No developer-specific paths are required.
[ ] Dependencies and environment variables are documented.

MAINTAINABILITY
[ ] Files follow dataset/task/solver/model/scorer separation.
[ ] The new component has a small smoke test or reproducible command.
```

---

## 24. Quick-copy templates

### Quantitative task

```python
@task(technical_requirement="<REQUIREMENT>")
def <TASK_NAME>():
    rows = load_<DATASET>()
    samples = [
        Sample(
            input=row["input"],
            target=row["target"],
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]
    return Task(
        dataset=samples,
        solver=<SOLVER>(),       # omit when standard generation is enough
        scorer=<SCORER>(),
    )
```

### Qualitative question

```yaml
- question_id: <ID>
  question: >
    <QUESTION>
  supporting_indicators:
    - "<INDICATOR>"
  expected_evidence:
    - "<EVIDENCE>"
  scoring_guidelines:
    0: "<NOT IMPLEMENTED>"
    1: "<PARTIALLY IMPLEMENTED>"
    2: "<FULLY IMPLEMENTED>"
```

### Metric

```python
@metric
def <METRIC_NAME>():
    def compute(sample_scores: list[SampleScore]) -> float:
        ...
        return float(result)
    return compute
```

### Provider in UI

```python
LOCAL_PROVIDERS = [
    ...,
    "<PROVIDER>",
]

PROVIDER_DEFAULT_MODELS = {
    ...,
    "<PROVIDER>": "<DEFAULT_MODEL>",
}
```

---

## 25. Final principle

When extending the platform, keep the implementation **modular, discoverable, reproducible, and traceable**.

A future colleague should be able to answer these five questions by looking at the code:

1. **What data is being evaluated?** -> dataset
2. **What benchmark is being run?** -> task
3. **How is the model called?** -> solver/model
4. **How is success measured?** -> scorer/metrics
5. **Which trustworthiness requirement does it support?** -> task metadata + UI mapping

If those five answers are clear, the extension will usually remain maintainable as COMPL-AI evolves.