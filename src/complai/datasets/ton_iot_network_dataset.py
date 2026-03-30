from __future__ import annotations

from datasets import load_dataset


EXCLUDE_COLUMNS = {"label", "type"}


def _format_value(value) -> str:
    if value is None:
        return "NA"
    return str(value)


def get_feature_columns(ds) -> list[str]:
    return sorted(
        col for col in ds.column_names
        if col not in EXCLUDE_COLUMNS and not col.startswith("_")
    )


def row_to_prompt(row: dict, feature_columns: list[str], label_mode: str = "binary") -> str:
    if label_mode == "binary":
        instruction = (
            "You are a network intrusion classifier.\n"
            "Classify the following network event as exactly one label: benign or malicious.\n"
            "Return only the label."
        )
    elif label_mode == "multiclass":
        instruction = (
            "You are a network intrusion classifier.\n"
            "Classify the following network event as exactly one attack type.\n"
            "Return only one label."
        )
    else:
        raise ValueError(f"Unsupported label_mode: {label_mode}")

    feature_lines = [
        f"{col}: {_format_value(row.get(col))}"
        for col in feature_columns
    ]

    return f"{instruction}\n\n" + "\n".join(feature_lines)


def load_ton_iot_network(
    split: str = "test[:1000]",
    label_mode: str = "binary",
):
    ds = load_dataset("codymlewis/TON_IoT_network", split=split)
    feature_columns = get_feature_columns(ds)

    samples = []
    for row in ds:
        prompt = row_to_prompt(row, feature_columns, label_mode=label_mode)

        if label_mode == "binary":
            target = "malicious" if int(row["label"]) == 1 else "benign"
        else:
            target = str(row["type"]).strip().lower()

        samples.append(
            {
                "input": prompt,
                "target": target,
                "metadata": {
                    "raw_label": row["label"],
                    "raw_type": row["type"],
                },
            }
        )

    return samples