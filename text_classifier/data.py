"""CSV loading, label validation, and reproducible stratified splitting."""

import csv
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split


def read_examples(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if not reader.fieldnames or not {"text", "label"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must have text and label columns")
        examples = []
        for line, row in enumerate(reader, start=2):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not text or not label:
                raise ValueError(f"{path}:{line} has an empty text or label")
            examples.append({"text": text, "label": label})
    if not examples:
        raise ValueError(f"{path} has no examples")
    return examples


def write_examples(path: str | Path, examples: list[dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(examples)


def split_examples(
    examples: list[dict[str, str]], test_fraction: float, validation_fraction: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("test and validation fractions must each be between 0 and 1")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test and validation fractions must sum to less than 1")
    counts = Counter(item["label"] for item in examples)
    if len(counts) < 2 or min(counts.values()) < 3:
        raise ValueError("at least two labels with at least three examples each are required")
    labels = [item["label"] for item in examples]
    try:
        remaining, test = train_test_split(
            examples, test_size=test_fraction, random_state=seed, stratify=labels
        )
        train, validation = train_test_split(
            remaining,
            test_size=validation_fraction / (1 - test_fraction),
            random_state=seed,
            stratify=[item["label"] for item in remaining],
        )
    except ValueError as exc:
        raise ValueError(
            "Dataset is too small for these stratified fractions; add examples or change fractions"
        ) from exc
    all_labels = set(counts)
    if any({item["label"] for item in part} != all_labels for part in (train, validation, test)):
        raise ValueError("Each split needs every label; add examples or change fractions")
    return train, validation, test


def label_names(train: list[dict[str, str]], *other_splits: list[dict[str, str]]) -> list[str]:
    names = sorted({item["label"] for item in train})
    if len(names) < 2:
        raise ValueError("Training data needs at least two labels")
    for split in other_splits:
        unknown = {item["label"] for item in split} - set(names)
        if unknown:
            raise ValueError(f"Unknown labels outside training data: {sorted(unknown)}")
    return names
