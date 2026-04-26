from __future__ import annotations

from typing import Any

import datasets
import torch
from torch.utils.data import DataLoader


def prepare_dataloaders(
    tokenizer: Any,
    *,
    dataset_path: str,
    dataset_name: str | None = None,
    text_column: str = "text",
    label_names_column: str = "label",
    max_length: int = 64,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader, datasets.DatasetDict]:
    """Load a text-classification dataset and format it for causal-LM fine-tuning."""
    dataset = datasets.load_dataset(dataset_path, dataset_name)
    label_feature = dataset["train"].features[label_names_column]
    label_names = getattr(label_feature, "names", None)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def format_examples(examples: dict[str, list[Any]]) -> dict[str, torch.Tensor]:
        texts = []
        for text, label in zip(examples[text_column], examples[label_names_column]):
            label_text = label_names[label] if label_names is not None else str(label)
            texts.append(f"Tweet text: {text} Label : {label_text}")

        tokenized = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tokenized["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        tokenized["labels"] = labels
        return tokenized

    columns_to_remove = dataset["train"].column_names
    tokenized_dataset = dataset.map(
        format_examples,
        batched=True,
        remove_columns=columns_to_remove,
    )
    tokenized_dataset.set_format("torch")

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        tokenized_dataset["validation"],
        shuffle=False,
        batch_size=batch_size,
    )
    return train_dataloader, eval_dataloader, dataset
