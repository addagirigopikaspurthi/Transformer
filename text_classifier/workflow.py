"""PyTorch training, evaluation, and inference."""

import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from .data import label_names, read_examples
from .metrics import classification_metrics


class TextDataset(Dataset):
    def __init__(self, examples: list[dict[str, str]], tokenizer, names: list[str], max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_to_id = {name: index for index, name in enumerate(names)}
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        encoded = self.tokenizer(example["text"], truncation=True, max_length=self.max_length)
        encoded["labels"] = self.label_to_id[example["label"]]
        return encoded


def select_device(choice: str) -> torch.device:
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(choice)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(examples, tokenizer, names, max_length, batch_size, shuffle):
    dataset = TextDataset(examples, tokenizer, names, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
    )


def evaluate_loader(model, loader, names, device) -> dict:
    model.eval()
    truth, predictions = [], []
    total_loss = 0.0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            size = batch["labels"].size(0)
            total_loss += outputs.loss.item() * size
            truth.extend(batch["labels"].cpu().tolist())
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
    metrics = classification_metrics(truth, predictions, names)
    metrics["loss"] = total_loss / len(truth)
    return metrics


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def train(args, on_epoch=None) -> dict:
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 2 or args.learning_rate <= 0:
        raise ValueError("epochs, batch size, max length, and learning rate must be positive")
    set_seed(args.seed)
    device = select_device(args.device)
    training = read_examples(args.train)
    validation = read_examples(args.validation)
    test = read_examples(args.test) if args.test else None
    names = label_names(training, validation, *([test] if test else []))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(names),
        id2label={index: name for index, name in enumerate(names)},
        label2id={name: index for index, name in enumerate(names)},
    ).to(device)
    train_loader = make_loader(training, tokenizer, names, args.max_length, args.batch_size, True)
    validation_loader = make_loader(validation, tokenizer, names, args.max_length, args.batch_size, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += outputs.loss.item() * batch["labels"].size(0)
        validation_metrics = evaluate_loader(model, validation_loader, names, device)
        entry = {
            "epoch": epoch,
            "train_loss": total_loss / len(training),
            "validation": validation_metrics,
        }
        history.append(entry)
        if on_epoch is not None:
            on_epoch(entry)
        print(f"epoch {epoch}/{args.epochs}: train_loss={entry['train_loss']:.4f} "
              f"val_loss={validation_metrics['loss']:.4f} "
              f"val_macro_f1={validation_metrics['macro']['f1']:.4f}", flush=True)
        if validation_metrics["macro"]["f1"] > best_f1:
            best_f1 = validation_metrics["macro"]["f1"]
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            write_json(output_dir / "labels.json", names)
            best_epoch = epoch
    write_json(output_dir / "history.json", history)
    result = {"best_epoch": best_epoch, "validation": history[best_epoch - 1]["validation"]}
    if test:
        best_model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
        test_loader = make_loader(test, tokenizer, names, args.max_length, args.batch_size, False)
        result["test"] = evaluate_loader(best_model, test_loader, names, device)
        print(f"test_macro_f1={result['test']['macro']['f1']:.4f}", flush=True)
    write_json(output_dir / "metrics.json", result)
    write_json(output_dir / "run_config.json", {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    })
    return result


def load_saved(model_dir: str, device_choice: str):
    path = Path(model_dir)
    names = json.loads((path / "labels.json").read_text(encoding="utf-8"))
    config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    device = select_device(device_choice)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    return model, tokenizer, names, config, device


def evaluate_saved(model_dir: str, data_path: str, batch_size: int, device_choice: str) -> dict:
    model, tokenizer, names, config, device = load_saved(model_dir, device_choice)
    examples = read_examples(data_path)
    unknown = {item["label"] for item in examples} - set(names)
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(unknown)}")
    loader = make_loader(examples, tokenizer, names, config["max_length"], batch_size, False)
    return evaluate_loader(model, loader, names, device)


def predict(model_dir: str, text: str, device_choice: str) -> dict:
    if not text.strip():
        raise ValueError("Text must not be empty")
    model, tokenizer, names, config, device = load_saved(model_dir, device_choice)
    model.eval()
    encoded = tokenizer(text, truncation=True, max_length=config["max_length"], return_tensors="pt")
    with torch.inference_mode():
        logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits[0]
        probabilities = logits.softmax(dim=-1).cpu().tolist()
    index = max(range(len(names)), key=lambda i: probabilities[i])
    return {"label": names[index], "confidence": probabilities[index],
            "probabilities": dict(zip(names, probabilities))}
