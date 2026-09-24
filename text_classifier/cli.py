"""Command-line entry point for text classification."""

import argparse
import json

from .data import read_examples, split_examples, write_examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate a BERT text classifier")
    commands = parser.add_subparsers(dest="command", required=True)

    split = commands.add_parser("split", help="Create stratified train/validation/test CSVs")
    split.add_argument("--input", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--test-fraction", type=float, default=0.15)
    split.add_argument("--validation-fraction", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=42)

    train = commands.add_parser("train", help="Train and save the best validation checkpoint")
    train.add_argument("--train", required=True)
    train.add_argument("--validation", required=True)
    train.add_argument("--test", help="Optional held-out test CSV")
    train.add_argument("--output-dir", required=True)
    train.add_argument("--model-name", default="google-bert/bert-base-uncased")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--max-length", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    evaluate = commands.add_parser("evaluate", help="Score a saved model on labeled CSV data")
    evaluate.add_argument("--model-dir", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    predict = commands.add_parser("predict", help="Classify one text with a saved model")
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--text", required=True)
    predict.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "split":
            from pathlib import Path
            examples = read_examples(args.input)
            train, validation, test = split_examples(
                examples, args.test_fraction, args.validation_fraction, args.seed
            )
            output_dir = Path(args.output_dir)
            for name, part in (("train", train), ("validation", validation), ("test", test)):
                write_examples(output_dir / f"{name}.csv", part)
            result = {"train": len(train), "validation": len(validation), "test": len(test)}
        elif args.command == "train":
            from .workflow import train
            result = train(args)
        elif args.command == "evaluate":
            if args.batch_size < 1:
                raise ValueError("batch size must be positive")
            from .workflow import evaluate_saved
            result = evaluate_saved(args.model_dir, args.data, args.batch_size, args.device)
        else:
            from .workflow import predict
            result = predict(args.model_dir, args.text, args.device)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
