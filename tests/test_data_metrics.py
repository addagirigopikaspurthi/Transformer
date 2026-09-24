import tempfile
import unittest
from pathlib import Path

from text_classifier.data import label_names, read_examples, split_examples, write_examples
from text_classifier.metrics import classification_metrics


class DataAndMetricTests(unittest.TestCase):
    def test_split_is_reproducible_and_disjoint(self):
        examples = [
            {"text": f"example {label} {number}", "label": label}
            for label in ("negative", "positive")
            for number in range(40)
        ]
        first = split_examples(examples, 0.2, 0.2, 7)
        second = split_examples(examples, 0.2, 0.2, 7)
        self.assertEqual(first, second)
        self.assertEqual(sum(map(len, first)), len(examples))
        self.assertEqual(len({item["text"] for part in first for item in part}), len(examples))
        self.assertEqual(label_names(*first), ["negative", "positive"])

    def test_csv_rejects_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("text,label\nhello,\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                read_examples(path)

    def test_csv_round_trip(self):
        examples = [{"text": "hello, world", "label": "positive"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            write_examples(path, examples)
            self.assertEqual(read_examples(path), examples)

    def test_metrics_include_each_class_and_confusion_matrix(self):
        result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["no", "yes"])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro"]["recall"], 0.75)
        self.assertEqual(result["per_class"]["yes"]["support"], 2)


if __name__ == "__main__":
    unittest.main()
