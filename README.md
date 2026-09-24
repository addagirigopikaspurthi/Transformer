# Transformer Fine-Tuning & Evaluation

A local web app and command-line workflow for text classification using BERT, Hugging Face Transformers, and PyTorch. Upload labeled CSV data, fine-tune a model, track progress, inspect precision/recall/F1, and try predictions in the browser. Training selects the checkpoint with the highest validation macro F1 and evaluates once on a held-out test set.

## Setup

Use Python 3.11 or newer (tested with 3.12). Install a suitable [PyTorch build](https://pytorch.org/get-started/locally/) for your CPU or GPU, then install the remaining dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows PowerShell, replace `.venv/bin/python` with `.venv\Scripts\python.exe`. Activate the virtual environment before using the `python` commands below, or replace `python` with that virtual environment executable. If `python` is not on PATH, use an installed Python executable to create the virtual environment.

The first training run downloads `google-bert/bert-base-uncased` from Hugging Face. Use `--model-name` to select a different pretrained classification backbone or a local model directory.

## Web app

Start the local server from the project directory:

```bash
python -m uvicorn web_app.server:app --host 127.0.0.1 --port 8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The dashboard has five sections: Overview, Datasets, Training, Evaluation, and Playground. Upload a CSV or select the bundled sample, choose BERT base for a real experiment, and start training. The tiny random BERT option checks the workflow quickly but its scores are not meaningful. Runs, uploaded CSVs, and saved models are stored locally in `web_data/` and remain available after a server restart. Use one server process; the background training queue is local to that process.

For a fast end-to-end API check, run `python -m scripts.smoke_web`. It downloads the tiny model on first use.

## Hosting the frontend on Vercel

The frontend can be hosted on Vercel. The training API needs a separate, continuously running Python container with persistent storage; the current in-process training thread and local model files do not fit a Vercel Function. A [Dockerfile](Dockerfile) is included for that backend.

1. Deploy the Docker container on a host that provides enough CPU/GPU memory for your chosen model. Attach a persistent volume at `/data`, expose its `PORT`, and give it a public HTTPS address. Keep this service at one instance because its training queue lives in process memory.
2. Set `TRANSFORMER_API_KEY` to a long private value on the backend. Set `TRANSFORMER_ALLOWED_ORIGINS` to your Vercel site's exact HTTPS origin (comma-separated if you use preview domains). The app prompts for the API key and keeps it in the browser session; it is never built into the static assets.
3. In the Vercel project, set `PUBLIC_API_URL` to the backend's HTTPS origin. Deploy this repository as an **Other** framework project. `npm run build` copies the frontend into `dist/` and writes its public backend URL. The build deliberately fails without this variable or if a Vercel deployment points to localhost.
4. Open the Vercel URL, enter the backend API key, upload data, and train. Uploaded CSVs, models, and run history remain on the backend volume.

The Vercel site serves only HTML, CSS, and JavaScript. Dataset uploads go directly to the protected backend, avoiding Vercel Function upload limits. Do not put the API key in `PUBLIC_API_URL` or any frontend environment variable. For a local build check, run `node scripts/build-vercel.mjs http://127.0.0.1:8765`.

## Data

Supply a UTF-8 CSV with a header and two columns named `text` and `label`. Labels may be strings or integers; every label needs at least three examples to make stratified train, validation, and test splits.

```csv
text,label
"The service was excellent",positive
"Delivery was late",negative
```

To create splits, then train:

```bash
python -m text_classifier split --input data/all.csv --output-dir data/splits
python -m text_classifier train --train data/splits/train.csv --validation data/splits/validation.csv --test data/splits/test.csv --output-dir outputs/experiment1
```

For a quick command check, use `examples/sentiment.csv` in place of `data/all.csv`. This tiny sample is only for trying the workflow; its scores do not measure real model quality.

You can also supply your own splits. Keep examples from the same source or author in one split if that matters for your task; the automatic split stratifies by label but cannot detect related examples.

Evaluate a saved model or classify new text:

```bash
python -m text_classifier evaluate --model-dir outputs/experiment1 --data data/splits/test.csv
python -m text_classifier predict --model-dir outputs/experiment1 --text "The service was excellent"
```

Run `python -m text_classifier --help` and each subcommand's `--help` for options such as batch size, epochs, learning rate, maximum sequence length, seed, and device.

Training writes a Hugging Face model and tokenizer, `labels.json`, `history.json`, and `metrics.json` to the output directory. The metrics file includes accuracy, macro and weighted precision/recall/F1, per-class scores, and the confusion matrix. No performance figure is claimed here: scores depend on the dataset and split you provide.

## Verification

```bash
python -m unittest discover -s tests -v
```
