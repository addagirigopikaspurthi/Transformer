"""Optional end-to-end web API check using the small sample dataset."""

import time

from fastapi.testclient import TestClient

from web_app.server import app


def main() -> None:
    client = TestClient(app)
    response = client.post("/api/runs", json={
        "dataset_id": "sample",
        "model_name": "hf-internal-testing/tiny-random-bert",
        "epochs": 1,
        "batch_size": 4,
        "max_length": 64,
        "test_fraction": 0.25,
        "validation_fraction": 0.25,
    })
    response.raise_for_status()
    run_id = response.json()["id"]
    print(f"Started {run_id}", flush=True)
    for _ in range(120):
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] == "complete":
            break
        if run["status"] == "failed":
            raise RuntimeError(run.get("error", "Training failed"))
        time.sleep(1)
    else:
        raise TimeoutError("Training did not finish within two minutes")
    print(f"Test macro F1: {run['metrics']['test']['macro']['f1']:.3f}")
    prediction = client.post(f"/api/runs/{run_id}/predict", json={"text": "The service was excellent"})
    prediction.raise_for_status()
    print(f"Prediction endpoint: {prediction.json()['label']}")


if __name__ == "__main__":
    main()
