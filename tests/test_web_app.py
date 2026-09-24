import unittest

from fastapi.testclient import TestClient

from web_app import server
from web_app.server import DATASETS, app


class WebAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_overview_includes_sample_dataset(self):
        response = self.client.get("/api/overview")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["datasets"][0]["id"], "sample")

    def test_upload_and_validation(self):
        payload = b"text,label\nhello,positive\ngoodbye,negative\n"
        response = self.client.post("/api/datasets", files={"file": ("reviews.csv", payload, "text/csv")})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["count"], 2)
        self.assertEqual(data["labels"], {"negative": 1, "positive": 1})
        too_small = self.client.post("/api/runs", json={"dataset_id": data["id"]})
        self.assertEqual(too_small.status_code, 400)
        (DATASETS / f"{data['id']}.csv").unlink()
        (DATASETS / f"{data['id']}.json").unlink()
        bad = self.client.post("/api/datasets", files={"file": ("bad.csv", b"wrong,columns\na,b\n", "text/csv")})
        self.assertEqual(bad.status_code, 400)

    def test_training_request_rejects_unknown_model(self):
        response = self.client.post("/api/runs", json={
            "dataset_id": "sample", "model_name": "untrusted/model"
        })
        self.assertEqual(response.status_code, 400)

    def test_upload_rejects_malformed_csv(self):
        before = set(DATASETS.glob("ds_*.csv"))
        response = self.client.post(
            "/api/datasets",
            files={"file": ("broken.csv", b'text,label\n"unterminated,positive\n', "text/csv")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(set(DATASETS.glob("ds_*.csv")), before)

    def test_access_key_guards_api(self):
        original = server.API_KEY
        server.API_KEY = "test-private-key"
        try:
            self.assertEqual(self.client.get("/api/overview").status_code, 401)
            response = self.client.get("/api/overview", headers={"Authorization": "Bearer test-private-key"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(self.client.get("/health").status_code, 200)
        finally:
            server.API_KEY = original


if __name__ == "__main__":
    unittest.main()
