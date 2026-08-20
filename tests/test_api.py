import io
import os
import sys

from fastapi.testclient import TestClient
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))
from main import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_predict_rejects_bad_content_type():
    fake_file = io.BytesIO(b"not an image")
    resp = client.post(
        "/predict", files={"file": ("test.txt", fake_file, "text/plain")}
    )
    assert resp.status_code == 400


def test_predict_returns_valid_response_shape_when_model_missing():
    # With no checkpoint present (fresh CI checkout), the API should
    # respond 503 rather than crash.
    img = Image.new("RGB", (224, 224), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    resp = client.post("/predict", files={"file": ("leaf.jpg", buf, "image/jpeg")})
    assert resp.status_code in (200, 503)


def test_metrics_summary_endpoint():
    resp = client.get("/metrics/summary")
    assert resp.status_code == 200
    assert "total_predictions" in resp.json()
