def test_public_html_routes(client):
    assert client.get("/").status_code == 200
    assert client.get("/index.html").status_code == 200
    assert client.get("/app/pages/python.html").status_code == 200


def test_repo_files_not_exposed(client):
    assert client.get("/requirements.txt").status_code == 404
    assert client.get("/wsgi.py").status_code == 404


def test_pages_only_serve_html(client):
    assert client.get("/app/pages/not-a-real-page.html").status_code == 404


def test_health_endpoints(client):
    hz = client.get("/healthz")
    assert hz.status_code == 200
    assert hz.get_json() == {"status": "ok"}
    rz = client.get("/readyz")
    assert rz.status_code == 200
    assert rz.get_json() == {"status": "ready"}


def test_security_headers(client):
    r = client.get("/")
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "SAMEORIGIN"


def test_onnx_route_rejects_non_onnx(client):
    assert client.get("/app/onnx_models/not-a-model.txt").status_code == 404


def test_onnx_models_served_when_present(client):
    from pathlib import Path

    onnx_path = (
        Path(__file__).resolve().parents[1] / "app" / "onnx_models" / "kmeans.onnx"
    )
    if not onnx_path.is_file():
        return
    r = client.get("/app/onnx_models/kmeans.onnx")
    assert r.status_code == 200
    assert len(r.data) > 32
