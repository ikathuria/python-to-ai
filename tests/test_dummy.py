"""Static site validation tests."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = ROOT / "app" / "pages"
ONNX_DIR  = ROOT / "app" / "onnx_models"


def test_root_html_files_exist():
    assert (ROOT / "index.html").is_file()
    assert (ROOT / "404.html").is_file()


def test_all_pages_nonempty():
    pages = list(PAGES_DIR.glob("*.html"))
    assert len(pages) >= 9, f"Expected >=9 topic pages, found {len(pages)}"
    for page in pages:
        size = page.stat().st_size
        assert size > 100, f"{page.name} is suspiciously small ({size} bytes)"


def test_onnx_models_present():
    models = list(ONNX_DIR.glob("*.onnx"))
    assert len(models) >= 8, f"Expected >=8 ONNX models, found {len(models)}"
    for m in models:
        assert m.stat().st_size > 100, f"{m.name} looks empty"


def test_export_script_runs(tmp_path):
    """export_tutorial_onnx.py must exit 0 and produce all expected models."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_tutorial_onnx.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Export script failed:\n{result.stderr}"
    expected = [
        "kmeans.onnx",
        "naive_bayes.onnx",
        "decision_tree_iris.onnx",
        "knn_iris.onnx",
        "logistic_regression_titanic.onnx",
        "linear_regression_insurance.onnx",
        "pca_iris.onnx",
    ]
    for name in expected:
        assert (ONNX_DIR / name).is_file(), f"Missing {name} after export"


def test_nav_links_consistent():
    """Every content page should link to all other topic pages."""
    expected_links = [
        "python.html", "ml_basics.html", "supervised_learning.html",
        "unsupervised_learning.html", "deep_learning.html",
        "computer_vision.html", "natural_language_processing.html",
        "recommendation_system.html", "time_series.html",
    ]
    content_pages = [
        PAGES_DIR / p for p in expected_links
        if (PAGES_DIR / p).stat().st_size > 500
    ]
    for page in content_pages:
        html = page.read_text(encoding="utf-8", errors="ignore")
        for link in expected_links:
            if link != page.name:
                assert link in html, f"{page.name} is missing nav link to {link}"
