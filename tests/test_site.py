"""Static site validation tests."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = ROOT / "app" / "pages"
ONNX_DIR  = ROOT / "app" / "onnx_models"

sys.path.insert(0, str(ROOT / "scripts"))
from lessons import LESSONS  # noqa: E402


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
        [sys.executable, str(ROOT / "scripts" / "export_tutorial_onnx.py"), str(tmp_path)],
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
        assert (tmp_path / name).is_file(), f"Missing {name} after export"


def test_nav_links_consistent():
    """Every lesson page should link to all other lesson pages."""
    expected_links = [f for _, f, _, _ in LESSONS]
    for name in expected_links:
        page = PAGES_DIR / name
        assert page.is_file(), f"Missing lesson page {name}"
        html = page.read_text(encoding="utf-8", errors="ignore")
        for link in expected_links:
            if link != name:
                assert link in html, f"{name} is missing nav link to {link}"


def test_nav_in_sync():
    """Menus and previous/next links must match scripts/lessons.py (fix: python scripts/sync_nav.py)."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sync_nav.py"), "--check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
