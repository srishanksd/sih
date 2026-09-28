import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))
from core.pipeline import run
from services.synthetic import field, sensors


def test_pipeline_returns_forecast():
    result = run(field(5), field(4), sensors(), 60)
    assert "objects" in result
    assert "hazards" in result
    assert result["quality"] > 0
