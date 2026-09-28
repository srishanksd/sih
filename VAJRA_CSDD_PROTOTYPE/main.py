"""Single entry point for the VAJRA-CSDD application."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from app import app  # noqa: E402

if __name__ == "__main__":
    print("=" * 58)
    print(" VAJRA-CSDD | Convective Nowcasting Prototype")
    print("=" * 58)
    print("Dashboard: http://127.0.0.1:5000")
    print("API health: http://127.0.0.1:5000/api/health")
    print("Press CTRL+C to stop the server.\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
