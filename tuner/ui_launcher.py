"""Entry point: `hpt-ui` runs Streamlit on the bundled app."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app = Path(__file__).resolve().parent / "ui_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
