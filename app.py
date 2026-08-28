from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    streamlit_app = Path(__file__).resolve().parent / "frontend" / "streamlit_app.py"
    runpy.run_path(str(streamlit_app), run_name="__main__")
