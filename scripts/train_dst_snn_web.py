#!/usr/bin/env python3
"""Run autonomous Playwright web observations into the DST-SNN online learner."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dst_snn.web_autonomous_learner import main  # noqa: E402


if __name__ == "__main__":
    main()
