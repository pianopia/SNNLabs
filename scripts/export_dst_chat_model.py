#!/usr/bin/env python3
"""Convert a DST-SNN .pt checkpoint into snn-chat-lab compatible JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dst_snn.chat_export import export_checkpoint_for_chat  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = export_checkpoint_for_chat(args.checkpoint, args.output)
    print(output)


if __name__ == "__main__":
    main()
