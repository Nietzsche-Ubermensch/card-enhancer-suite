"""
CLI entrypoint for the unified card-enhancement pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rewards import quality_score_from_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Card Enhancement Pipeline")
    parser.add_argument("image", type=Path, help="Input card image")
    parser.add_argument("--gigapixel", type=Path, help="Path to Gigapixel AI.exe")
    parser.add_argument("--elan", type=Path, help="Path to ELAN checkpoint")
    parser.add_argument("--scale", default="X2")
    parser.add_argument("--mode", default="STANDARD")
    args = parser.parse_args()

    # Stub: in production this would call the scanner + upscaler
    metadata = {
        "subjectName": args.image.stem,
        "cardNumber": None,
        "manufacturer": None,
        "year": None,
        "stats": {},
    }
    reward = quality_score_from_metadata(metadata)

    result = {
        "input": str(args.image),
        "metadata": metadata,
        "quality_reward": reward,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
