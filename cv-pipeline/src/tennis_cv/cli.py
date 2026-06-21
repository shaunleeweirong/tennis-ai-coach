from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from .pipeline import analyze
from .types import AnalysisError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse a tennis serve clip.")
    parser.add_argument("video", help="Path to the serve video clip")
    parser.add_argument("--out", default="overlays", help="Directory for overlay images")
    args = parser.parse_args(argv)

    try:
        result = analyze(args.video, args.out)
    except AnalysisError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    payload = {
        "stroke_type": result.stroke_type.value,
        "overall_score": result.score.overall,
        "duration": result.duration,
        "metrics": [dataclasses.asdict(m) for m in result.metrics],
        "overlay_paths": result.overlay_paths,
        "coaching": result.coaching.model_dump(),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
