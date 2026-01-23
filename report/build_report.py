"""Build the LaTeX report from generated results."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.report.latex import build_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Medical QA report")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: report/)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip PDF compilation",
    )
    args = parser.parse_args()

    build_report(output_dir=args.output, compile_pdf=not args.no_compile)


if __name__ == "__main__":
    main()
