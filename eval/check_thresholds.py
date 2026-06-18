import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULT_THRESHOLDS = Path("eval/thresholds.yaml")
DEFAULT_RUNS_DIR = Path("eval/runs")


def load_thresholds(path: Path) -> dict[str, float]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {key: float(value) for key, value in data.items()}


def find_latest_run(runs_dir: Path) -> Path:
    runs = sorted(runs_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No evaluation runs found in {runs_dir}")

    return runs[-1]


def check_report(report: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    aggregates = report.get("aggregates", {})
    failures: list[str] = []

    for metric_name, threshold in thresholds.items():
        actual = aggregates.get(metric_name)
        if actual is None:
            failures.append(f"Missing aggregate metric: {metric_name}")
            continue

        if float(actual) < threshold:
            failures.append(
                f"{metric_name}={actual} is below threshold {threshold}"
            )

    min_correctness_threshold = thresholds.get("min_correctness")
    if min_correctness_threshold is not None:
        for item in report.get("items", []):
            correctness = float(item.get("scores", {}).get("correctness", 0))
            if correctness < min_correctness_threshold:
                failures.append(
                    f"{item.get('id')}: correctness={correctness} below "
                    f"{min_correctness_threshold}. Explanation: "
                    f"{item.get('explanation', '')}"
                )

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check eval run thresholds.")
    parser.add_argument("run", nargs="?", default=None)
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_path = Path(args.run) if args.run else find_latest_run(Path(args.runs_dir))
    thresholds = load_thresholds(Path(args.thresholds))
    report = json.loads(run_path.read_text(encoding="utf-8"))

    failures = check_report(report, thresholds)

    print(f"Checking run: {run_path}")
    print(json.dumps(report.get("aggregates", {}), ensure_ascii=False, indent=2))

    if failures:
        print("Threshold check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Threshold check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())