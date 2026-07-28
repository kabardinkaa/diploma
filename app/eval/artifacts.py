from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


METRIC_COLUMNS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "has_citation",
)


def load_golden_dataset(path: Path, *, minimum: int = 30) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) < minimum:
        raise ValueError(f"Golden dataset must contain at least {minimum} rows")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Golden row {index} must be an object")
        for field in ("user_input", "reference"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Golden row {index} has empty {field}")
        contexts = row.get("reference_contexts")
        if (
            not isinstance(contexts, list)
            or not contexts
            or any(not isinstance(item, str) or not item.strip() for item in contexts)
        ):
            raise ValueError(f"Golden row {index} has invalid reference_contexts")
    return rows


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result_paths(
    results_dir: Path,
    label: str,
    *,
    now: datetime | None = None,
) -> tuple[Path, Path]:
    timestamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d_%H%M%S")
    safe_label = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in label
    ).strip("_")
    if not safe_label:
        raise ValueError("Evaluation label must contain a letter or digit")
    stem = f"{timestamp}_{safe_label}"
    return results_dir / f"{stem}.csv", results_dir / f"{stem}_aggregate.json"


def _valid_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    means: dict[str, float | None] = {}
    for name in METRIC_COLUMNS:
        values = [
            number
            for row in rows
            if (number := _valid_number(row.get(name))) is not None
        ]
        means[name] = statistics.fmean(values) if values else None

    latencies = [
        number
        for row in rows
        if (number := _valid_number(row.get("latency_ms"))) is not None
    ]
    ordered = sorted(latencies)

    def percentile(fraction: float) -> float | None:
        if not ordered:
            return None
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "row_count": len(rows),
        "failed_count": sum(bool(row.get("error")) for row in rows),
        "mean_metrics": means,
        "latency_ms": {
            "avg": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(0.50),
            "p95": percentile(0.95),
        },
    }
