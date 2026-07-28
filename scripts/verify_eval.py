from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PACKAGES = {
    "ragas": "ragas",
    "pandas": "pandas",
    "phoenix": "arize-phoenix",
    "anthropic": "anthropic",
    "openinference.instrumentation.llama_index": (
        "openinference-instrumentation-llama-index"
    ),
    "opentelemetry": "opentelemetry-sdk",
    "openai": "openai",
}


def main() -> None:
    os.environ.setdefault(
        "PHOENIX_WORKING_DIR",
        str(Path(".tmp/phoenix").resolve()),
    )

    print("Evaluation dependency versions:")
    for module_name, distribution_name in PACKAGES.items():
        importlib.import_module(module_name)
        version = importlib.metadata.version(distribution_name)
        print(f"- {distribution_name}: {version}")

    from ragas.metrics import discrete_metric
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    metric_classes = (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    print("RAGAS metric signatures:")
    for metric_class in metric_classes:
        print(f"- {metric_class.__name__}: {inspect.signature(metric_class.ascore)}")
    print(f"- discrete_metric: {inspect.signature(discrete_metric)}")
    from app.core.config import get_settings

    settings = get_settings()
    print("Evaluation runtime:")
    print(f"- provider: {settings.eval_judge_provider}")
    print(f"- judge model: {settings.eval_judge_model}")
    print(f"- base URL: {settings.eval_judge_base_url}")
    print(f"- embedding model: {settings.eval_embedding_model}")
    print(f"- judge max tokens: {settings.eval_judge_max_tokens}")
    print("Evaluation imports: OK")


if __name__ == "__main__":
    main()
