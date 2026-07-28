from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app.core.config import get_settings
from app.eval.config import chunk_256_variant
from app.services.ingestion import IngestionService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the isolated chunk_size=256 evaluation collection."
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/corporate"))
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    base_settings = get_settings()
    variant = chunk_256_variant(base_settings)
    settings = variant.apply(base_settings)
    service = IngestionService(settings)
    try:
        report = await service.full_reindex(args.corpus)
        print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
