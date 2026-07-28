from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.ingestion import IngestionService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index corporate support documents")
    parser.add_argument("root", type=Path, help="Corpus root directory")
    parser.add_argument(
        "--mode",
        choices=("incremental", "full"),
        default="incremental",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    service = IngestionService(get_settings())
    try:
        report = await service.ingest_path(args.root, mode=args.mode)
        print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
        return 1 if report.failed_files else 0
    finally:
        await service.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
