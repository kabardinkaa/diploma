from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import AsyncQdrantClient

from app.core.config import get_settings
from app.services.ingestion import IngestionService, SUPPORTED_EXTENSIONS
from app.services.rag import RAGService

IN_BASE = [
    "Как восстановить доступ к VPN?",
    "Что делать, если CRM не принимает пароль?",
    "Почему не отправляется корпоративная почта?",
    "Как подтвердить вход через MFA?",
    "Что проверить, если не подключается рабочий Wi-Fi?",
    "Как настроить softphone для звонков?",
    "Почему собеседник не слышит микрофон гарнитуры?",
    "Как установить обязательное обновление программы?",
]

OUT_OF_BASE = [
    "Как оформить ежегодный отпуск?",
    "Какое сегодня меню в столовой?",
    "Когда перечисляют заработную плату?",
    "Где получить пропуск на парковку?",
    "Как согласовать зарубежную командировку?",
    "Кто заказывает торт на день рождения?",
    "Как получить справку для налоговой?",
    "Есть ли скидка на абонемент в спортзал?",
]


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 3),
        "median": round(statistics.median(values), 3),
        "max": round(max(values), 3),
    }


async def retrieval_check() -> dict:
    settings = get_settings()
    service = RAGService(settings)
    try:
        await service.build()
        groups: dict[str, list[dict]] = {"in_base": [], "out_of_base": []}
        for group, questions in (
            ("in_base", IN_BASE),
            ("out_of_base", OUT_OF_BASE),
        ):
            for question in questions:
                result = await service._retrieve(question, history=None)
                groups[group].append(
                    {
                        "question": question,
                        "top_score": result["top_score"],
                        "confident": result["confident"],
                        "source": (
                            result["sources"][0]["file_name"]
                            if result["sources"]
                            else None
                        ),
                    }
                )
        return {
            "collection": service.collection_name,
            "points_count": await service.points_count(),
            "threshold": settings.rag_min_score,
            "in_base_distribution": distribution(
                [item["top_score"] for item in groups["in_base"]]
            ),
            "out_of_base_distribution": distribution(
                [item["top_score"] for item in groups["out_of_base"]]
            ),
            **groups,
        }
    finally:
        await service.close()


async def incremental_check() -> dict:
    base_settings = get_settings()
    collection = "corporate_rag_incremental_smoke"
    with tempfile.TemporaryDirectory(prefix="corporate-rag-") as directory:
        root = Path(directory) / "data"
        source = PROJECT_ROOT / "data" / "corporate" / "vpn"
        target = root / "vpn"
        target.mkdir(parents=True)
        document = target / "vpn_smoke_v2026.md"
        shutil.copyfile(next(source.glob("*.md")), document)
        settings = base_settings.model_copy(
            update={
                "rag_data_dir": root,
                "rag_production_collection": collection,
                "rag_docstore_path": Path(directory) / "docstore.json",
            }
        )
        service = IngestionService(settings)
        api_key_value = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key
            else None
        )
        client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=api_key_value or None,
        )
        try:
            initial = await service.full_reindex(root)
            document.write_text(
                document.read_text(encoding="utf-8")
                + "\n\nУникальная проверка incremental update.\n",
                encoding="utf-8",
            )
            changed = await service.incremental_reindex(root)
            return {
                "initial_changed": initial.changed_files,
                "incremental_changed": changed.changed_files,
                "incremental_unchanged": changed.unchanged_files,
                "failed": changed.failed_files,
                "points_before": initial.points_count,
                "points_after": changed.points_count,
            }
        finally:
            await service.close()
            try:
                await client.delete_collection(collection)
            finally:
                await client.close()


def inventory() -> dict:
    root = PROJECT_ROOT / "data"
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return {
        "total": len(files),
        "formats": sorted({path.suffix.lower() for path in files}),
    }


async def main(run_incremental: bool) -> None:
    result = {
        "inventory": inventory(),
        "retrieval": await retrieval_check(),
    }
    if run_incremental:
        result["incremental"] = await incremental_check()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--incremental-check", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.incremental_check))
