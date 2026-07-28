from __future__ import annotations

from pathlib import Path
from typing import Literal

import structlog
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, model_validator

from app.core.config import get_settings
from app.services.ingestion import (
    SUPPORTED_EXTENSIONS,
    IngestionService,
    validate_category,
)

logger = structlog.get_logger("documents-router")
router = APIRouter(prefix="/documents", tags=["documents"])


class ReindexRequest(BaseModel):
    mode: Literal["full", "incremental", "files"]
    files: list[str] | None = None

    @model_validator(mode="after")
    def validate_files_mode(self) -> "ReindexRequest":
        if self.mode == "files" and not self.files:
            raise ValueError("files must be non-empty when mode='files'")
        return self


async def _ingest_uploaded(path: Path, root: Path) -> None:
    service = IngestionService(get_settings())
    try:
        await service.ingest_files([path], root=root)
    except Exception:
        logger.exception("rag.upload_ingestion_failed", file=path.name)
    finally:
        await service.close()


async def _run_reindex(payload: ReindexRequest) -> None:
    settings = get_settings()
    root = Path(settings.rag_data_dir)
    service = IngestionService(settings)
    try:
        if payload.mode == "full":
            await service.full_reindex(root)
        elif payload.mode == "incremental":
            await service.incremental_reindex(root)
        else:
            selected: list[Path] = []
            root_resolved = root.resolve()
            for item in payload.files or []:
                path = (root / item).resolve()
                if root_resolved not in path.parents or not path.is_file():
                    raise ValueError(f"unsafe or missing file: {item}")
                if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    raise ValueError(f"unsupported file: {item}")
                selected.append(path)
            await service.ingest_files(selected, root=root)
    except Exception:
        logger.exception("rag.reindex_failed", mode=payload.mode)
    finally:
        await service.close()


@router.post("/upload", status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form(default="uploads"),
) -> dict[str, str]:
    original_name = file.filename or ""
    safe_name = Path(original_name).name
    if not safe_name or safe_name != original_name:
        raise HTTPException(status_code=400, detail="Unsafe filename")
    extension = Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported document format")
    try:
        safe_category = validate_category(category)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    root = Path(get_settings().rag_data_dir)
    target_dir = root / safe_category
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / safe_name
    target.write_bytes(await file.read())
    background_tasks.add_task(_ingest_uploaded, target, root)
    return {
        "status": "accepted",
        "file_name": safe_name,
        "category": safe_category,
    }


@router.post("/reindex", status_code=202)
async def reindex_documents(
    payload: ReindexRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    background_tasks.add_task(_run_reindex, payload)
    return {"status": "accepted", "mode": payload.mode}
