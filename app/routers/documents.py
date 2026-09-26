from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import structlog
from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.admin.deps import AdminDep
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    FileConflictError,
    IngestionBusyError,
    InvalidDocumentError,
    ReindexLimitError,
    UnsupportedFileTypeError,
)
from app.core.uploads import (
    normalize_safe_filename,
    publish_without_overwrite,
    save_upload_limited,
    validate_document_file,
)
from app.services.ingestion import (
    SUPPORTED_EXTENSIONS,
    IngestionService,
    validate_category,
)
from app.schemas.openapi import error_response, http_error_response

logger = structlog.get_logger("documents-router")
router = APIRouter(prefix="/documents", tags=["Documents / Ingestion"])
_ingestion_lock = asyncio.Lock()


class ReindexRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"mode": "incremental"},
                {"mode": "files", "files": ["vpn/vpn_access.md"]},
            ]
        }
    )

    mode: Literal["full", "incremental", "files"] = Field(
        description="Ingestion scope; full mode rebuilds only `corporate_rag`"
    )
    files: list[str] | None = Field(
        default=None,
        description="Relative corpus paths required when mode is `files`",
    )

    @model_validator(mode="after")
    def validate_files_mode(self) -> "ReindexRequest":
        if self.mode == "files" and not self.files:
            raise ValueError("files must be non-empty when mode='files'")
        return self


class UploadAccepted(BaseModel):
    status: Literal["accepted"]
    file_name: str
    category: str


class ReindexAccepted(BaseModel):
    status: Literal["accepted"]
    mode: Literal["full", "incremental", "files"]


async def _reserve_ingestion() -> None:
    if _ingestion_lock.locked():
        raise IngestionBusyError()
    await _ingestion_lock.acquire()


def _release_ingestion() -> None:
    if _ingestion_lock.locked():
        _ingestion_lock.release()


async def _validate_with_parser(path: Path, root: Path, settings: Settings) -> None:
    service = IngestionService(settings)
    try:
        await service.parse_file(path, root)
    except Exception:
        raise InvalidDocumentError() from None
    finally:
        await service.close()


async def _ingest_uploaded(path: Path, root: Path) -> None:
    service: IngestionService | None = None
    try:
        service = IngestionService(get_settings())
        report = await service.ingest_files([path], root=root)
        if report.failed_files:
            failed_path = path.with_name(f"{path.name}.failed")
            failed_path.unlink(missing_ok=True)
            logger.warning("rag.upload_invalid_document", file=path.name)
    except Exception:
        logger.exception("rag.upload_ingestion_failed", file=path.name)
    finally:
        if service is not None:
            await service.close()
        _release_ingestion()


def _select_reindex_files(
    payload: ReindexRequest,
    root: Path,
    settings: Settings,
) -> list[Path]:
    root_resolved = root.resolve()
    if not root.is_dir():
        raise InvalidDocumentError()
    if payload.mode in {"full", "incremental"}:
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.resolve().is_relative_to(root_resolved)
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and not path.name.endswith(".failed")
        )
    else:
        files = []
        seen: set[Path] = set()
        for item in payload.files or []:
            item_path = Path(item)
            try:
                normalized = normalize_safe_filename(item_path.name)
            except InvalidDocumentError:
                raise InvalidDocumentError() from None
            path = (root / item).resolve()
            if (
                normalized != item_path.name
                or root_resolved not in path.parents
                or not path.is_file()
                or path.suffix.lower() not in SUPPORTED_EXTENSIONS
            ):
                raise InvalidDocumentError()
            if path not in seen:
                files.append(path)
                seen.add(path)

    if payload.mode == "full" and not files:
        raise InvalidDocumentError()
    if len(files) > settings.reindex_max_files:
        raise ReindexLimitError()
    try:
        total_bytes = sum(path.stat().st_size for path in files)
    except OSError:
        raise InvalidDocumentError() from None
    if total_bytes > settings.reindex_max_total_bytes:
        raise ReindexLimitError()
    return files


async def _run_reindex(
    payload: ReindexRequest,
    files: list[Path],
    root: Path,
) -> None:
    settings = get_settings()
    service: IngestionService | None = None
    try:
        service = IngestionService(settings)
        mode: Literal["incremental", "full"] = (
            "full" if payload.mode == "full" else "incremental"
        )
        await service.ingest_files(files, root=root, mode=mode)
    except Exception:
        logger.exception("rag.reindex_failed", mode=payload.mode)
    finally:
        if service is not None:
            await service.close()
        _release_ingestion()


@router.post(
    "/upload",
    status_code=202,
    response_model=UploadAccepted,
    summary="Загрузить документ в production corpus",
    description=(
        "Administrative multipart upload protected by `X-Admin-Token`. Accepts "
        "bounded PDF, DOCX, HTML/HTM, or Markdown after filename, MIME/signature, "
        "archive, and parser validation. Existing corpus files are never overwritten."
    ),
    responses={
        400: error_response(
            "Unsafe filename",
            code="invalid_document",
            message="Недопустимое имя файла",
        ),
        403: http_error_response(
            "Missing or invalid admin token",
            detail="Admin token required",
        ),
        409: error_response(
            "File already exists or another ingestion operation is running",
            code="file_conflict",
            message="Файл с таким именем уже существует",
        ),
        413: error_response(
            "Upload exceeds the configured size or archive limits",
            code="payload_too_large",
            message="Размер файла превышает допустимый лимит",
        ),
        415: error_response(
            "Unsupported extension, MIME type, or signature",
            code="unsupported_file_type",
            message="Тип файла не поддерживается",
        ),
        422: error_response(
            "Document is corrupt or fails parser validation",
            code="invalid_document",
            message="Документ повреждён или не соответствует заявленному формату",
        ),
    },
)
async def upload_document(
    background_tasks: BackgroundTasks,
    _: AdminDep,
    file: UploadFile = File(...),
    category: str = Form(default="uploads"),
) -> dict[str, str]:
    settings = get_settings()
    safe_name = normalize_safe_filename(file.filename or "")
    extension = Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError()
    try:
        safe_category = validate_category(category)
    except ValueError:
        raise InvalidDocumentError() from None

    root = Path(settings.rag_data_dir)
    target_dir = root / safe_category
    target = target_dir / safe_name
    if target.exists():
        raise FileConflictError()

    await _reserve_ingestion()
    scheduled = False
    temporary_path: Path | None = None
    try:
        if target.exists():
            raise FileConflictError()
        temporary_path = await save_upload_limited(
            file,
            target_dir,
            suffix=extension,
            max_bytes=settings.document_upload_max_bytes,
        )
        validate_document_file(
            temporary_path,
            extension=extension,
            content_type=file.content_type,
            archive_max_entries=settings.document_archive_max_entries,
            archive_max_uncompressed_bytes=(
                settings.document_archive_max_uncompressed_bytes
            ),
            archive_max_compression_ratio=(
                settings.document_archive_max_compression_ratio
            ),
        )
        await _validate_with_parser(temporary_path, root, settings)
        publish_without_overwrite(temporary_path, target)
        temporary_path = None
        background_tasks.add_task(_ingest_uploaded, target, root)
        scheduled = True
        return {
            "status": "accepted",
            "file_name": safe_name,
            "category": safe_category,
        }
    finally:
        await file.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if not scheduled:
            _release_ingestion()


@router.post(
    "/reindex",
    status_code=202,
    response_model=ReindexAccepted,
    summary="Запустить переиндексацию corpus",
    description=(
        "Administrative asynchronous reindex protected by `X-Admin-Token`. "
        "Operations are serialized and bounded by configured file-count and total-size "
        "limits. Full mode affects only `corporate_rag` and its ingestion state."
    ),
    responses={
        403: http_error_response(
            "Missing or invalid admin token",
            detail="Admin token required",
        ),
        409: error_response(
            "Another ingestion operation is running",
            code="ingestion_busy",
            message="Операция индексации уже выполняется",
        ),
        413: error_response(
            "Reindex exceeds configured file-count or total-size limits",
            code="reindex_limit_exceeded",
            message="Объём операции переиндексации превышает допустимый лимит",
        ),
        422: error_response(
            "Invalid mode, path, file list, or empty full corpus",
            code="invalid_document",
            message="Документ повреждён или не соответствует заявленному формату",
        ),
    },
)
async def reindex_documents(
    payload: ReindexRequest,
    background_tasks: BackgroundTasks,
    _: AdminDep,
) -> dict[str, str]:
    settings = get_settings()
    root = Path(settings.rag_data_dir)
    await _reserve_ingestion()
    scheduled = False
    try:
        files = _select_reindex_files(payload, root, settings)
        background_tasks.add_task(_run_reindex, payload, files, root)
        scheduled = True
        return {"status": "accepted", "mode": payload.mode}
    finally:
        if not scheduled:
            _release_ingestion()
