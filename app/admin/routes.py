from uuid import UUID

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict, Field

from app.admin.deps import AdminDep, InternalDep, RepositoryDep
from app.chat.domain import AdminStats, AdminUser, BroadcastTask
from app.schemas.openapi import error_response, http_error_response


router = APIRouter(prefix="/chats/admin")


class BroadcastIn(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Плановые работы начнутся в 20:00.",
                "interface_filter": "telegram",
            }
        }
    )

    message: str = Field(..., min_length=1)
    interface_filter: str | None = None


class BroadcastResultIn(BaseModel):
    sent: int = Field(ge=0)
    failed: int = Field(ge=0)
    status: str = "done"


@router.get(
    "/stats",
    response_model=AdminStats,
    tags=["Admin"],
    summary="Получить агрегированную статистику",
    description="Operator-only aggregate chat statistics.",
    responses={
        403: http_error_response(
            "Missing or invalid admin token",
            detail="Admin token required",
        )
    },
)
async def stats(
    _: AdminDep,
    repository: RepositoryDep,
) -> AdminStats:
    return await repository.admin_stats()


@router.get(
    "/users",
    response_model=list[AdminUser],
    tags=["Admin"],
    summary="Получить список пользователей",
    description="Operator-only bounded user activity view.",
    responses={
        403: http_error_response(
            "Missing or invalid admin token",
            detail="Admin token required",
        ),
        422: error_response(
            "Invalid pagination limit",
            code="validation_error",
            message="Ошибка валидации запроса",
        ),
    },
)
async def users(
    _: AdminDep,
    repository: RepositoryDep,
    limit: int = Query(default=50, ge=1, le=500),
) -> list[AdminUser]:
    return await repository.list_admin_users(limit=limit)


@router.post(
    "/broadcast",
    response_model=BroadcastTask,
    tags=["Admin"],
    summary="Создать broadcast task",
    description=(
        "Creates an operator broadcast task. Delivery is performed later by the "
        "trusted Telegram broadcast worker."
    ),
    responses={
        403: http_error_response(
            "Missing or invalid admin token",
            detail="Admin token required",
        ),
        422: error_response(
            "Request validation failed",
            code="validation_error",
            message="Ошибка валидации запроса",
        ),
    },
)
async def broadcast(
    request: BroadcastIn,
    _: AdminDep,
    repository: RepositoryDep,
) -> BroadcastTask:
    return await repository.create_broadcast(
        message=request.message,
        interface_filter=request.interface_filter,
    )


@router.get(
    "/internal/broadcasts/pending",
    response_model=list[BroadcastTask],
    tags=["Internal"],
    summary="Получить pending broadcasts",
    description=(
        "Internal service-to-service polling endpoint for the Telegram worker; "
        "not intended for public clients."
    ),
    responses={
        403: http_error_response(
            "Missing or invalid internal token",
            detail="Internal token required",
        ),
        422: error_response(
            "Invalid polling limit",
            code="validation_error",
            message="Ошибка валидации запроса",
        ),
    },
)
async def pending_broadcasts(
    _: InternalDep,
    repository: RepositoryDep,
    limit: int = Query(default=10, ge=1, le=50),
) -> list[BroadcastTask]:
    return await repository.list_pending_broadcasts(limit=limit)


@router.post(
    "/internal/broadcasts/{task_id}/result",
    response_model=BroadcastTask | None,
    tags=["Internal"],
    summary="Сохранить результат broadcast",
    description=(
        "Internal service-to-service acknowledgement from the Telegram worker; "
        "not intended for public clients."
    ),
    responses={
        403: http_error_response(
            "Missing or invalid internal token",
            detail="Internal token required",
        ),
        422: error_response(
            "Invalid task ID or result payload",
            code="validation_error",
            message="Ошибка валидации запроса",
        ),
    },
)
async def update_broadcast_result(
    task_id: UUID,
    request: BroadcastResultIn,
    _: InternalDep,
    repository: RepositoryDep,
) -> BroadcastTask | None:
    return await repository.update_broadcast_result(
        task_id=task_id,
        sent=request.sent,
        failed=request.failed,
        status=request.status,
    )
