from uuid import UUID

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.admin.deps import AdminDep, InternalDep, RepositoryDep
from app.chat.domain import AdminStats, AdminUser, BroadcastTask


router = APIRouter(prefix="/chats/admin", tags=["admin"])


class BroadcastIn(BaseModel):
    message: str = Field(..., min_length=1)
    interface_filter: str | None = None


class BroadcastResultIn(BaseModel):
    sent: int = Field(ge=0)
    failed: int = Field(ge=0)
    status: str = "done"


@router.get("/stats", response_model=AdminStats)
async def stats(
    _: AdminDep,
    repository: RepositoryDep,
) -> AdminStats:
    return await repository.admin_stats()


@router.get("/users", response_model=list[AdminUser])
async def users(
    _: AdminDep,
    repository: RepositoryDep,
    limit: int = Query(default=50, ge=1, le=500),
) -> list[AdminUser]:
    return await repository.list_admin_users(limit=limit)


@router.post("/broadcast", response_model=BroadcastTask)
async def broadcast(
    request: BroadcastIn,
    _: AdminDep,
    repository: RepositoryDep,
) -> BroadcastTask:
    return await repository.create_broadcast(
        message=request.message,
        interface_filter=request.interface_filter,
    )


@router.get("/internal/broadcasts/pending", response_model=list[BroadcastTask])
async def pending_broadcasts(
    _: InternalDep,
    repository: RepositoryDep,
    limit: int = Query(default=10, ge=1, le=50),
) -> list[BroadcastTask]:
    return await repository.list_pending_broadcasts(limit=limit)


@router.post("/internal/broadcasts/{task_id}/result", response_model=BroadcastTask | None)
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
