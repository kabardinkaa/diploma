from typing import Annotated

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from app.chat.deps import ChatRepositoryDep, SettingsDep
from app.chat.repository import ChatRepository
from app.security.tokens import secret_matches


admin_token_header = APIKeyHeader(
    name="X-Admin-Token",
    scheme_name="AdminToken",
    description="Operator token for administrative and approved write operations.",
    auto_error=False,
)
internal_token_header = APIKeyHeader(
    name="X-Internal-Token",
    scheme_name="InternalToken",
    description="Service-to-service token used by trusted internal clients.",
    auto_error=False,
)


def _forbidden(message: str) -> HTTPException:
    return HTTPException(status_code=403, detail=message)


async def require_admin(
    settings: SettingsDep,
    x_admin_token: Annotated[str | None, Security(admin_token_header)] = None,
) -> None:
    if not x_admin_token:
        raise _forbidden("Admin token required")
    if not secret_matches(x_admin_token, settings.admin_token):
        raise _forbidden("Invalid admin token")


async def require_internal(
    settings: SettingsDep,
    x_internal_token: Annotated[str | None, Security(internal_token_header)] = None,
) -> None:
    if not x_internal_token:
        raise _forbidden("Internal token required")
    if not secret_matches(x_internal_token, settings.internal_token):
        raise _forbidden("Invalid internal token")


async def require_internal_in_public(
    settings: SettingsDep,
    x_internal_token: Annotated[str | None, Security(internal_token_header)] = None,
) -> None:
    if settings.environment.lower() not in {"prod", "production", "public"}:
        return
    await require_internal(settings, x_internal_token)


AdminDep = Annotated[None, Depends(require_admin)]
InternalDep = Annotated[None, Depends(require_internal)]
RepositoryDep = ChatRepositoryDep
