from typing import Annotated

from fastapi import Depends, Header, HTTPException

from app.chat.deps import ChatRepositoryDep, SettingsDep
from app.chat.repository import ChatRepository


async def require_admin(
    settings: SettingsDep,
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> None:
    token = settings.admin_token

    if token is None or not x_admin_token:
        raise HTTPException(status_code=403, detail="Admin token required")

    if x_admin_token != token.get_secret_value():
        raise HTTPException(status_code=403, detail="Invalid admin token")


async def require_internal(
    settings: SettingsDep,
    x_internal_token: Annotated[str | None, Header(alias="X-Internal-Token")] = None,
) -> None:
    token = settings.internal_token

    if token is None or not x_internal_token:
        raise HTTPException(status_code=403, detail="Internal token required")

    if x_internal_token != token.get_secret_value():
        raise HTTPException(status_code=403, detail="Invalid internal token")


AdminDep = Annotated[None, Depends(require_admin)]
InternalDep = Annotated[None, Depends(require_internal)]
RepositoryDep = ChatRepositoryDep
