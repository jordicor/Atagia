"""Routes for user-controlled verbatim pins."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query, status

from atagia.api.dependencies import AuthContext, ensure_user_access, get_auth_context, get_connection, get_runtime
from atagia.models.schemas_api import (
    VerbatimPinCreateRequest,
    VerbatimPinRecord,
    VerbatimPinUpdateRequest,
)
from atagia.models.schemas_memory import MemoryScope, VerbatimPinStatus, VerbatimPinTargetKind
from atagia.services.verbatim_pin_service import VerbatimPinService

if TYPE_CHECKING:
    from atagia.app import AppRuntime

router = APIRouter(prefix="/v1/verbatim-pins", tags=["verbatim-pins"])


@router.post("", response_model=VerbatimPinRecord)
async def create_verbatim_pin(
    payload: VerbatimPinCreateRequest,
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    runtime: AppRuntime = Depends(get_runtime),
) -> VerbatimPinRecord:
    ensure_user_access(payload.user_id, auth_context)
    created = await VerbatimPinService(runtime).create_verbatim_pin(
        connection,
        **payload.model_dump(),
    )
    return VerbatimPinRecord.model_validate(created)


@router.get("", response_model=list[VerbatimPinRecord])
async def list_verbatim_pins(
    user_id: str = Query(...),
    status_filter: list[VerbatimPinStatus] | None = Query(default=None, alias="status"),
    scope_filter: list[MemoryScope] | None = Query(default=None, alias="scope"),
    target_kind_filter: list[VerbatimPinTargetKind] | None = Query(default=None, alias="target_kind"),
    target_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    include_deleted: bool = Query(default=False),
    active_only: bool = Query(default=False),
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    runtime: AppRuntime = Depends(get_runtime),
) -> list[VerbatimPinRecord]:
    ensure_user_access(user_id, auth_context)
    rows = await VerbatimPinService(runtime).list_verbatim_pins(
        connection,
        user_id=user_id,
        limit=limit,
        offset=offset,
        scope_filter=scope_filter,
        target_kind_filter=target_kind_filter,
        status_filter=status_filter,
        target_id=target_id,
        include_deleted=include_deleted,
        active_only=active_only,
    )
    return [VerbatimPinRecord.model_validate(row) for row in rows]


@router.get("/{pin_id}", response_model=VerbatimPinRecord)
async def get_verbatim_pin(
    pin_id: str,
    user_id: str = Query(...),
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    runtime: AppRuntime = Depends(get_runtime),
) -> VerbatimPinRecord:
    ensure_user_access(user_id, auth_context)
    row = await VerbatimPinService(runtime).get_verbatim_pin(
        connection,
        user_id=user_id,
        pin_id=pin_id,
    )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Verbatim pin not found for user")
    return VerbatimPinRecord.model_validate(row)


@router.patch("/{pin_id}", response_model=VerbatimPinRecord)
async def update_verbatim_pin(
    pin_id: str,
    payload: VerbatimPinUpdateRequest,
    user_id: str = Query(...),
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    runtime: AppRuntime = Depends(get_runtime),
) -> VerbatimPinRecord:
    ensure_user_access(user_id, auth_context)
    updated = await VerbatimPinService(runtime).update_verbatim_pin(
        connection,
        user_id=user_id,
        pin_id=pin_id,
        **payload.model_dump(exclude_none=True),
    )
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Verbatim pin not found for user")
    return VerbatimPinRecord.model_validate(updated)


@router.delete("/{pin_id}", response_model=VerbatimPinRecord)
async def delete_verbatim_pin(
    pin_id: str,
    user_id: str = Query(...),
    auth_context: AuthContext = Depends(get_auth_context),
    connection: aiosqlite.Connection = Depends(get_connection),
    runtime: AppRuntime = Depends(get_runtime),
) -> VerbatimPinRecord:
    ensure_user_access(user_id, auth_context)
    deleted = await VerbatimPinService(runtime).delete_verbatim_pin(
        connection,
        user_id=user_id,
        pin_id=pin_id,
    )
    if deleted is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Verbatim pin not found for user")
    return VerbatimPinRecord.model_validate(deleted)
