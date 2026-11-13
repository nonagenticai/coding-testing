"""FastAPI application: 5 tenant-scoped endpoints + /health."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Annotated, Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from py2rs_fixture_svc.db import Note, Tag, get_session

UPSTREAM_TIMEOUT_SECONDS = 5.0


class NoteCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: uuid.UUID | None = Field(
        default=None, description="Client-supplied id (recommended for replay); uuid4 if omitted."
    )
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(max_length=10_000)
    tag: str | None = Field(default=None, min_length=1, max_length=64)


class NoteOut(BaseModel):
    id: uuid.UUID
    title: str
    body: str
    tag: str | None


class NoteList(BaseModel):
    items: list[NoteOut]


class EnrichedNote(NoteOut):
    summary: str
    sentiment: str


class Health(BaseModel):
    status: str


class ErrorOut(BaseModel):
    detail: str


def require_tenant(
    x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-Id")] = None,
) -> str:
    """Trusted-header auth, as in the coding services: the gateway injects X-Tenant-Id."""
    if x_tenant_id is None or not x_tenant_id.strip():
        raise HTTPException(status_code=401, detail="missing_tenant")
    if len(x_tenant_id) > 64:
        raise HTTPException(status_code=401, detail="invalid_tenant")
    return x_tenant_id


def get_upstream_client() -> Iterator[httpx.Client]:
    base_url = os.environ.get("UPSTREAM_URL")
    if not base_url:
        raise HTTPException(status_code=502, detail="upstream_not_configured")
    with httpx.Client(base_url=base_url, timeout=UPSTREAM_TIMEOUT_SECONDS) as client:
        yield client


Tenant = Annotated[str, Depends(require_tenant)]
Db = Annotated[Session, Depends(get_session)]
Upstream = Annotated[httpx.Client, Depends(get_upstream_client)]

ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorOut},
    404: {"model": ErrorOut},
    422: {"model": ErrorOut},
}

app = FastAPI(title="py2rs-fixture-svc", version="0.1.0")


@app.exception_handler(RequestValidationError)
def _validation_error(_request: Request, _exc: RequestValidationError) -> JSONResponse:
    # Flattened on purpose: FastAPI's default 422 body is framework-specific and not portable.
    return JSONResponse(status_code=422, content={"detail": "invalid_request"})


def _to_out(note: Note) -> NoteOut:
    return NoteOut(
        id=note.id, title=note.title, body=note.body, tag=note.tag.name if note.tag else None
    )


def _load_note(db: Session, tenant: str, note_id: uuid.UUID) -> Note:
    note = db.scalar(select(Note).where(Note.tenant_id == tenant, Note.id == note_id))
    if note is None:
        raise HTTPException(status_code=404, detail="note_not_found")
    return note


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok")


@app.post(
    "/notes",
    status_code=201,
    response_model=NoteOut,
    responses=ERRORS | {409: {"model": ErrorOut}},
)
def create_note(payload: NoteCreate, tenant: Tenant, db: Db) -> NoteOut:
    note_id = payload.id or uuid.uuid4()
    tag: Tag | None = None
    if payload.tag is not None:
        tag = db.scalar(select(Tag).where(Tag.tenant_id == tenant, Tag.name == payload.tag))
        if tag is None:
            # Tag id derived from (tenant, name) so replays produce identical rows.
            tag = Tag(
                id=uuid.uuid5(uuid.NAMESPACE_URL, f"tag:{tenant}:{payload.tag}"),
                tenant_id=tenant,
                name=payload.tag,
            )
            db.add(tag)
    note = Note(id=note_id, tenant_id=tenant, title=payload.title, body=payload.body, tag=tag)
    db.add(note)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="note_exists") from exc
    return _to_out(note)


@app.get("/notes/{note_id}", response_model=NoteOut, responses=ERRORS)
def get_note(note_id: uuid.UUID, tenant: Tenant, db: Db) -> NoteOut:
    return _to_out(_load_note(db, tenant, note_id))


@app.get("/notes", response_model=NoteList, responses=ERRORS)
def list_notes(
    tenant: Tenant,
    db: Db,
    tag: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
) -> NoteList:
    stmt = select(Note).where(Note.tenant_id == tenant)
    if tag is not None:
        stmt = stmt.join(Tag, Note.tag_id == Tag.id).where(Tag.name == tag)
    # Stable ordering: title, then id. Never created_at (wall clock).
    stmt = stmt.order_by(Note.title.asc(), Note.id.asc())
    return NoteList(items=[_to_out(n) for n in db.scalars(stmt).unique()])


@app.delete("/notes/{note_id}", status_code=204, response_class=Response, responses=ERRORS)
def delete_note(note_id: uuid.UUID, tenant: Tenant, db: Db) -> Response:
    note = _load_note(db, tenant, note_id)
    db.delete(note)
    db.commit()
    return Response(status_code=204)


@app.post(
    "/notes/{note_id}/enrich",
    response_model=EnrichedNote,
    responses=ERRORS | {502: {"model": ErrorOut}},
)
def enrich_note(note_id: uuid.UUID, tenant: Tenant, db: Db, upstream: Upstream) -> EnrichedNote:
    note = _load_note(db, tenant, note_id)
    request_body = {"note_id": str(note.id), "title": note.title, "text": note.body}
    try:
        resp = upstream.post("/enrich", json=request_body, headers={"X-Tenant-Id": tenant})
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="upstream_unavailable") from exc
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="upstream_error")
    try:
        data = resp.json()
        summary = data["summary"]
        sentiment = data["sentiment"]
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="upstream_bad_response") from exc
    if not isinstance(summary, str) or not isinstance(sentiment, str):
        raise HTTPException(status_code=502, detail="upstream_bad_response")
    base = _to_out(note)
    return EnrichedNote(**base.model_dump(), summary=summary, sentiment=sentiment)
