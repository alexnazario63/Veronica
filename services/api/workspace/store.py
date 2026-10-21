"""Async, owner-scoped artifact history backed by SQLAlchemy 2 and SQLite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class _Base(DeclarativeBase):
    pass


class _ArtifactVersion(_Base):
    __tablename__ = "netcanvas_artifact_versions"
    artifact_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(200), index=True)
    artifact_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(40))


class ArtifactStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._engine: AsyncEngine | None = None
        self._sessions: async_sessionmaker[AsyncSession] | None = None

    async def initialize(self) -> None:
        if self._engine is not None:
            return
        url = f"sqlite+aiosqlite:///{self.path.as_posix()}"
        self._engine = create_async_engine(url)
        self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)
        async with self._engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)

    async def save(self, owner_id: str, artifact: dict[str, Any], created_at: str) -> dict[str, Any]:
        await self.initialize()
        aid, version = str(artifact["id"]), int(artifact["version"])
        payload = json.dumps(artifact, ensure_ascii=False, sort_keys=True, default=str)
        assert self._sessions is not None
        async with self._sessions() as session:
            existing = await session.get(_ArtifactVersion, (aid, version))
            if existing and existing.owner_id != owner_id:
                raise PermissionError("Artifact belongs to another user.")
            if existing:
                existing.artifact_json = payload
                existing.created_at = created_at
            else:
                session.add(_ArtifactVersion(artifact_id=aid, version=version,
                    owner_id=owner_id, artifact_json=payload, created_at=created_at))
            await session.commit()
        return artifact

    async def list(self, owner_id: str) -> list[dict[str, Any]]:
        await self.initialize()
        assert self._sessions is not None
        async with self._sessions() as session:
            result = await session.execute(select(_ArtifactVersion).where(
                _ArtifactVersion.owner_id == owner_id).order_by(_ArtifactVersion.created_at.desc()))
            rows = result.scalars().all()
            return [{"artifact": json.loads(row.artifact_json), "created_at": row.created_at} for row in rows]

    async def versions(self, owner_id: str, artifact_id: str) -> list[dict[str, Any]]:
        items = await self.list(owner_id)
        return [item for item in items if item["artifact"]["id"] == artifact_id]
