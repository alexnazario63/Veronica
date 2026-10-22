"""
Async and Sync Artifact History and Versioning Store backed by SQLite / SQLAlchemy 2.
Garante compatibilidade total com INTERFACES.md, suporte a tipos ricos de artefatos
(config_diff, change_plan_10_steps, diagnostic_report_11_steps, topology_blast_radius, rca_document),
histórico imutável de versões, cálculo de diffs e vínculo direto com DeterministicChangeController.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


SUPPORTED_ARTIFACT_KINDS: Set[str] = {
    "config_diff",
    "change_plan_10_steps",
    "diagnostic_report_11_steps",
    "topology_blast_radius",
    "rca_document",
    "rca_report",
    "change_plan",
    "markdown",
    "code",
    "generic",
}


def apply_patch_diff(original_content: str, patch_diff: str) -> str:
    """Aplica patch unificado (unified diff) ou substituição de bloco de texto."""
    if not patch_diff or not patch_diff.strip():
        return original_content

    patch_lines = patch_diff.splitlines()
    hunk_re = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")

    hunks: List[Dict[str, Any]] = []
    current_hunk: Optional[Dict[str, Any]] = None

    for line in patch_lines:
        m = hunk_re.match(line)
        if m:
            if current_hunk:
                hunks.append(current_hunk)
            current_hunk = {
                "old_start": int(m.group(1)),
                "old_len": int(m.group(2) or 1),
                "new_start": int(m.group(3)),
                "new_len": int(m.group(4) or 1),
                "lines": [],
            }
        elif current_hunk is not None:
            current_hunk["lines"].append(line)

    if current_hunk:
        hunks.append(current_hunk)

    if not hunks:
        # Se não há cabeçalhos @@, mas há linhas com marcadores + e -:
        orig_lines = original_content.splitlines()
        to_remove: List[str] = []
        to_add: List[str] = []
        for line in patch_lines:
            if line.startswith("-") and not line.startswith("---"):
                to_remove.append(line[1:].strip())
            elif line.startswith("+") and not line.startswith("+++"):
                to_add.append(line[1:])
        if to_remove or to_add:
            new_lines = []
            for ol in orig_lines:
                if ol.strip() in to_remove:
                    continue
                new_lines.append(ol)
            new_lines.extend(to_add)
            return "\n".join(new_lines)
        # Caso padrão: substituição de texto direto
        return patch_diff.strip()

    orig_lines = original_content.splitlines()
    # Aplica hunks de trás para frente para manter índices válidos
    hunks.sort(key=lambda h: h["old_start"], reverse=True)

    result_lines = list(orig_lines)
    for hunk in hunks:
        start_idx = max(0, hunk["old_start"] - 1)
        old_count = hunk["old_len"]
        new_block = []
        for l in hunk["lines"]:
            if l.startswith("+") and not l.startswith("+++"):
                new_block.append(l[1:])
            elif l.startswith(" ") or not l.startswith("-"):
                # Linha de contexto mantida
                if not l.startswith("---"):
                    new_block.append(l[1:] if l.startswith(" ") else l)

        result_lines[start_idx : start_idx + old_count] = new_block

    return "\n".join(result_lines)


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
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._engine: AsyncEngine | None = None
        self._sessions: async_sessionmaker[AsyncSession] | None = None
        self.initialize_sync()

    def _connect_sync(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def initialize_sync(self) -> None:
        """Cria e migra a tabela SQLite para garantir colunas ricas de metadados."""
        with self._connect_sync() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS netcanvas_artifact_versions (
                    artifact_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    owner_id TEXT NOT NULL,
                    title TEXT,
                    content TEXT,
                    kind TEXT,
                    author TEXT,
                    plan_id TEXT,
                    patch_diff TEXT,
                    artifact_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (artifact_id, version)
                )
                """
            )
            # Adiciona colunas ausentes caso o banco tenha sido criado por versões anteriores
            cursor = conn.execute("PRAGMA table_info(netcanvas_artifact_versions)")
            existing_cols = {row["name"] for row in cursor.fetchall()}
            for col_name, col_type in [
                ("title", "TEXT"),
                ("content", "TEXT"),
                ("kind", "TEXT"),
                ("author", "TEXT"),
                ("plan_id", "TEXT"),
                ("patch_diff", "TEXT"),
            ]:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE netcanvas_artifact_versions ADD COLUMN {col_name} {col_type}")
            conn.commit()

    async def initialize(self) -> None:
        self.initialize_sync()
        if self._engine is not None:
            return
        url = f"sqlite+aiosqlite:///{self.path.as_posix()}"
        self._engine = create_async_engine(url)
        self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)
        async with self._engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)

    # ==================== Métodos Contratuais (INTERFACES.md) ====================

    def save_artifact(
        self,
        artifact_id: str,
        title: str,
        content: str,
        kind: str,
        author: str,
        plan_id: Optional[str] = None,
        patch_diff: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tenant_id: str = "default",
    ) -> int:
        """
        Salva nova versão do artefato e retorna o número incremental da versão.
        Compatibilidade total com docs/INTERFACES.md Seção 6.
        """
        self.initialize_sync()
        aid = str(artifact_id).strip()
        kind_clean = str(kind).strip().lower()

        # Vincula automaticamente plan_id se for change_plan_10_steps
        if not plan_id and (aid.startswith("chg-") or kind_clean == "change_plan_10_steps"):
            plan_id = aid

        now_iso = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()

        with self._connect_sync() as conn:
            cursor = conn.execute(
                "SELECT MAX(version) AS max_v FROM netcanvas_artifact_versions WHERE artifact_id = ?",
                (aid,),
            )
            row = cursor.fetchone()
            current_max = row["max_v"] if row and row["max_v"] is not None else 0
            new_version = current_max + 1

            sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()

            artifact_dict: Dict[str, Any] = {
                "id": aid,
                "version": new_version,
                "title": title,
                "type": kind_clean,
                "kind": kind_clean,
                "author": author,
                "proposed_content": content,
                "content": content,
                "plan_id": plan_id,
                "patch_diff": patch_diff,
                "sha256_hash": sha256,
                "tenant_id": tenant_id,
                "created_at": now_iso,
            }
            if metadata:
                artifact_dict["metadata"] = metadata

            artifact_json = json.dumps(artifact_dict, ensure_ascii=False, sort_keys=True, default=str)

            conn.execute(
                """
                INSERT OR REPLACE INTO netcanvas_artifact_versions
                (artifact_id, version, owner_id, title, content, kind, author, plan_id, patch_diff, artifact_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    aid,
                    new_version,
                    author,
                    title,
                    content,
                    kind_clean,
                    author,
                    plan_id,
                    patch_diff,
                    artifact_json,
                    now_iso,
                ),
            )
            conn.commit()

        return new_version

    def get_artifact(self, artifact_id: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Recupera versão específica ou a versão mais recente do artefato."""
        self.initialize_sync()
        aid = str(artifact_id).strip()
        with self._connect_sync() as conn:
            if version is not None:
                cursor = conn.execute(
                    "SELECT * FROM netcanvas_artifact_versions WHERE artifact_id = ? AND version = ?",
                    (aid, int(version)),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM netcanvas_artifact_versions WHERE artifact_id = ? ORDER BY version DESC LIMIT 1",
                    (aid,),
                )
            row = cursor.fetchone()
            if not row:
                return None
            data = json.loads(row["artifact_json"])
            # Garante campos canônicos com colunas atualizadas
            data["id"] = row["artifact_id"]
            data["version"] = row["version"]
            if row["title"]:
                data["title"] = row["title"]
            if row["kind"]:
                data["kind"] = row["kind"]
                data["type"] = row["kind"]
            if row["author"]:
                data["author"] = row["author"]
            if row["content"]:
                data["content"] = row["content"]
                data["proposed_content"] = row["content"]
            if row["plan_id"] is not None:
                data["plan_id"] = row["plan_id"]
            if row["patch_diff"] is not None:
                data["patch_diff"] = row["patch_diff"]
            return data

    def list_artifacts(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lista a versão mais recente de todos os artefatos cadastrados."""
        self.initialize_sync()
        with self._connect_sync() as conn:
            cursor = conn.execute(
                """
                SELECT v.*
                FROM netcanvas_artifact_versions v
                INNER JOIN (
                    SELECT artifact_id, MAX(version) AS max_v
                    FROM netcanvas_artifact_versions
                    GROUP BY artifact_id
                ) latest ON v.artifact_id = latest.artifact_id AND v.version = latest.max_v
                ORDER BY v.created_at DESC
                """
            )
            rows = cursor.fetchall()
            artifacts: List[Dict[str, Any]] = []
            for r in rows:
                data = json.loads(r["artifact_json"])
                data["id"] = r["artifact_id"]
                data["version"] = r["version"]
                if r["title"]:
                    data["title"] = r["title"]
                if r["kind"]:
                    data["kind"] = r["kind"]
                    data["type"] = r["kind"]
                if r["author"]:
                    data["author"] = r["author"]
                if r["plan_id"] is not None:
                    data["plan_id"] = r["plan_id"]
                if r["patch_diff"] is not None:
                    data["patch_diff"] = r["patch_diff"]
                if tenant_id and data.get("tenant_id") and data.get("tenant_id") != tenant_id:
                    continue
                artifacts.append(data)
            return artifacts

    def get_version_history(self, artifact_id: str) -> List[Dict[str, Any]]:
        """Retorna histórico cronológico completo de versões de um artefato."""
        self.initialize_sync()
        aid = str(artifact_id).strip()
        with self._connect_sync() as conn:
            cursor = conn.execute(
                "SELECT * FROM netcanvas_artifact_versions WHERE artifact_id = ? ORDER BY version ASC",
                (aid,),
            )
            rows = cursor.fetchall()
            history: List[Dict[str, Any]] = []
            for r in rows:
                data = json.loads(r["artifact_json"])
                data["id"] = r["artifact_id"]
                data["version"] = r["version"]
                if r["title"]:
                    data["title"] = r["title"]
                if r["kind"]:
                    data["kind"] = r["kind"]
                    data["type"] = r["kind"]
                if r["author"]:
                    data["author"] = r["author"]
                if r["content"]:
                    data["content"] = r["content"]
                    data["proposed_content"] = r["content"]
                if r["plan_id"] is not None:
                    data["plan_id"] = r["plan_id"]
                if r["patch_diff"] is not None:
                    data["patch_diff"] = r["patch_diff"]
                data.setdefault("created_at", r["created_at"])
                history.append(data)
            return history

    def propose_patch(
        self,
        artifact_id: str,
        version: int,
        patch_diff: str,
        author: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Aplica patch sobre uma versão existente e gera a próxima versão incremental."""
        base_art = self.get_artifact(artifact_id, version=version)
        if not base_art:
            raise ValueError(f"Artifact '{artifact_id}' version {version} not found.")

        base_content = base_art.get("content") or base_art.get("proposed_content", "")
        new_content = apply_patch_diff(base_content, patch_diff)

        effective_author = author or base_art.get("author", "usr-eng-01")
        new_version = self.save_artifact(
            artifact_id=artifact_id,
            title=base_art.get("title", f"Artifact {artifact_id}"),
            content=new_content,
            kind=base_art.get("kind") or base_art.get("type", "generic"),
            author=effective_author,
            plan_id=base_art.get("plan_id"),
            patch_diff=patch_diff,
            metadata=base_art.get("metadata"),
            tenant_id=base_art.get("tenant_id", "default"),
        )

        updated_art = self.get_artifact(artifact_id, version=new_version)
        return {
            "artifact_id": artifact_id,
            "base_version": version,
            "new_version": new_version,
            "patch_diff": patch_diff,
            "status": "PROPOSED",
            "artifact": updated_art,
        }

    def diff_versions(self, artifact_id: str, v1: int, v2: int) -> Dict[str, Any]:
        """Compara duas versões históricas quaisquer do artefato e gera o unified diff."""
        art1 = self.get_artifact(artifact_id, version=v1)
        art2 = self.get_artifact(artifact_id, version=v2)
        if not art1 or not art2:
            raise ValueError(f"Versions ({v1}, {v2}) for artifact '{artifact_id}' not found.")

        c1 = (art1.get("content") or art1.get("proposed_content") or "").splitlines(keepends=True)
        c2 = (art2.get("content") or art2.get("proposed_content") or "").splitlines(keepends=True)

        diff = list(
            difflib.unified_diff(
                c1,
                c2,
                fromfile=f"{artifact_id}_v{v1}",
                tofile=f"{artifact_id}_v{v2}",
            )
        )
        diff_text = "".join(diff)
        lines_added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
        lines_removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

        return {
            "artifact_id": artifact_id,
            "v1": v1,
            "v2": v2,
            "diff": diff_text,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "identical": (diff_text == ""),
        }

    def link_change_plan(self, artifact_id: str, plan_id: str) -> None:
        """Vincula diretamente o artefato ao ID do plano no DeterministicChangeController."""
        self.initialize_sync()
        aid = str(artifact_id).strip()
        with self._connect_sync() as conn:
            cursor = conn.execute("SELECT version, artifact_json FROM netcanvas_artifact_versions WHERE artifact_id = ?", (aid,))
            rows = cursor.fetchall()
            for r in rows:
                v = r["version"]
                data = json.loads(r["artifact_json"])
                data["plan_id"] = plan_id
                conn.execute(
                    "UPDATE netcanvas_artifact_versions SET plan_id = ?, artifact_json = ? WHERE artifact_id = ? AND version = ?",
                    (plan_id, json.dumps(data, ensure_ascii=False, sort_keys=True, default=str), aid, v),
                )
            conn.commit()

    # ==================== Métodos Legados / Async (Compatibilidade) ====================

    async def save(self, owner_id: str, artifact: dict[str, Any], created_at: str) -> dict[str, Any]:
        await self.initialize()
        aid = str(artifact["id"])
        version = int(artifact["version"])
        title = str(artifact.get("title", f"Artifact {aid}"))
        content = str(artifact.get("proposed_content") or artifact.get("content", ""))
        kind = str(artifact.get("type") or artifact.get("kind", "generic"))
        plan_id = artifact.get("plan_id")
        patch_diff = artifact.get("patch_diff")
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

        # Sincroniza tabela com colunas estruturadas
        with self._connect_sync() as conn:
            conn.execute(
                """
                UPDATE netcanvas_artifact_versions
                SET title = ?, content = ?, kind = ?, author = ?, plan_id = ?, patch_diff = ?
                WHERE artifact_id = ? AND version = ?
                """,
                (title, content, kind, owner_id, plan_id, patch_diff, aid, version),
            )
            conn.commit()

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
