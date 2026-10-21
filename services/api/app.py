"""
API FastAPI e Streaming da Plataforma Veronica (Seções 12, 16.6, 16.7 e 16.8).
Expõe todos os endpoints documentados com autenticação/RBAC, validação de parâmetros,
canal de observabilidade (SSE) e entrega da interface gráfica web.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Body, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from core.contracts import NetCanvasArtifact, MessageDelta
from services.api.streaming import json_bytes, sse_frame
from services.api.workspace import ArtifactStore

from core.training import TrainingHyperparameters
from services.api.services_hub import VeronicaPlatformHub


# ==================== Schemas de Requisição ====================

class CreateConversationRequest(BaseModel):
    title: str = Field(default="Investigação BGP / OSPF Core")


class RenameConversationRequest(BaseModel):
    title: str


class SendMessageRequest(BaseModel):
    content: str
    modality: str = Field(default="TEXT")
    target_devices: Optional[List[str]] = None
    mode: Optional[str] = Field(default="chat")
    model_profile: Literal["vrn-fast", "vrn-pro", "vrn-ultra"] = "vrn-pro"
    response_style: Literal["technical", "natural", "mixed", "engineer"] = "technical"
    image_data: Optional[str] = Field(default=None)


class ConversationFeedbackRequest(BaseModel):
    message_id: str = Field(default="msg-manual")
    original_prompt: str
    incorrect_response: str
    corrected_response: str
    domain: str = Field(default="networking.general")
    vendor: str = Field(default="Generic")
    os_name: str = Field(default="Unspecified")
    os_version: str = Field(default="Unspecified")
    evidence_reference: str = Field(default="Feedback técnico via Chat GUI")
    auto_approve_memory: bool = Field(default=False)


class ApproveCorrectionRequest(BaseModel):
    dataset_version: str = Field(default="v1.1.0-curated")
    reviewer_notes: str = Field(default="Correção técnica validada contra documentação oficial.")


class VoiceTranscribeRequest(BaseModel):
    conversation_id: str
    raw_transcript: str = ""
    duration_sec: float = 4.0
    audio_size_bytes: int = 32000
    rms_energy_db: float = -20.0
    auto_send_to_chat: bool = True
    audio_data_base64: Optional[str] = None
    audio_format: str = "wav"


class VoiceConsentRequest(BaseModel):
    conversation_id: str
    consent: bool


class VoiceSynthesizeRequest(BaseModel):
    conversation_id: str
    text: str


class DocumentIngestRequest(BaseModel):
    title: str
    content: str
    file_format: str = "md"
    vendor: str = "Huawei"
    family_model: str = "NetEngine 8000"
    os_name: str = "VRP 8"
    version_build: str = "V800R022C00"
    feature: str = "Routing & MPLS"
    original_source: str = "Documentação Técnica Submetida"
    source_uri: str = "upload://manual-tecnico.md"
    page_section: str = "Seção 1"
    published_date: str = "2025-01-15"
    reliability_score: float = 0.80
    applicability_restrictions: str = "Validar versão exata de VRP antes do uso."
    is_official: bool = False


class StartExperimentRequest(BaseModel):
    epochs: int = Field(default=5, ge=1, le=50)
    batch_size: int = Field(default=4, ge=1, le=32)
    seq_len: int = Field(default=24, ge=8, le=128)
    learning_rate: float = Field(default=0.01, gt=0.0, le=0.5)
    min_lr: float = Field(default=0.001, gt=0.0, le=0.1)
    weight_decay: float = Field(default=0.01, ge=0.0, le=0.5)
    optimizer_type: str = Field(default="adamw")
    max_grad_norm: float = Field(default=1.0, gt=0.0, le=10.0)
    grad_accum_steps: int = Field(default=1, ge=1, le=16)
    run_async: bool = Field(default=False)


class ExperimentControlRequest(BaseModel):
    action: str  # "pause", "resume", "cancel"


class PromoteCheckpointRequest(BaseModel):
    checkpoint_id: str
    target_environment: str = "STAGING"  # "STAGING" ou "PRODUCTION"


class RestoreCheckpointRequest(BaseModel):
    checkpoint_id: str


class DiagnosticRequest(BaseModel):
    problem_description: str
    target_devices: Optional[List[str]] = None
    incomplete_telemetry_simulation: bool = False


class CreateChangePlanRequest(BaseModel):
    device_id: str
    title: str
    proposed_commands: List[str]
    maintenance_window: str = "2026-10-18 02:00-04:00 BRT"
    origin_channel: str = "TEXT_UI"


class ApproveChangeRequest(BaseModel):
    confirmed_plan_hash: str
    confirmed_maintenance_window: str
    signature_ed25519: str
    approval_timestamp: str
    approval_channel: str = "EXPLICIT_WEB_APPROVAL"


class ExecuteChangeRequest(BaseModel):
    approval_token: str


class SeeImageRequest(BaseModel):
    image_data: str
    context_hint: str = Field(default="")


class ReadTextRequest(BaseModel):
    content: str
    title: str = Field(default="Documento Técnico")


class AutonomousIntervalRequest(BaseModel):
    interval_seconds: Optional[float] = Field(default=None, ge=0.002, le=3600.0)
    interval_ms: Optional[float] = Field(default=None, ge=2.0, le=3600000.0)
    workers: Optional[int] = Field(default=None, ge=1, le=64)


class AdminRuntimeConfigRequest(BaseModel):
    workers: Optional[int] = Field(default=None, ge=1, le=64)
    interval_seconds: Optional[float] = Field(default=None, ge=0.002, le=3600.0)


class WorkspaceArtifactRequest(BaseModel):
    artifact: NetCanvasArtifact


class WorkspaceEditRequest(BaseModel):
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    replacement: str


def create_app(data_dir: Optional[Path] = None) -> FastAPI:

    base_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "runtime_data"
    hub = VeronicaPlatformHub(base_dir)

    class _FastJSONResponse(JSONResponse):
        def render(self, content: Any) -> bytes:
            return json_bytes(content)

    app = FastAPI(
        title="VERONICA — Inteligência Artificial Nativa para Engenharia de Redes",
        version="1.0.0",
        description=(
            "Plataforma de IA com arquitetura neural própria (Autograd + VeronicaTransformer), "
            "RAG documental rastreável, diagnóstico em 11 etapas, configuração segura em 10 etapas "
            "e Neural Observatory."
        ),
        default_response_class=_FastJSONResponse,
    )
    app.state.hub = hub
    app.state.artifact_store = ArtifactStore(base_dir / "netcanvas_artifacts.sqlite3")
    app.state.agent_orchestrators = {}

    @app.on_event("startup")
    async def initialize_api_stores() -> None:
        await app.state.artifact_store.initialize()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _handle_domain_errors(fn_call: Any) -> Any:
        try:
            return fn_call()
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # ==================== 1. Conversas e Chat ====================

    @app.get("/api/conversations")
    def list_conversations(x_user_id: str = Header(default="usr-eng-01")) -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: {
                "user_id": x_user_id,
                "conversations": hub.conversation_service.list_conversations(x_user_id),
            }
        )

    @app.post("/api/conversations")
    def create_conversation(
        req: CreateConversationRequest,
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.conversation_service.create_conversation(
                user_id=x_user_id,
                title=req.title,
                model_version=hub.trainer.model_version,
            )
        )

    @app.patch("/api/conversations/{conversation_id}")
    def rename_conversation(
        conversation_id: str,
        req: RenameConversationRequest,
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.conversation_service.rename_conversation(
                conversation_id=conversation_id,
                user_id=x_user_id,
                new_title=req.title,
            )
        )

    @app.get("/api/conversations/{conversation_id}")
    def get_conversation_history(
        conversation_id: str,
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.conversation_service.get_conversation(conversation_id, x_user_id)
        )

    @app.delete("/api/conversations/{conversation_id}")
    def delete_conversation_endpoint(
        conversation_id: str,
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> Dict[str, Any]:
        """Exclui permanentemente uma conversa e seu histórico (estilo Claude/ChatGPT)."""
        def _delete() -> Dict[str, Any]:
            result = hub.conversation_service.delete_conversation(conversation_id, x_user_id)
            memory = hub.development_engine.memory
            forgotten = memory.forget_conversation(conversation_id, x_user_id)
            memory.working_memory = [
                item for item in memory.working_memory
                if not (item.metadata.get("user_id") == x_user_id and item.metadata.get("conversation_id") == conversation_id)
            ]
            memory.auto_save_binary()
            result["episodic_memories_forgotten"] = forgotten
            return result
        return _handle_domain_errors(_delete)

    @app.post("/api/conversations/{conversation_id}/messages")
    def send_message(
        conversation_id: str,
        req: SendMessageRequest,
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.chat_service.handle_message(
                conversation_id=conversation_id,
                user_id=x_user_id,
                message_text=req.content,
                modality=req.modality,
                target_devices=req.target_devices,
                mode=req.mode or "chat",
                image_data=req.image_data,
            )
        )

    @app.post("/api/conversations/{conversation_id}/cancel")
    def cancel_generation(
        conversation_id: str,
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.chat_service.cancel_generation(conversation_id)
        )

    @app.post("/api/conversations/{conversation_id}/stream")
    async def stream_conversation_message(
        conversation_id: str,
        req: SendMessageRequest,
        request: Request,
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> StreamingResponse:
        _handle_domain_errors(
            lambda: hub.conversation_service.get_conversation(conversation_id, x_user_id)
        )

        async def _token_stream():
            try:
                for event in hub.chat_service.stream_message(
                    conversation_id=conversation_id,
                    user_id=x_user_id,
                    message_text=req.content,
                    modality=req.modality,
                    target_devices=req.target_devices,
                    mode=req.mode or "chat",
                    model_profile=req.model_profile,
                    response_style=req.response_style,
                ):
                    if await request.is_disconnected():
                        hub.chat_service.cancel_generation(conversation_id)
                        break
                    yield sse_frame(event, event.get("type"))
                    await asyncio.sleep(0.005)
            except Exception:
                logging.getLogger(__name__).exception("Chat stream failed for conversation %s", conversation_id)
                if not await request.is_disconnected():
                    yield sse_frame(
                        {"type": "error", "message": "A Veronica encontrou um erro ao gerar a resposta. Tente novamente."},
                        "error",
                    )
            finally:
                if await request.is_disconnected():
                    hub.chat_service.cancel_generation(conversation_id)

        return StreamingResponse(_token_stream(), media_type="text/event-stream")

    @app.post("/api/conversations/{conversation_id}/agent-stream")
    async def stream_agent_events(
        conversation_id: str,
        req: SendMessageRequest,
        request: Request,
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> StreamingResponse:
        _handle_domain_errors(lambda: hub.conversation_service.get_conversation(conversation_id, x_user_id))

        async def _events():
            # When an orchestrator is attached, pass its typed ReAct events directly
            # through SSE. Existing ChatService generation remains the compatibility
            # fallback and is adapted to the same discriminated event envelope.
            try:
                orchestrator = getattr(hub, "agent_orchestrator", None)
                if orchestrator is None and hasattr(hub, "build_agent_orchestrator"):
                    orchestrator = app.state.agent_orchestrators.get(x_user_id)
                    if orchestrator is None:
                        orchestrator = hub.build_agent_orchestrator(x_user_id)
                        app.state.agent_orchestrators[x_user_id] = orchestrator
                if orchestrator is not None:
                    # Restore persisted conversation context when an Agent session is new.
                    if not orchestrator.sessions.get(conversation_id):
                        from knowledge.ingestion.sanitizer import sanitize_untrusted_text
                        stored = hub.conversation_service.get_conversation(conversation_id, x_user_id)
                        restored = [
                            {
                                "role": message.get("role"),
                                "content": sanitize_untrusted_text(
                                    str(message.get("content", "")), source="stored_conversation_context"
                                ).text,
                            }
                            for message in stored.get("messages", [])
                            if message.get("role") in {"user", "assistant"} and message.get("content")
                        ]
                        orchestrator.sessions[conversation_id] = restored[-12:]
                    hub.conversation_service.append_message(
                        conversation_id=conversation_id, user_id=x_user_id,
                        role="user", content=req.content, modality=req.modality,
                        structured_payload={"model_profile": req.model_profile,
                                            "response_style": req.response_style},
                    )
                    answer_chunks = []
                    async for typed_event in orchestrator.run(
                        req.content, session_id=conversation_id,
                        context={"user_id": x_user_id, "modality": req.modality,
                                 "target_devices": req.target_devices,
                                 "model_profile": req.model_profile,
                                 "response_style": req.response_style},
                    ):
                        if await request.is_disconnected():
                            hub.chat_service.cancel_generation(conversation_id)
                            break
                        payload = typed_event.model_dump(mode="json")
                        if payload.get("type") == "message_delta":
                            answer_chunks.append(str(payload.get("delta", "")))
                        if payload.get("type") == "artifact_open":
                            artifact = payload.get("artifact", {})
                            await app.state.artifact_store.save(
                                x_user_id, artifact,
                                __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                            )
                        yield sse_frame(payload, payload.get("type"))
                    if answer_chunks:
                        hub.conversation_service.append_message(
                            conversation_id=conversation_id, user_id=x_user_id,
                            role="assistant", content="".join(answer_chunks),
                            modality="TEXT", structured_payload={"agent_stream": True,
                                "model_profile": req.model_profile, "response_style": req.response_style},
                        )
                    return
                for item in hub.chat_service.stream_message(
                    conversation_id=conversation_id, user_id=x_user_id,
                    message_text=req.content, modality=req.modality,
                    target_devices=req.target_devices, mode=req.mode or "agent",
                    model_profile=req.model_profile, response_style=req.response_style,
                ):
                    if await request.is_disconnected():
                        hub.chat_service.cancel_generation(conversation_id)
                        break
                    event = MessageDelta(delta=item.get("token_text", "")) if item.get("type") == "token" else item
                    if hasattr(event, "model_dump"):
                        payload = event.model_dump(mode="json")
                    elif isinstance(event, dict) and event.get("type") == "complete":
                        payload = {"type": "message_delta", "delta": event.get("content", ""),
                                   "message_id": event.get("message_id"), "done": True,
                                   "structured_payload": event.get("structured_payload")}
                    else:
                        payload = event
                    yield sse_frame(payload, payload.get("type") if isinstance(payload, dict) else None)
                    await asyncio.sleep(0)
            except Exception:
                logging.getLogger(__name__).exception("Agent stream failed for conversation %s", conversation_id)
                if not await request.is_disconnected():
                    yield sse_frame(
                        {"type": "error", "message": "O modo Agent encontrou um erro ao gerar a resposta. Tente novamente."},
                        "error",
                    )
            finally:
                if await request.is_disconnected():
                    hub.chat_service.cancel_generation(conversation_id)

        return StreamingResponse(_events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/workspace/artifacts")
    async def list_workspace_artifacts(x_user_id: str = Header(default="usr-eng-01")) -> Dict[str, Any]:
        return {"artifacts": await app.state.artifact_store.list(x_user_id)}

    @app.put("/api/workspace/artifacts")
    async def save_workspace_artifact(req: WorkspaceArtifactRequest,
                                     x_user_id: str = Header(default="usr-eng-01")) -> Dict[str, Any]:
        artifact = req.artifact.model_dump(mode="json")
        created = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        await app.state.artifact_store.save(x_user_id, artifact, created)
        return {"artifact": artifact, "created_at": created}

    @app.get("/api/workspace/artifacts/{artifact_id}/versions")
    async def get_workspace_artifact_versions(artifact_id: str,
                                              x_user_id: str = Header(default="usr-eng-01")) -> Dict[str, Any]:
        return {"artifact_id": artifact_id, "versions": await app.state.artifact_store.versions(x_user_id, artifact_id)}

    @app.patch("/api/workspace/artifacts/{artifact_id}/versions/{version}/proposed-content")
    async def edit_workspace_artifact(artifact_id: str, version: int, req: WorkspaceEditRequest,
                                      x_user_id: str = Header(default="usr-eng-01")) -> Dict[str, Any]:
        versions = await app.state.artifact_store.versions(x_user_id, artifact_id)
        current = next((row["artifact"] for row in versions if row["artifact"]["version"] == version), None)
        if current is None:
            raise HTTPException(status_code=404, detail="Artifact version not found.")
        lines = (current.get("proposed_content") or "").splitlines()
        if req.start_line > len(lines) + 1 or req.end_line < req.start_line or req.end_line > max(len(lines), req.start_line):
            raise HTTPException(status_code=400, detail="Selected line range is outside the artifact content.")
        lines[req.start_line - 1:req.end_line] = req.replacement.splitlines()
        current["proposed_content"] = "\n".join(lines)
        current["version"] = version + 1
        hash_payload = {key: value for key, value in current.items() if key != "sha256_hash"}
        current["sha256_hash"] = __import__("hashlib").sha256(
            json.dumps(hash_payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        created = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        await app.state.artifact_store.save(x_user_id, current, created)
        return {"artifact": current, "created_at": created, "recalculation": "queued_for_assisted_recalculation"}

    @app.post("/api/conversations/{conversation_id}/feedback")
    def submit_conversation_feedback(
        conversation_id: str,
        req: ConversationFeedbackRequest,
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> Dict[str, Any]:
        def _exec() -> Dict[str, Any]:
            hub.conversation_service.get_conversation(conversation_id, x_user_id)
            record = hub.continuous_learning.submit_human_correction(
                submitted_by=x_user_id,
                prompt=req.original_prompt,
                incorrect_output=req.incorrect_response,
                corrected_output=req.corrected_response,
                vendor=req.vendor,
                os_name=req.os_name,
                os_version=req.os_version,
                domain=req.domain,
                evidence_reference=f"{req.evidence_reference} (conv={conversation_id}, msg={req.message_id})",
            )
            # Correcoes entram como memoria factual global somente apos aprovacao humana.
            return {
                **record,
                "conversation_id": conversation_id,
                "message_id": req.message_id,
                "indexed_in_rag_memory": False,
                "requires_human_approval": record.get("status") == "PENDING_VERIFICATION",
            }

        return _handle_domain_errors(_exec)

    @app.post("/api/learning/auto-research")
    def auto_research_knowledge(
        body: Dict[str, Any] = Body(...),
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> Dict[str, Any]:
        query = str(body.get("query", "")).strip()
        vendor = body.get("vendor")
        if not query:
            raise ValueError("EMPTY_QUERY: Informe o tópico ou modelo para pesquisa oficial.")
        return _handle_domain_errors(lambda: hub.auto_research_and_learn(query, vendor=vendor))

    @app.post("/api/learning/corrections/{correction_id}/approve")
    def approve_learning_correction(
        correction_id: str,
        req: Optional[ApproveCorrectionRequest] = Body(default=None),
        x_user_id: str = Header(default="usr-app-01"),
    ) -> Dict[str, Any]:
        ds_ver = req.dataset_version if req else "v1.1.0-curated"
        notes = req.reviewer_notes if req else "Human-reviewed evidence accepted."

        def _approve() -> Dict[str, Any]:
            record = hub.continuous_learning.verify_and_approve_correction(
                correction_id=correction_id,
                reviewer_id=x_user_id,
                dataset_version=ds_ver,
                reviewer_notes=notes,
            )
            hub.index_correction_into_rag(record)
            return {**record, "indexed_in_rag_memory": True}

        return _handle_domain_errors(_approve)

    @app.get("/api/learning/status")
    def get_learning_status() -> Dict[str, Any]:
        all_corrections = [asdict(c) for c in hub.continuous_learning.corrections.values()]
        pending_corrections = [c for c in all_corrections if c["status"] == "PENDING_VERIFICATION"]
        approved_corrections = [c for c in all_corrections if c["status"] == "APPROVED_FOR_TRAINING"]
        rejected_corrections = [c for c in all_corrections if c["status"] == "REJECTED"]
        incidents = [asdict(inc) for inc in hub.continuous_learning.validated_incidents.values()]
        return {
            "validated_incidents": incidents,
            "corrections": all_corrections,
            "pending_corrections": pending_corrections,
            "approved_corrections": approved_corrections,
            "rejected_corrections": rejected_corrections,
            "versioned_dataset_size": len(hub.continuous_learning.versioned_training_dataset),
            "versioned_training_dataset": hub.continuous_learning.versioned_training_dataset,
            "auto_update_production_weights": hub.continuous_learning.auto_update_production_weights,
            "allow_external_policy_modification": hub.continuous_learning.allow_external_policy_modification,
        }

    # ==================== 2. Áudio (ASR / VAD / TTS & Consentimento) ====================

    @app.post("/api/audio/consent")
    def set_audio_consent(
        req: VoiceConsentRequest,
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.conversation_service.set_audio_consent(
                conversation_id=req.conversation_id,
                user_id=x_user_id,
                consent=req.consent,
            )
        )

    @app.post("/api/audio/transcribe")
    def transcribe_voice(
        req: VoiceTranscribeRequest,
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> Dict[str, Any]:
        def _exec() -> Dict[str, Any]:
            conv = hub.conversation_service.get_conversation(req.conversation_id, x_user_id)
            asr_res = hub.speech_recognition_service.process_voice_input(
                raw_transcript=req.raw_transcript,
                duration_sec=req.duration_sec,
                audio_size_bytes=req.audio_size_bytes,
                consent_granted=bool(conv.get("audio_consent", False)),
                rms_energy_db=req.rms_energy_db,
                audio_data_base64=req.audio_data_base64,
                audio_format=req.audio_format,
            )
            chat_reply = None
            tts_reply = None
            if req.auto_send_to_chat and asr_res["normalized_transcript"]:
                chat_reply = hub.chat_service.handle_message(
                    conversation_id=req.conversation_id,
                    user_id=x_user_id,
                    message_text=asr_res["normalized_transcript"],
                    modality="VOICE",
                )
                spoken_text = (
                    chat_reply.get("structured_payload", {}).get("spoken_summary")
                    or chat_reply["content"]
                )
                tts_reply = hub.speech_synthesis_service.synthesize_segments(
                    conversation_id=req.conversation_id,
                    text=spoken_text,
                )
            return {
                "asr": asr_res,
                "assistant_reply": chat_reply,
                "tts": tts_reply,
            }

        return _handle_domain_errors(_exec)

    @app.post("/api/audio/synthesize")
    def synthesize_voice(req: VoiceSynthesizeRequest) -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.speech_synthesis_service.synthesize_segments(
                conversation_id=req.conversation_id,
                text=req.text,
            )
        )

    @app.post("/api/audio/interrupt/{conversation_id}")
    def interrupt_voice(conversation_id: str) -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.speech_synthesis_service.interrupt(conversation_id)
        )

    @app.delete("/api/audio/recordings/{conversation_id}")
    def delete_audio_recordings(
        conversation_id: str,
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> Dict[str, Any]:
        def _exec() -> Dict[str, Any]:
            res = hub.conversation_service.delete_voice_recordings(
                conversation_id=conversation_id,
                user_id=x_user_id,
            )
            hub.speech_synthesis_service.active_streams.pop(conversation_id, None)
            hub.audit_log.record(
                actor=x_user_id,
                action="VOICE_RECORDINGS_DELETED",
                target=conversation_id,
                status="PURGED",
                details={"deleted_voice_messages_count": res["deleted_voice_messages_count"]},
            )
            return res

        return _handle_domain_errors(_exec)

    # ==================== 3. Conhecimento Documental (RAG & Ingestão) ====================

    @app.get("/api/knowledge/search")
    def search_knowledge(
        q: str,
        vendor: Optional[str] = None,
        os_name: Optional[str] = None,
        include_obsolete: bool = True,
    ) -> Dict[str, Any]:
        return hub.rag.search(
            query=q,
            vendor_filter=vendor,
            os_filter=os_name,
            include_obsolete=include_obsolete,
        )

    @app.get("/api/knowledge/documents")
    def list_documents() -> Dict[str, Any]:
        docs = [d.to_dict() for d in hub.rag.documents.values()]
        return {"total": len(docs), "documents": docs}

    @app.post("/api/knowledge/upload")
    def upload_document(
        req: DocumentIngestRequest,
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> Dict[str, Any]:
        def _exec() -> Dict[str, Any]:
            hub.rbac.require_permission(x_user_id, "knowledge:ingest")
            res = hub.ingestor.ingest_document(
                title=req.title,
                raw_content=req.content,
                file_format=req.file_format,
                vendor=req.vendor,
                family_model=req.family_model,
                os_name=req.os_name,
                version_build=req.version_build,
                feature=req.feature,
                original_source=req.original_source,
                source_uri=req.source_uri,
                page_section=req.page_section,
                published_date=req.published_date,
                reliability_score=req.reliability_score,
                applicability_restrictions=req.applicability_restrictions,
                is_official=req.is_official,
            )
            doc = res["document"]
            hub.rag.add_document(doc)
            hub.audit_log.record(
                actor=x_user_id,
                action="KNOWLEDGE_DOCUMENT_INGESTED",
                target=doc.doc_id,
                status="QUARANTINED" if res["security_quarantined"] else "INDEXED",
                details={
                    "title": doc.title,
                    "vendor": doc.vendor,
                    "secrets_redacted": res["secrets_redacted"],
                    "quarantine_reasons": res["quarantine_reasons"],
                    "neural_weights_modified": False,
                },
            )
            return {
                "document": doc.to_dict(),
                "secrets_redacted": res["secrets_redacted"],
                "security_quarantined": res["security_quarantined"],
                "quarantine_reasons": res["quarantine_reasons"],
                "neural_weights_modified": False,
            }

        return _handle_domain_errors(_exec)

    # ==================== 4. Neural Observatory & Experimentos ====================

    @app.get("/api/neural/architecture")
    def get_neural_architecture() -> Dict[str, Any]:
        return hub.architecture_service.get_architecture()

    @app.get("/api/neural/state")
    def get_training_state() -> Dict[str, Any]:
        return hub.metrics_service.get_live_metrics()

    @app.get("/api/neural/metrics")
    def get_neural_metrics() -> Dict[str, Any]:
        return hub.metrics_service.get_live_metrics()

    @app.get("/api/neural/parameters")
    def get_neural_parameters() -> Dict[str, Any]:
        return hub.metrics_service.get_parameter_inspection()

    @app.get("/api/neural/telemetry")
    def get_hardware_telemetry_endpoint() -> Dict[str, Any]:
        return hub.metrics_service.get_hardware_telemetry()

    @app.websocket("/api/neural/telemetry/ws")
    async def neural_telemetry_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        interval = 1.0
        try:
            while True:
                metrics = hub.metrics_service.get_live_metrics()
                metrics["hardware"] = hub.metrics_service.get_hardware_telemetry()
                await websocket.send_bytes(json_bytes(metrics))
                try:
                    # A short receive timeout lets clients request a sampling period
                    # without making disconnect detection wait for the next sample.
                    message = await asyncio.wait_for(websocket.receive_json(), timeout=interval)
                    if isinstance(message, dict) and "interval_seconds" in message:
                        interval = max(0.1, min(30.0, float(message["interval_seconds"])))
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            return
        except (ValueError, RuntimeError):
            return

    @app.get("/api/experiments")
    def list_experiments() -> Dict[str, Any]:
        return {
            "active_state": hub.trainer.state,
            "current_experiment_id": hub.trainer.current_experiment_id,
            "experiments": hub.training_service.experiments,
            "history_steps": len(hub.trainer.history),
        }

    @app.get("/api/experiments/compare")
    def compare_experiments() -> Dict[str, Any]:
        return hub.training_service.compare_experiments()

    @app.post("/api/experiments/start")
    def start_experiment(
        req: StartExperimentRequest,
        x_user_id: str = Header(default="usr-trn-01"),
    ) -> Dict[str, Any]:
        hparams = TrainingHyperparameters(
            epochs=req.epochs,
            batch_size=req.batch_size,
            seq_len=req.seq_len,
            learning_rate=req.learning_rate,
            min_lr=req.min_lr,
            weight_decay=req.weight_decay,
            optimizer_type=req.optimizer_type,
            max_grad_norm=req.max_grad_norm,
            grad_accum_steps=req.grad_accum_steps,
        )
        return _handle_domain_errors(
            lambda: hub.training_service.start_experiment(
                user_id=x_user_id,
                hparams=hparams,
                run_async=req.run_async,
            )
        )

    @app.post("/api/experiments/control")
    def control_experiment(
        req: ExperimentControlRequest,
        x_user_id: str = Header(default="usr-trn-01"),
    ) -> Dict[str, Any]:
        def _exec() -> Dict[str, Any]:
            hub.rbac.require_permission(x_user_id, "training:control")
            act = req.action.lower()
            if act == "pause":
                hub.trainer.pause()
            elif act == "resume":
                hub.trainer.resume()
            elif act in ("cancel", "stop"):
                hub.trainer.cancel()
            else:
                raise ValueError(f"Ação de treinamento desconhecida: {req.action}")
            return {"action": act, "trainer_state": hub.trainer.state}

        return _handle_domain_errors(_exec)

    @app.get("/api/checkpoints")
    def list_checkpoints() -> Dict[str, Any]:
        return {
            "active_environments": hub.model_registry_service.active_environments,
            "checkpoints": hub.model_registry_service.list_checkpoints(),
        }

    @app.post("/api/checkpoints/promote")
    def promote_checkpoint(
        req: PromoteCheckpointRequest,
        x_user_id: str = Header(default="usr-app-01"),
    ) -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.model_registry_service.promote_checkpoint(
                checkpoint_id=req.checkpoint_id,
                target_env=req.target_environment,
                approver_user_id=x_user_id,
            )
        )

    @app.post("/api/checkpoints/restore")
    def restore_checkpoint(
        req: RestoreCheckpointRequest,
        x_user_id: str = Header(default="usr-trn-01"),
    ) -> Dict[str, Any]:
        def _exec() -> Dict[str, Any]:
            manifest = hub.model_registry_service.restore_checkpoint(
                checkpoint_id=req.checkpoint_id,
                user_id=x_user_id,
            )
            return {
                **manifest,
                "restored": True,
                "checkpoint_id": req.checkpoint_id,
                "manifest": manifest,
            }

        return _handle_domain_errors(_exec)

    @app.get("/api/benchmarks")
    def get_benchmarks() -> Dict[str, Any]:
        if not hub.benchmark_service.benchmark_runs:
            hub.benchmark_service.run_technical_benchmark()
        return {
            "latest": hub.benchmark_service.benchmark_runs[-1],
            "history": hub.benchmark_service.benchmark_runs,
        }

    @app.post("/api/benchmarks/run")
    def run_benchmark(x_user_id: str = Header(default="usr-trn-01")) -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.benchmark_service.run_technical_benchmark()
        )

    @app.get("/api/events/training")
    async def stream_training_events() -> StreamingResponse:
        """
        Canal Server-Sent Events (SSE) publicando eventos de treinamento contendo
        timestamp, experiment_id, model_version, global_step e métricas reais (Seção 16.7).
        """
        async def _event_gen():
            for rec in hub.trainer.history[-10:]:
                payload = json.dumps(rec, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0.01)

        return StreamingResponse(_event_gen(), media_type="text/event-stream")

    # ==================== 5. Operações de Rede, Diagnóstico e Mudanças Seguras ====================

    @app.get("/api/network/inventory")
    def get_network_inventory() -> Dict[str, Any]:
        return hub.network_ops_service.get_inventory_and_matrix()

    @app.get("/api/network/topology")
    def get_network_topology() -> Dict[str, Any]:
        return hub.topology.export_topology()

    @app.post("/api/network/diagnostics")
    def run_network_diagnostics(
        req: DiagnosticRequest,
        x_user_id: str = Header(default="usr-noc-01"),
    ) -> Dict[str, Any]:
        def _exec() -> Dict[str, Any]:
            hub.rbac.require_permission(x_user_id, "diagnostics:run")
            return hub.diagnostic_engine.investigate(
                problem_description=req.problem_description,
                target_devices=req.target_devices,
                incomplete_telemetry_simulation=req.incomplete_telemetry_simulation,
            )

        return _handle_domain_errors(_exec)

    @app.get("/api/network/changes")
    def list_changes() -> Dict[str, Any]:
        return {
            "changes": list(hub.change_controller.change_records.values()),
        }

    @app.post("/api/network/changes")
    def create_change_plan(
        req: CreateChangePlanRequest,
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.change_controller.create_change_plan_stages_1_to_5(
                requester_user_id=x_user_id,
                device_id=req.device_id,
                title=req.title,
                proposed_commands=req.proposed_commands,
                maintenance_window=req.maintenance_window,
                origin_channel=req.origin_channel,
            )
        )

    @app.post("/api/network/changes/{change_id}/approve")
    def approve_change(
        change_id: str,
        req: ApproveChangeRequest,
        x_user_id: str = Header(default="usr-app-01"),
    ) -> Dict[str, Any]:
        if req.approval_channel.upper() != "EXPLICIT_WEB_APPROVAL":
            code = "VOICE_APPROVAL_PROHIBITED" if req.approval_channel.upper() == "VOICE" else "HUMAN_WEB_APPROVAL_REQUIRED"
            raise HTTPException(status_code=403, detail=f"{code}: Stage 6 requires explicit human web approval.")
        return _handle_domain_errors(
            lambda: hub.change_controller.approve_change_stage_6(
                change_id=change_id,
                approver_user_id=x_user_id,
                confirmed_plan_hash=req.confirmed_plan_hash,
                confirmed_maintenance_window=req.confirmed_maintenance_window,
                approval_channel=req.approval_channel,
                signature_ed25519=req.signature_ed25519,
                approval_timestamp=req.approval_timestamp,
            )
        )

    @app.post("/api/network/changes/{change_id}/execute")
    def execute_change(
        change_id: str,
        req: ExecuteChangeRequest,
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.change_controller.execute_change_stages_7_to_10(
                change_id=change_id,
                executor_user_id=x_user_id,
                approval_token=req.approval_token,
            )
        )

    # ==================== 6. Segurança, RBAC e Auditoria ====================

    @app.get("/api/security/users")
    def list_rbac_users() -> Dict[str, Any]:
        return {"users": list(hub.rbac.users.values())}

    @app.get("/api/security/audit")
    def get_audit_log() -> Dict[str, Any]:
        valid, err = hub.audit_log.verify_chain_integrity()
        return {
            "chain_integrity_verified": valid,
            "integrity_error": err,
            "entries": hub.audit_log.list_entries(),
        }

    # ==================== 7. Ontogênese, Neurogênese & Memórias Humanóides ====================

    @app.get("/api/development/status")
    def get_development_status() -> Dict[str, Any]:
        return hub.development_engine.get_full_development_status()

    @app.get("/api/development/neurons")
    def get_development_neurons() -> Dict[str, Any]:
        return hub.development_engine.neurogenesis.circuit.export_observatory_graph()

    @app.get("/api/development/memories")
    def get_development_memories() -> Dict[str, Any]:
        return hub.development_engine.memory.export_full_memory_map()

    @app.get("/api/memory/binary/stats")
    def get_binary_memory_stats() -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.development_engine.memory.get_binary_stats()
        )

    @app.post("/api/development/see")
    def see_and_learn_endpoint(
        req: SeeImageRequest,
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.development_engine.see_and_learn(
                image_input=req.image_data,
                context_hint=req.context_hint,
                user_id=x_user_id,
            )
        )

    @app.post("/api/development/read")
    def read_and_learn_endpoint(
        req: ReadTextRequest,
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.development_engine.read_and_learn(
                text_content=req.content,
                title=req.title,
                user_id=x_user_id,
            )
        )

    @app.post("/api/development/consolidate")
    def consolidate_memories_endpoint(
        x_user_id: str = Header(default="usr-eng-01"),
    ) -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.development_engine.sleep_and_consolidate(user_id=x_user_id)
        )

    @app.get("/api/development/reflection")
    def get_development_reflection() -> Dict[str, Any]:
        return hub.development_engine.introspective_reflection()

    # ==================== Auto-Aprendizado Autônomo & Neurogênese Espontânea ====================

    @app.get("/api/development/autonomous-learning/status")
    def get_autonomous_learning_status() -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.autonomous_learning.get_status()
        )

    @app.post("/api/development/autonomous-learning/start")
    def start_autonomous_learning() -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.autonomous_learning.start()
        )

    @app.post("/api/development/autonomous-learning/stop")
    def stop_autonomous_learning() -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.autonomous_learning.stop()
        )

    @app.post("/api/development/autonomous-learning/cycle")
    def trigger_autonomous_learning_cycle() -> Dict[str, Any]:
        return _handle_domain_errors(
            lambda: hub.autonomous_learning.step_cycle()
        )

    @app.post("/api/development/autonomous-learning/interval")
    @app.post("/api/development/autonomous-learning/config")
    def set_autonomous_learning_interval(req: AutonomousIntervalRequest) -> Dict[str, Any]:
        def _set():
            return hub.autonomous_learning.set_config(
                workers=req.workers,
                interval_seconds=req.interval_seconds,
                interval_ms=req.interval_ms,
            )
        return _handle_domain_errors(_set)

    @app.get("/api/admin/overview")
    def get_admin_overview() -> Dict[str, Any]:
        memory = hub.development_engine.memory
        development = hub.development_engine.get_full_development_status()
        development.pop("working_memory", None)
        development.pop("introspective_narrative", None)
        memory_map = memory.export_full_memory_map()
        memory_summary = {
            "total_semantic_concepts": memory_map["total_semantic_concepts"],
            "total_episodic_memories": memory_map["total_episodic_memories"],
            "total_consolidation_cycles": memory_map["total_consolidation_cycles"],
            "semantic_nodes": memory_map["semantic_nodes"],
        }
        return {
            "model": hub.config.to_dict(),
            "development": development,
            "autonomous_learning": hub.autonomous_learning.get_status(),
            "binary_memory": memory.get_binary_stats(),
            "memory": memory_summary,
            "runtime": {"array_backend": "numpy", "device_writes_enabled": False},
        }

    @app.get("/api/admin/research-quality")
    def get_research_quality() -> Dict[str, Any]:
        return hub.get_open_domain_quality()

    @app.get("/api/admin/memory-graph")
    def get_admin_memory_graph(limit: int = 400) -> Dict[str, Any]:
        circuit = hub.development_engine.neurogenesis.circuit
        neurons = circuit.neurons[-max(1, min(limit, 1000)):]
        nodes = [neuron.export_meta() for neuron in neurons]
        edges: List[Dict[str, Any]] = []
        for index, left in enumerate(neurons):
            left_concepts = set(left.co_activated_concepts)
            if not left_concepts:
                continue
            for right in neurons[index + 1:]:
                shared = left_concepts.intersection(right.co_activated_concepts)
                if shared:
                    edges.append({
                        "source": left.neuron_id,
                        "target": right.neuron_id,
                        "strength": round(min(left.synaptic_strength, right.synaptic_strength), 3),
                        "label": next(iter(shared)),
                    })
                    if len(edges) >= 1600:
                        break
            if len(edges) >= 1600:
                break
        return {"nodes": nodes, "edges": edges, "total_neurons": len(circuit.neurons)}

    @app.get("/api/admin/config")
    def get_admin_runtime_config() -> Dict[str, Any]:
        return {
            "model": hub.config.to_dict(),
            "autonomous_learning": hub.autonomous_learning.get_status(),
            "runtime": {"array_backend": "numpy", "device_writes_enabled": False},
        }

    @app.put("/api/admin/config")
    def update_admin_runtime_config(req: AdminRuntimeConfigRequest) -> Dict[str, Any]:
        if req.workers is None and req.interval_seconds is None:
            raise HTTPException(status_code=400, detail="Informe workers ou interval_seconds.")
        return hub.autonomous_learning.set_config(
            workers=req.workers,
            interval_seconds=req.interval_seconds,
        )

    @app.post("/api/admin/memory/consolidate")
    def consolidate_admin_memory() -> Dict[str, Any]:
        return hub.development_engine.sleep_and_consolidate(user_id="admin")

    # ==================== 8. Servir Interface Gráfica Web (GUI / Vite) ====================

    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    frontend_dist = frontend_dir / "dist"
    frontend_public = frontend_dir / "public"

    if (frontend_dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    if frontend_public.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_public)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_veronica_gui() -> HTMLResponse:
        dist_index = frontend_dist / "index.html"
        if dist_index.exists():
            return HTMLResponse(dist_index.read_text(encoding="utf-8"))
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return HTMLResponse(index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Veronica Platform API is running.</h1>")

    @app.get("/admin", response_class=HTMLResponse)
    def serve_veronica_admin() -> HTMLResponse:
        return serve_veronica_gui()

    return app


app = create_app()
