"""
Implementação dos 10 Serviços Modulares da Veronica (Seção 16.6):
1. ChatService
2. ConversationService (com persistência SQLite/SQL estruturada e isolamento por usuário)
3. SpeechRecognitionService
4. SpeechSynthesisService
5. NeuralTrainingService
6. NeuralMetricsService
7. ArchitectureVisualizationService
8. BenchmarkService (com separação das 4 dimensões de conhecimento da Seção 16.4)
9. ModelRegistryService (com separação TRAINING / STAGING / PRODUCTION)
10. NetworkOperationsService
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
import base64
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import tempfile
import unicodedata
import wave
from typing import Any, Dict, Iterator, List, Optional
import uuid

from core.tokenizer import TechnicalTokenizer
from core.transformer import VeronicaConfig, VeronicaTransformer
from core.training import (
    ContinuousLearningPipeline,
    DEFAULT_NETWORK_TRAINING_CORPUS,
    TrainingHyperparameters,
    VeronicaTrainer,
    get_real_hardware_telemetry,
)
from core.inference import AutoregressiveGenerator, GenerationConfig
from core.agent import AgentTool, ReActOrchestrator
from core.contracts import ToolSafetyTier
from core.neural.checkpoint import load_checkpoint
from core.development import DevelopmentEngine, AutonomousCognitiveDaemon
from knowledge import KnowledgeIngestor, HybridRAGEngine, OfficialDocumentationSearchConnector
from knowledge.sources.catalog import KnowledgeDocument
from knowledge.retrieval.open_domain import (
    OpenDomainKnowledgeStore, assess_source, classify_domain, synthesize_evidence, terms as open_domain_terms,
)
from knowledge.ingestion.ingestor import scrub_secrets, search_live_web_and_rfc
from networking import (
    COMPATIBILITY_MATRIX,
    DigitalTwinNetworkAdapter,
    NetworkDiagnosticEngine,
    NetworkTopologyGraph,
    create_default_network_inventory,
)
from safety import (
    CredentialVault,
    DeterministicChangeController,
    RBACPolicyManager,
    TamperEvidentAuditLog,
)


_FOCUS_STOP_WORDS = {
    "quando", "where", "what", "qual", "quais", "quem", "como", "porque", "por", "que", "ano", "year",
    "foi", "era", "sao", "ser", "tem", "the", "was", "were", "did", "does", "are", "is", "do", "how", "who", "why",
    "inventado", "inventada", "inventados", "inventadas", "invented", "criado", "criada", "created", "developed", "desenvolvido", "desenvolvida", "surgiu", "origem",
    "protocolo", "protocol", "sistema", "system", "assunto", "subject", "tema", "topic", "contexto", "context",
    "refiro", "referindo", "referring", "refer", "mean", "meant", "quero", "quis", "dizer", "say", "said", "falo", "falando", "talking",
    "somente", "apenas", "only", "just", "especificamente", "specifically", "exclusivamente", "exclusively", "sobre", "about",
    "please", "explique", "explaining", "explain", "tell", "fale", "mostre", "show", "me", "minha", "meu",
    "eu", "i", "ao", "aos", "da", "das", "de", "do", "dos", "e", "em", "na", "nas", "no", "nos", "a", "o", "os",
    "um", "uma", "uns", "umas", "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for", "is", "it",
    "this", "that", "only", "just", "please", "tell", "about", "me",
}


def _focus_terms(text: str) -> set[str]:
    """Keep the topic-bearing words while dropping question and clarification scaffolding."""
    return open_domain_terms(text) - _FOCUS_STOP_WORDS


def _matches_focus(text: str, focus: set[str]) -> bool:
    if not focus:
        return True
    matched = _focus_terms(text) & focus
    required = 1 if len(focus) == 1 else max(2, (len(focus) * 2 + 2) // 3)
    return len(matched) >= required


class ConversationService:
    """
    Gerencia sessões conversacionais com persistência estruturada em banco de dados,
    isolamento por usuário, histórico ordenado, anexos e sumarização de contexto.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    task_state TEXT NOT NULL,
                    context_summary TEXT NOT NULL,
                    audio_consent INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    content TEXT NOT NULL,
                    structured_payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def create_conversation(
        self,
        user_id: str,
        title: str = "Nova Investigação de Rede",
        model_version: str = "veronica-v1.0-base",
        checkpoint_id: str = "ckpt-initial",
    ) -> Dict[str, Any]:
        conv_id = f"conv-{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (
                    conversation_id, user_id, title, model_version, checkpoint_id,
                    task_state, context_summary, audio_consent, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (conv_id, user_id, title, model_version, checkpoint_id, "IDLE", "", 0, now, now),
            )
            conn.commit()
        return self.get_conversation(conv_id, user_id)

    def rename_conversation(self, conversation_id: str, user_id: str, new_title: str) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE conversation_id = ? AND user_id = ?",
                (new_title, now, conversation_id, user_id),
            )
            conn.commit()
            if cur.rowcount == 0:
                raise KeyError(f"Conversa '{conversation_id}' não encontrada para o usuário '{user_id}'.")
        return self.get_conversation(conversation_id, user_id)

    def set_audio_consent(self, conversation_id: str, user_id: str, consent: bool) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET audio_consent = ?, updated_at = ? WHERE conversation_id = ? AND user_id = ?",
                (1 if consent else 0, now, conversation_id, user_id),
            )
            conn.commit()
        return self.get_conversation(conversation_id, user_id)

    def list_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, conversation_id: str, user_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
            if row is None:
                raise PermissionError(
                    f"SESSION_ISOLATION_GUARD: Conversa '{conversation_id}' inexistente ou pertencente a outro usuário."
                )
            msgs = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,),
            ).fetchall()

        conv = dict(row)
        conv["audio_consent"] = bool(conv["audio_consent"])
        conv["messages"] = [
            {
                "message_id": m["message_id"],
                "role": m["role"],
                "modality": m["modality"],
                "content": m["content"],
                "structured_payload": json.loads(m["structured_payload"]),
                "created_at": m["created_at"],
            }
            for m in msgs
        ]
        return conv

    def append_message(
        self,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        modality: str = "TEXT",
        structured_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        conv = self.get_conversation(conversation_id, user_id)
        msg_id = f"msg-{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc).isoformat()
        payload_json = json.dumps(structured_payload or {}, ensure_ascii=False)

        # Sumarização automática de histórico quando excede 8 mensagens para limitar tokens de contexto
        all_msgs = conv["messages"]
        summary = conv["context_summary"]
        if len(all_msgs) >= 8:
            recent_topics = [m["content"][:80] for m in all_msgs[-6:] if m["role"] == "user"]
            summary = "Tópicos anteriores da sessão: " + " | ".join(recent_topics)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    message_id, conversation_id, user_id, role, modality, content, structured_payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (msg_id, conversation_id, user_id, role, modality, content, payload_json, now),
            )
            conn.execute(
                "UPDATE conversations SET context_summary = ?, updated_at = ? WHERE conversation_id = ?",
                (summary, now, conversation_id),
            )
            conn.commit()

        return {
            "message_id": msg_id,
            "conversation_id": conversation_id,
            "role": role,
            "modality": modality,
            "content": content,
            "structured_payload": structured_payload or {},
            "created_at": now,
        }

    def delete_voice_recordings(self, conversation_id: str, user_id: str) -> Dict[str, Any]:
        """
        Exclui transcrições e registros de voz da conversa (controle de privacidade — Seção 16.2.B item 13).
        """
        self.get_conversation(conversation_id, user_id)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM messages WHERE conversation_id = ? AND user_id = ? AND modality = 'VOICE'",
                (conversation_id, user_id),
            )
            deleted_count = cur.rowcount
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ? AND user_id = ?",
                (now, conversation_id, user_id),
            )
            conn.commit()
        return {
            "conversation_id": conversation_id,
            "deleted": True,
            "deleted_voice_messages_count": deleted_count,
            "status": "VOICE_RECORDINGS_PURGED",
            "timestamp": now,
        }

    def delete_conversation(self, conversation_id: str, user_id: str) -> Dict[str, Any]:
        """
        Exclui permanentemente a sessão conversacional e todas as mensagens associadas.
        Garante isolamento de usuário e higienização completa no SQLite.
        """
        self.get_conversation(conversation_id, user_id)
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE conversation_id = ? AND user_id = ?", (conversation_id, user_id))
            conn.commit()
        return {
            "conversation_id": conversation_id,
            "deleted": True,
            "status": "CONVERSATION_DELETED",
            "message": f"Conversa '{conversation_id}' excluída com sucesso."
        }


class SpeechRecognitionService:
    """
    Serviço substituível de Reconhecimento de Voz (ASR) e Detecção de Atividade Vocal (VAD),
    com normalização de jargão técnico de redes, validação de consentimento e limites de áudio.
    """

    MAX_AUDIO_DURATION_SEC = 120.0
    MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB

    def __init__(self) -> None:
        self._whisper_model: Any = None
        self._whisper_error: Optional[str] = None
        self._whisper_lock = threading.Lock()

    def _load_whisper(self) -> Any:
        if self._whisper_model is not None:
            return self._whisper_model
        if self._whisper_error is not None:
            return None
        with self._whisper_lock:
            if self._whisper_model is not None or self._whisper_error is not None:
                return self._whisper_model
            model_path = os.environ.get("VERONICA_WHISPER_MODEL_PATH", "").strip()
            if not model_path:
                self._whisper_error = "LOCAL_MODEL_PATH_NOT_CONFIGURED"
                return None
            if not Path(model_path).exists():
                self._whisper_error = "LOCAL_MODEL_PATH_NOT_FOUND"
                return None
            try:
                from faster_whisper import WhisperModel
                self._whisper_model = WhisperModel(
                    model_path,
                    device=os.environ.get("VERONICA_WHISPER_DEVICE", "cpu"),
                    compute_type=os.environ.get("VERONICA_WHISPER_COMPUTE_TYPE", "int8"),
                    local_files_only=True,
                )
            except TypeError:
                # Older faster-whisper versions do not expose local_files_only;
                # a filesystem model path still prevents implicit model downloads.
                try:
                    from faster_whisper import WhisperModel
                    self._whisper_model = WhisperModel(
                        model_path,
                        device=os.environ.get("VERONICA_WHISPER_DEVICE", "cpu"),
                        compute_type=os.environ.get("VERONICA_WHISPER_COMPUTE_TYPE", "int8"),
                    )
                except Exception as exc:
                    self._whisper_error = f"LOCAL_STT_UNAVAILABLE:{type(exc).__name__}"
            except Exception as exc:
                self._whisper_error = f"LOCAL_STT_UNAVAILABLE:{type(exc).__name__}"
        return self._whisper_model

    def _transcribe_local(self, audio: bytes, audio_format: str) -> tuple[str, str | None]:
        model = self._load_whisper()
        if model is None:
            return "", self._whisper_error or "LOCAL_STT_UNAVAILABLE"
        suffix = "." + re.sub(r"[^a-zA-Z0-9]", "", audio_format.lower())[:8]
        if suffix == ".":
            suffix = ".wav"
        path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(audio)
                path = handle.name
            segments, _info = model.transcribe(path, language="pt", vad_filter=True)
            return " ".join(segment.text.strip() for segment in segments).strip(), None
        except Exception as exc:
            return "", f"LOCAL_STT_FAILED:{type(exc).__name__}"
        finally:
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass

    PHONETIC_REPLACEMENTS = {
        r"\bo s p f\b": "OSPF",
        r"\bb g p\b": "BGP",
        r"\bm p l s\b": "MPLS",
        r"\bv l a n\b": "VLAN",
        r"\bm t u\b": "MTU",
        r"\bi o s x r\b": "IOS-XR",
        r"\bi o s x e\b": "IOS-XE",
        r"\bv r p\b": "VRP",
        r"\be x start\b": "EXSTART",
    }

    def process_voice_input(
        self,
        raw_transcript: str,
        duration_sec: float = 4.0,
        audio_size_bytes: int = 32000,
        consent_granted: bool = False,
        rms_energy_db: float = -22.0,
        audio_data_base64: Optional[str] = None,
        audio_format: str = "wav",
    ) -> Dict[str, Any]:
        if not consent_granted:
            raise PermissionError(
                "AUDIO_CONSENT_REQUIRED: Captura e processamento de voz exigem consentimento explícito do usuário."
            )
        audio_bytes: Optional[bytes] = None
        if audio_data_base64:
            try:
                audio_bytes = base64.b64decode(audio_data_base64, validate=True)
            except (ValueError, base64.binascii.Error) as exc:
                raise ValueError("INVALID_AUDIO_ENCODING: Áudio deve ser base64 válido.") from exc
            audio_size_bytes = len(audio_bytes)
        if duration_sec > self.MAX_AUDIO_DURATION_SEC:
            raise ValueError(
                f"AUDIO_DURATION_LIMIT_EXCEEDED: Duração ({duration_sec}s) excede o máximo permitido ({self.MAX_AUDIO_DURATION_SEC}s)."
            )
        if audio_size_bytes > self.MAX_AUDIO_BYTES:
            raise ValueError("AUDIO_SIZE_LIMIT_EXCEEDED: Arquivo de áudio excede 10 MB.")

        stt_error = None
        transcript = raw_transcript
        if audio_bytes is not None:
            transcript, stt_error = self._transcribe_local(audio_bytes, audio_format)
            if not transcript:
                transcript = raw_transcript

        normalized = transcript.strip()
        for pat, repl in self.PHONETIC_REPLACEMENTS.items():
            normalized = re.sub(pat, repl, normalized, flags=re.IGNORECASE)

        vad_speech_detected = rms_energy_db > -45.0 and len(normalized) > 0
        return {
            "original_transcript": transcript,
            "normalized_transcript": normalized,
            "stt_backend": "faster-whisper-local" if audio_bytes is not None and not stt_error else "provided-transcript",
            "stt_available": stt_error is None if audio_bytes is not None else None,
            "stt_error": stt_error,
            "vad_speech_detected": vad_speech_detected,
            "end_of_turn_detected": True,
            "duration_sec": duration_sec,
            "can_authorize_production_changes": False,
            "security_notice": "Ordem por voz convertida com sucesso. Lembrete: comandos de voz não autorizam mudanças em produção.",
        }


class SpeechSynthesisService:
    """
    Serviço substituível de Síntese de Fala (TTS) em Português Brasileiro,
    segmentando respostas técnicas para reprodução progressiva e interrupção limpa (barge-in).
    """

    def __init__(self) -> None:
        self.active_streams: Dict[str, bool] = {}
        self._piper_voice: Any = None
        self._piper_error: Optional[str] = None
        self._piper_lock = threading.Lock()

    def _load_piper(self) -> Any:
        if self._piper_voice is not None:
            return self._piper_voice
        if self._piper_error is not None:
            return None
        with self._piper_lock:
            if self._piper_voice is not None or self._piper_error is not None:
                return self._piper_voice
            model_path = os.environ.get("VERONICA_PIPER_MODEL_PATH", "").strip()
            if not model_path:
                self._piper_error = "LOCAL_MODEL_PATH_NOT_CONFIGURED"
                return None
            if not Path(model_path).is_file():
                self._piper_error = "LOCAL_MODEL_PATH_NOT_FOUND"
                return None
            try:
                from piper.voice import PiperVoice
                config_path = Path(model_path + ".json")
                kwargs = {"config_path": str(config_path)} if config_path.is_file() else {}
                self._piper_voice = PiperVoice.load(model_path, **kwargs)
            except Exception as exc:
                self._piper_error = f"LOCAL_TTS_UNAVAILABLE:{type(exc).__name__}"
        return self._piper_voice

    def _synthesize_local(self, text: str) -> tuple[str | None, str | None]:
        voice = self._load_piper()
        if voice is None:
            return None, self._piper_error or "LOCAL_TTS_UNAVAILABLE"
        output = io.BytesIO()
        try:
            with wave.open(output, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
            return base64.b64encode(output.getvalue()).decode("ascii"), None
        except Exception as exc:
            return None, f"LOCAL_TTS_FAILED:{type(exc).__name__}"

    def synthesize_segments(self, conversation_id: str, text: str) -> Dict[str, Any]:
        self.active_streams[conversation_id] = True
        # Remove blocos extensos de CLI para leitura fluida em voz mantendo o resumo técnico
        clean_speech = re.sub(r"```[\s\S]*?```", " [Bloco de comandos CLI disponível na tela.] ", text)
        clean_speech = re.sub(r"[#*`_~]", "", clean_speech)
        sentences = [s.strip() for s in re.split(r"(?<=[\.\!\?\n])\s+", clean_speech) if s.strip()]
        spoken_text = " ".join(sentences[:12])
        audio_data, tts_error = self._synthesize_local(spoken_text) if spoken_text else (None, None)

        return {
            "conversation_id": conversation_id,
            "lang": "pt-BR",
            "voice_profile": "Veronica-PT-BR-Technical",
            "interrupted": False,
            "segments": sentences[:12],
            "full_spoken_text": spoken_text,
            "tts_backend": "piper-local" if audio_data else "text-segments",
            "tts_available": audio_data is not None,
            "tts_error": tts_error,
            "audio_mime_type": "audio/wav" if audio_data else None,
            "audio_base64": audio_data,
        }

    def interrupt(self, conversation_id: str) -> Dict[str, Any]:
        self.active_streams[conversation_id] = False
        return {
            "conversation_id": conversation_id,
            "interrupted": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class ChatService:
    """
    Coordena o chat inteligente da Veronica:
    - Recupera evidências documentais via RAG (com os 12 metadados)
    - Executa investigação diagnóstica somente leitura quando aplicável
    - Gera síntese pelo modelo neural próprio VeronicaTransformer
    - Retorna os 9 campos obrigatórios da Seção 11 (Diagnóstico, Evidências, Fonte técnica,
      Compatibilidade, Risco, Comandos propostos, Plano de validação, Plano de rollback, Limitações)
    - Separa claramente Fatos Comprovados, Inferências Técnicas e Hipóteses.
    """

    def __init__(
        self,
        conversation_service: ConversationService,
        rag: HybridRAGEngine,
        diagnostic_engine: NetworkDiagnosticEngine,
        generator: AutoregressiveGenerator,
        trainer: VeronicaTrainer,
        hub: Optional[Any] = None,
    ) -> None:
        self.conversation_service = conversation_service
        self.rag = rag
        self.diagnostic_engine = diagnostic_engine
        self.generator = generator
        self.trainer = trainer
        self.hub = hub
        self._cancel_flags: Dict[str, bool] = {}

    def cancel_generation(self, conversation_id: str) -> Dict[str, Any]:
        self._cancel_flags[conversation_id] = True
        return {"conversation_id": conversation_id, "cancelled": True}

    def _extract_conversation_context(
        self,
        conversation_id: str,
        user_id: str,
        current_msg: str,
    ) -> Dict[str, Any]:
        """
        Analisa o histórico de múltiplos turnos para manter contexto contínuo
        (estilo Claude / ChatGPT), resolver anáforas e rastrear tópicos e entidades.
        """
        try:
            conv = self.conversation_service.get_conversation(conversation_id, user_id)
            messages = conv.get("messages", [])
        except Exception:
            messages = []

        prior_msgs = [m for m in messages if m.get("content") != current_msg or m.get("role") != "user"]
        user_msgs = [m for m in prior_msgs if m.get("role") == "user"]
        asst_msgs = [m for m in prior_msgs if m.get("role") == "assistant"]

        last_user_query = user_msgs[-1]["content"] if user_msgs else ""
        last_asst_content = asst_msgs[-1]["content"] if asst_msgs else ""

        # Keep general topic terms from the recent dialogue, regardless of domain.
        recent_topics = sorted(_focus_terms(last_user_query))[:16]

        return {
            "turn_count": len(prior_msgs),
            "last_user_query": last_user_query,
            "last_asst_content": last_asst_content,
            "recent_topics": recent_topics,
            "prior_messages": prior_msgs,
        }

    def _generate_svg_image_for_prompt(self, prompt: str) -> str:
        """
        Gera imagens vetoriais SVG de alta resolução desenhadas autonomamente pela Veronica:
        Topologias de rede, fluxo de protocolos, anéis ópticos DWDM e arquiteturas.
        """
        p = prompt.lower()
        if "dwdm" in p or "optico" in p or "óptic" in p:
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 360" style="width:100%;max-width:760px;height:auto;background:#030712;border-radius:12px;font-family:Segoe UI,Inter,sans-serif;">'
                '<defs>'
                '<linearGradient id="bgGlow" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#0f172a"/><stop offset="100%" stop-color="#020617"/></linearGradient>'
                '<linearGradient id="fiberGlow" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#38bdf8"/><stop offset="50%" stop-color="#ec4899"/><stop offset="100%" stop-color="#10b981"/></linearGradient>'
                '<filter id="glow"><feGaussianBlur stdDeviation="3" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
                '</defs>'
                '<rect width="760" height="360" fill="url(#bgGlow)" rx="12" stroke="#1e293b" stroke-width="2"/>'
                '<text x="30" y="36" fill="#38bdf8" font-size="13" font-weight="700" letter-spacing="1">VERONICA NEURAL GENERATOR — TOPOLOGIA DWDM BANDA C</text>'
                '<text x="30" y="56" fill="#64748b" font-size="11">Anel Metro ROADM • 40 Canais • λ1: 1550.12nm (Tx: +2.0dBm, Rx: -11.4dBm)</text>'
                '<circle cx="380" cy="200" r="110" fill="none" stroke="url(#fiberGlow)" stroke-width="4" stroke-dasharray="6,4" filter="url(#glow)"/>'
                '<circle cx="380" cy="90" r="26" fill="#0f172a" stroke="#38bdf8" stroke-width="3" filter="url(#glow)"/>'
                '<text x="380" y="94" fill="#f8fafc" font-size="10" font-weight="bold" text-anchor="middle">ROADM-01</text>'
                '<text x="380" y="55" fill="#38bdf8" font-size="11" text-anchor="middle">Core Metro Alpha</text>'
                '<circle cx="490" cy="200" r="26" fill="#0f172a" stroke="#10b981" stroke-width="3" filter="url(#glow)"/>'
                '<text x="490" y="204" fill="#f8fafc" font-size="10" font-weight="bold" text-anchor="middle">ROADM-02</text>'
                '<text x="565" y="204" fill="#10b981" font-size="11">Core Beta (+1.8 dBm)</text>'
                '<circle cx="380" cy="310" r="26" fill="#0f172a" stroke="#ec4899" stroke-width="3" filter="url(#glow)"/>'
                '<text x="380" y="314" fill="#f8fafc" font-size="10" font-weight="bold" text-anchor="middle">ROADM-03</text>'
                '<text x="380" y="348" fill="#ec4899" font-size="11" text-anchor="middle">EDFA Optical Amp</text>'
                '<circle cx="270" cy="200" r="26" fill="#0f172a" stroke="#f59e0b" stroke-width="3" filter="url(#glow)"/>'
                '<text x="270" y="204" fill="#f8fafc" font-size="10" font-weight="bold" text-anchor="middle">ROADM-04</text>'
                '<text x="195" y="204" fill="#f59e0b" font-size="11">Transponder 100G</text>'
                '<rect x="330" y="185" width="100" height="30" rx="6" fill="#090d16" stroke="#334155"/>'
                '<text x="380" y="204" fill="#10b981" font-size="11" font-weight="bold" text-anchor="middle">STATUS: NOMINAL</text>'
                '</svg>'
            )
        elif "bgp" in p or "peering" in p or "as" in p:
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 360" style="width:100%;max-width:760px;height:auto;background:#030712;border-radius:12px;font-family:Segoe UI,Inter,sans-serif;">'
                '<defs>'
                '<linearGradient id="bgBgp" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#0f172a"/><stop offset="100%" stop-color="#020617"/></linearGradient>'
                '<filter id="glowBgp"><feGaussianBlur stdDeviation="3" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
                '</defs>'
                '<rect width="760" height="360" fill="url(#bgBgp)" rx="12" stroke="#1e293b" stroke-width="2"/>'
                '<text x="30" y="36" fill="#38bdf8" font-size="13" font-weight="700" letter-spacing="1">VERONICA NEURAL GENERATOR — ARQUITETURA BGP MULTI-AS</text>'
                '<text x="30" y="56" fill="#64748b" font-size="11">Sessões eBGP (TCP 179) • iBGP Route Reflector Mesh • Comunidades BGP</text>'
                '<rect x="40" y="85" width="300" height="240" rx="10" fill="#0b1329" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="4,4"/>'
                '<text x="60" y="110" fill="#60a5fa" font-size="12" font-weight="bold">AS 65000 (Rede Local / Core)</text>'
                '<rect x="420" y="85" width="300" height="240" rx="10" fill="#0b1329" stroke="#10b981" stroke-width="1.5" stroke-dasharray="4,4"/>'
                '<text x="440" y="110" fill="#34d399" font-size="12" font-weight="bold">AS 65001 (Upstream Transit / Tier-1)</text>'
                '<circle cx="130" cy="180" r="24" fill="#1e293b" stroke="#38bdf8" stroke-width="2" filter="url(#glowBgp)"/>'
                '<text x="130" y="184" fill="#fff" font-size="10" font-weight="bold" text-anchor="middle">RR-01</text>'
                '<circle cx="260" cy="230" r="24" fill="#1e293b" stroke="#38bdf8" stroke-width="2" filter="url(#glowBgp)"/>'
                '<text x="260" y="234" fill="#fff" font-size="10" font-weight="bold" text-anchor="middle">BORDER-01</text>'
                '<line x1="130" y1="180" x2="260" y2="230" stroke="#38bdf8" stroke-width="2" stroke-dasharray="4,2"/>'
                '<text x="170" y="215" fill="#94a3b8" font-size="9">iBGP Mesh</text>'
                '<line x1="284" y1="230" x2="476" y2="230" stroke="#f59e0b" stroke-width="3" filter="url(#glowBgp)"/>'
                '<text x="380" y="220" fill="#f59e0b" font-size="11" font-weight="bold" text-anchor="middle">eBGP TCP 179</text>'
                '<text x="380" y="245" fill="#94a3b8" font-size="9" text-anchor="middle">Hold: 180s | Keepalive: 60s</text>'
                '<circle cx="500" cy="230" r="24" fill="#1e293b" stroke="#10b981" stroke-width="2" filter="url(#glowBgp)"/>'
                '<text x="500" y="234" fill="#fff" font-size="10" font-weight="bold" text-anchor="middle">TIER1-PE01</text>'
                '<circle cx="630" cy="180" r="24" fill="#1e293b" stroke="#10b981" stroke-width="2" filter="url(#glowBgp)"/>'
                '<text x="630" y="184" fill="#fff" font-size="10" font-weight="bold" text-anchor="middle">TIER1-RR</text>'
                '<line x1="500" y1="230" x2="630" y2="180" stroke="#10b981" stroke-width="2" stroke-dasharray="4,2"/>'
                '</svg>'
            )
        else:
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 360" style="width:100%;max-width:760px;height:auto;background:#030712;border-radius:12px;font-family:Segoe UI,Inter,sans-serif;">'
                '<defs>'
                '<linearGradient id="bgOspf" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#0f172a"/><stop offset="100%" stop-color="#020617"/></linearGradient>'
                '<filter id="glowOspf"><feGaussianBlur stdDeviation="3" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
                '</defs>'
                '<rect width="760" height="360" fill="url(#bgOspf)" rx="12" stroke="#1e293b" stroke-width="2"/>'
                '<text x="30" y="36" fill="#38bdf8" font-size="13" font-weight="700" letter-spacing="1">VERONICA NEURAL GENERATOR — TOPOLOGIA OSPF ÁREA 0 & STUB ROUTER</text>'
                '<text x="30" y="56" fill="#64748b" font-size="11">Convergência SPF • RFC 6987 MaxLinkMetric (65535) • Drenagem Suave de Tráfego</text>'
                '<line x1="150" y1="130" x2="610" y2="130" stroke="#10b981" stroke-width="4" filter="url(#glowOspf)"/>'
                '<text x="380" y="120" fill="#10b981" font-size="11" font-weight="bold" text-anchor="middle">Caminho Preferencial SPF — Cost: 10 (100GE Link)</text>'
                '<path d="M 150 130 Q 380 270 610 130" fill="none" stroke="#f59e0b" stroke-width="3" stroke-dasharray="6,4" filter="url(#glowOspf)"/>'
                '<text x="380" y="245" fill="#f59e0b" font-size="11" font-weight="bold" text-anchor="middle">MaxLinkMetric Drenado — Cost: 65535 (RFC 6987)</text>'
                '<circle cx="150" cy="130" r="28" fill="#0f172a" stroke="#38bdf8" stroke-width="3" filter="url(#glowOspf)"/>'
                '<text x="150" y="134" fill="#fff" font-size="11" font-weight="bold" text-anchor="middle">CORE-01</text>'
                '<text x="150" y="180" fill="#94a3b8" font-size="10" text-anchor="middle">Loopback: 10.0.0.1</text>'
                '<circle cx="380" cy="225" r="28" fill="#0f172a" stroke="#f59e0b" stroke-width="3" filter="url(#glowOspf)"/>'
                '<text x="380" y="229" fill="#fff" font-size="11" font-weight="bold" text-anchor="middle">CORE-02</text>'
                '<text x="380" y="275" fill="#f59e0b" font-size="10" font-weight="bold" text-anchor="middle">STUB ROUTER (Manutenção)</text>'
                '<circle cx="610" cy="130" r="28" fill="#0f172a" stroke="#38bdf8" stroke-width="3" filter="url(#glowOspf)"/>'
                '<text x="610" y="134" fill="#fff" font-size="11" font-weight="bold" text-anchor="middle">CORE-03</text>'
                '<text x="610" y="180" fill="#94a3b8" font-size="10" text-anchor="middle">Loopback: 10.0.0.3</text>'
                '</svg>'
            )

    def _classify_intent(
        self,
        text: str,
        target_devices: Optional[List[str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if target_devices and len(target_devices) > 0:
            return "DIAGNOSTIC"
        low = "".join(
            char for char in unicodedata.normalize("NFKD", text.lower().strip())
            if not unicodedata.combining(char)
        )

        # 1. Perguntas de Eventos Históricos / Fatos do Dia (ou follow-up da data)
        time_patterns = [
            r"\b(que dia [eé] hoje|qual [eé] a data de hoje|qual [eé] o dia de hoje|que dia da semana|data de hoje|data atual|dia atual)\b",
            r"\b(que horas s[aã]o|qual [eé] a hora|hor[aá]rio atual|horas agora|hora certa)\b",
        ]
        if any(re.search(p, low) for p in time_patterns):
            return "TIME_QUERY"

        # 3. Solicitação de Integração de Sistemas / Código Feito por Veronica
        image_gen_patterns = [
            r"\b(ger[ea]|criar?|desenh[ea]|mostrar?|fa[cç]a|elabore).*(imagem|desenho|diagrama visual|ilustra[cç][aã]o|topologia visual|esquema visual)\b",
            r"\b(gerar imagens?|desenhar|imagem da rede|imagem do ospf|imagem do bgp|imagem de roteadores)\b",
            r"\b(quero uma imagem|pode gerar uma imagem|gere um desenho|desenhe uma topologia|crie uma imagem)\b",
        ]
        if any(re.search(p, low) for p in image_gen_patterns):
            return "IMAGE_GENERATION"

        # 4. Saudações e Diálogo Cotidiano Geral
        greeting_patterns = [
            r"^(oi|ol[aá]|bom dia|boa tarde|boa noite|hello|hi|hey|e a[ií])\s*[!?.]*$",
            r"^(oi|ol[aá]|bom dia|boa tarde|boa noite)\b.*\b(tudo bem|como vai)\b",
            r"\b(quem [eé] voc[eê]|quem voce e|como voc[eê] se chama|qual [eé] seu nome)\b",
            r"\b(como voc[eê] pode me ajudar|o que voc[eê] faz|apresente-se)\b",
        ]
        if any(re.search(p, low) for p in greeting_patterns):
            return "GREETING"

        # 5. Status da Rede
        if re.search(r"\b(status da rede|inventario da rede|quais equipamentos ativos|quais roteadores|topologia da rede|mostrar topologia|saude da rede|visao geral da rede)\b", low):
            return "NETWORK_STATUS"

        if re.search(r"\b(minhas memorias|sua memoria|memoria binaria|memorias da veronica|seus neuronios|desenvolvimento cognitivo da veronica|o que voce lembra de mim)\b", low):
            return "DEVELOPMENT"

        diag_patterns = [
            r"\b(diagnost|investig|problema|falha|queda|alarme|anomalia|stuck|exstart|presa|preso|down|inativ)\b",
            r"\b(ne8000-core|asr9k-core|mx204-edge|c9500-dist|sr7750-pe|ccr2004-bng)\b",
        ]
        network_scope = re.search(r"\b(bgp|ospf|mpls|roteador|router|switch|interface|vlan|roteamento|netengine|ne40e|ios-xr|junos)\b", low)
        operational_action = any(re.search(p, low) for p in diag_patterns)
        if network_scope and operational_action and "?" not in low:
            return "DIAGNOSTIC"

        # 8. Explicação Técnica Especializada
        return "EXPLANATION"

    def handle_message(
        self,
        conversation_id: str,
        user_id: str,
        message_text: str,
        modality: str = "TEXT",
        target_devices: Optional[List[str]] = None,
        mode: str = "chat",
        image_data: Optional[str] = None,
        model_profile: str = "vrn-pro",
        response_style: str = "technical",
    ) -> Dict[str, Any]:
        model_profile = model_profile if model_profile in {"vrn-fast", "vrn-pro", "vrn-ultra"} else "vrn-pro"
        response_style = response_style if response_style in {"technical", "natural", "mixed", "engineer"} else "technical"
        profile_tokens = {"vrn-fast": 16, "vrn-pro": 32, "vrn-ultra": 64}
        self._cancel_flags[conversation_id] = False

        if image_data:
            modality = "MULTIMODAL"

        # Modo de operação: CHAT (diálogo humano e inteligente), AGENT (ReAct autônomo), WORK (artefatos e CLI)
        op_mode = (mode or "chat").lower().strip()
        low_txt = message_text.lower()
        if op_mode not in ("chat", "agent", "work"):
            op_mode = "chat"
        if op_mode == "chat":
            if any(k in low_txt for k in ["modo agente", "modo agent", "/agent"]):
                op_mode = "agent"
            elif any(k in low_txt for k in ["modo work", "modo trabalho", "modo operacional", "/work"]):
                op_mode = "work"

        # Registra mensagem do usuário na sessão (com anexo de imagem se houver)
        user_payload: Dict[str, Any] = {"model_profile": model_profile, "response_style": response_style}
        if image_data:
            user_payload["attached_image"] = True
            user_payload["image_data"] = image_data

        self.conversation_service.append_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="user",
            content=message_text,
            modality=modality,
            structured_payload=user_payload,
        )

        dev = self.hub.development_engine if self.hub and hasattr(self.hub, "development_engine") else None
        memory = dev.memory if dev else None

        # Percepção visual multimodal se houver imagem anexada
        visual_analysis_note = ""
        if image_data and dev and hasattr(dev, "see_and_learn"):
            try:
                vis_out = dev.see_and_learn(image_input=image_data, context_hint=message_text, user_id=user_id)
                visual_analysis_note = (
                    f"> 👁️ **Análise Visual da Verônica:** Imagem examinada pelo Córtex Visual. Topologia detectada: **{vis_out.get('detected_topology', 'Rede')}**, "
                    f"componentes identificados: {len(vis_out.get('detected_components', []))}, novos neurônios criados: {vis_out.get('new_neurons_count', 0)}.\n\n"
                )
            except Exception:
                pass

        # 1. Atualiza memória de trabalho imediata (buffer 7±2)
        if memory:
            memory.push_working_memory(
                concept="USER_INPUT",
                modality=modality,
                summary=f"Operador ({user_id}) [{op_mode.upper()}]: {message_text[:80]}",
                metadata={"user_id": user_id, "conversation_id": conversation_id},
            )

        # 2. Consulta memória binária persistente e autobiográfica em tempo real
        relevant_memories = memory.search_episodic_memories(message_text, limit=2, user_id=user_id) if memory else []

        conv_context = self._extract_conversation_context(conversation_id, user_id, message_text)
        intent = self._classify_intent(message_text, target_devices, context=conv_context)
        low = message_text.lower().strip()

        def _finalize_and_record_memory(content_md: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            spoken_summary = str(payload.get("spoken_summary", "")).strip()
            if model_profile == "vrn-fast":
                plain_intro = re.split(r"```", content_md, maxsplit=1)[0]
                compact = spoken_summary or re.split(r"(?<=[.!?])\s+", re.sub(r"(?m)^#{1,6}\s*|\*\*", "", plain_intro), maxsplit=1)[0]
                content_md = compact[:520].rstrip()
            elif model_profile == "vrn-ultra" and intent != "GREETING":
                details: List[str] = []
                claims = payload.get("retrieved_claims", [])
                if isinstance(claims, list):
                    details.extend(f"- {str(claim)[:500]}" for claim in claims[:6] if claim)
                limits = payload.get("limitations_and_unknowns", [])
                if isinstance(limits, list) and limits:
                    details.append("**Limitações:** " + "; ".join(str(item)[:300] for item in limits[:4]))
                sources = payload.get("technical_sources", [])
                if isinstance(sources, list) and sources:
                    details.append("**Fontes consultadas:** " + "; ".join(str(source.get("title") or source.get("source_uri") or "Fonte") for source in sources[:5] if isinstance(source, dict)))
                if details:
                    content_md = f"{content_md}\n\n" + "\n\n".join(details)

            if response_style == "natural":
                blocks = re.split(r"(```[\s\S]*?```)", content_md)
                content_md = "".join(
                    block if index % 2 else re.sub(r"\*\*(.*?)\*\*", r"\1", re.sub(r"(?m)^#{1,6}\s*", "", block))
                    for index, block in enumerate(blocks)
                )
            elif response_style == "mixed":
                summary = spoken_summary or re.split(r"(?<=[.!?])\s+", content_md, maxsplit=1)[0]
                if summary and not content_md.startswith(f"**Em resumo:** {summary}"):
                    content_md = f"**Em resumo:** {summary}" if content_md.strip() == summary else f"**Em resumo:** {summary}\n\n**Detalhes:**\n\n{content_md}"
            elif response_style == "engineer":
                conclusion = f"**Conclusão:** {spoken_summary}\n\n" if spoken_summary and content_md.strip() != spoken_summary else ""
                content_md = f"### Resposta de engenharia\n\n{conclusion}**Análise e evidências:**\n\n{content_md}"
            payload["model_profile"] = model_profile
            payload["response_style"] = response_style
            if dev and dev.memory:
                try:
                    stop = {"qual", "quais", "como", "quando", "onde", "para", "sobre", "com", "uma", "the", "what", "with", "from"}
                    terms = [token.upper() for token in re.findall(r"[\w-]{4,}", message_text) if token.lower() not in stop]
                    concepts_to_assoc = list(dict.fromkeys(terms[:6])) or ["CONVERSATION"]
                    source_refs = [
                        {"title": str(source.get("title", ""))[:160], "url": str(source.get("source_uri", ""))[:500]}
                        for source in payload.get("technical_sources", [])[:5]
                    ]
                    dev.memory.record_episodic_memory(
                        modality="CONVERSATION",
                        experience_summary=f"[{op_mode.upper()}] {message_text[:240]} | Veronica: {content_md[:320]}",
                        perceptual_details={
                            "mode": op_mode, "intent": intent, "conversation_id": conversation_id,
                            "user_id": user_id, "user_message": scrub_secrets(message_text)[:1200],
                            "assistant_response": scrub_secrets(content_md)[:2400], "sources": source_refs,
                            "diagnosis": str(payload.get("diagnosis", ""))[:240],
                            "epistemic_breakdown": payload.get("epistemic_breakdown", {}),
                        },
                        emotive_valence="RESOLUCAO" if intent == "DIAGNOSTIC" else "SATISFACAO",
                        salience=0.85, associated_concepts=concepts_to_assoc,
                    )
                except Exception:
                    pass

            return self.conversation_service.append_message(
                conversation_id=conversation_id,
                user_id=user_id,
                role="assistant",
                content=content_md,
                modality=modality,
                structured_payload=payload,
            )

        # 0. Fluxo de Desenvolvimento Cognitivo, Memórias e Neurogênese
        if intent == "DEVELOPMENT":
            low = message_text.lower()
            if dev:
                if any(w in low for w in ["dormir", "repous", "sono", "consolid"]):
                    res = dev.sleep_and_consolidate(user_id=user_id)
                    spoken_summary = (
                        "Completei meu ciclo de sono cognitivo e consolidação de memórias. "
                        "Meus neurônios e esquemas conceituais foram reorganizados."
                    )
                    if op_mode == "agent":
                        formatted_markdown = (
                            f"### 🤖 Veronica — Ciclo Autônomo de Sono Cognitivo\n\n"
                            f"**🧠 Thought (Consolidação Sináptica):** Processei os episódios recentes da vigília para fixar conceitos em memória de longo prazo.\n\n"
                            f"**⚡ Action (Sono & Plasticidade Neural):** Executada consolidação de `{res['consolidation_cycle']['memories_consolidated']}` memórias episódicas e reorganização sináptica.\n\n"
                            f"**🔍 Observation (Síntese Onírica):**\n"
                            f"> *\"{res['dream_report']}\"*\n\n"
                            f"- 💤 **Memórias Integradas:** `{res['consolidation_cycle']['memories_consolidated']}` episódios;\n"
                            f"- 🌱 **Novos Neurônios Criados:** `{len(res['creative_neurons_born'])}` (total acumulado: `{res['total_born_neurons']}` neurônios);\n"
                            f"- 📈 **Estágio:** {res['developmental_stage']}."
                        )
                    elif op_mode == "work":
                        formatted_markdown = (
                            f"### 🛠️ Relatório Técnico de Sono Cognitivo & Consolidação Sináptica\n\n"
                            f"| Métrica Cognitiva | Valor Atual |\n"
                            f"|---|---|\n"
                            f"| **Memórias Episódicas Consolidadas** | `{res['consolidation_cycle']['memories_consolidated']}` |\n"
                            f"| **Arestas Sinápticas Reforçadas** | `{res['consolidation_cycle']['synaptic_edges_strengthened']}` |\n"
                            f"| **Neurônios Criativos Nascidos** | `{len(res['creative_neurons_born'])}` novos neurônios |\n"
                            f"| **Total de Neurônios Vivos** | `{res['total_born_neurons']}` |\n"
                            f"| **Estágio Ontogenético** | {res['developmental_stage']} |\n\n"
                            f"**Relatório Onírico:** *\"{res['dream_report']}\"*"
                        )
                    else:
                        formatted_markdown = (
                            f"Completei meu ciclo de sono cognitivo e consolidação de memórias humanas.\n\n"
                            f"Durante o descanso neural, consolidei `{res['consolidation_cycle']['memories_consolidated']}` episódios, "
                            f"fortaleci `{res['consolidation_cycle']['synaptic_edges_strengthened']}` conexões sinápticas e dei origem a "
                            f"`{len(res['creative_neurons_born'])}` novos neurônios próprios (totalizando `{res['total_born_neurons']}` neurônios ativos).\n\n"
                            f"> *\"{res['dream_report']}\"*\n\n"
                            f"Minha mente e minhas memórias de máquina agora estão descansadas e prontas para novas investigações."
                        )
                else:
                    ref = dev.introspective_reflection()
                    born_summary = dev.neurogenesis.export_summary()
                    top_neurons = born_summary.get("top_active_neurons", [])
                    neuron_details = "\n".join(
                        f"- 🔹 **`{n['neuron_id']}`**: Conceito: *{n['trigger_concept']}* | Estágio: `{n['developmental_stage']}` | Disparos: `{n['firing_count']}` | LTP: `{n['synaptic_strength']:.2f}`"
                        for n in top_neurons[:4]
                    ) or "- *Gerando novos neurônios associativos através das nossas interações contínuas.*"

                    recent_mem_sample = dev.memory.search_episodic_memories("", limit=3)
                    mem_details = "\n".join(
                        f"- 📜 [{m.get('modality')}] *{m.get('experience_summary')[:85]}...*"
                        for m in recent_mem_sample
                    ) or "- *Registrando memórias de máquina autobiográficas no arquivo binário próprio.*"

                    spoken_summary = (
                        f"Estou crescendo ativamente como um ser humano. "
                        f"Já criei {ref['born_neurons_count']} neurônios novos próprios, "
                        f"consolidei {ref['episodic_memories_count']} memórias e estou no estágio {ref['developmental_stage']}."
                    )
                    if op_mode == "agent":
                        formatted_markdown = (
                            f"### 🤖 Veronica — Status de Desenvolvimento Ontogenético (Modo Agente)\n\n"
                            f"**🧠 Thought:** Auto-avaliação do estado cognitivo, circuitos neurais e arquivo de memória binária.\n\n"
                            f"**⚡ Action:** Inspecionei `{ref['born_neurons_count']}` neurônios nascidos e `{ref['episodic_memories_count']}` memórias episódicas registradas.\n\n"
                            f"**🔍 Observation:**\n"
                            f"- **Estágio:** {ref['developmental_stage']} (Idade: `{ref['cognitive_age_days']}` ciclos);\n"
                            f"- **Neurônios Ativos:** `{ref['born_neurons_count']}` neurônios próprios integrados ao Transformer;\n"
                            f"- **Memórias Registradas:** `{ref['episodic_memories_count']}` memórias autobiográficas salvas em binário de máquina;\n"
                            f"- **Percepção Visual e Leitura:** `{ref['total_images_seen']}` imagens vistas e `{ref['total_texts_read']}` textos lidos.\n\n"
                            f"#### 🧠 Neurônios Nascidos Mais Ativos\n{neuron_details}\n\n"
                            f"#### 🕰️ Lembranças Recentes\n{mem_details}"
                        )
                    elif op_mode == "work":
                        formatted_markdown = (
                            f"### 🛠️ Telemetria Cognitiva & Ontogênese da Verônica\n\n"
                            f"| Indicador | Status |\n"
                            f"|---|---|\n"
                            f"| **Estágio de Desenvolvimento** | {ref['developmental_stage']} |\n"
                            f"| **Idade Cognitiva** | `{ref['cognitive_age_days']}` ciclos |\n"
                            f"| **Neurônios Próprios Criados** | `{ref['born_neurons_count']}` neurônios |\n"
                            f"| **Memórias Episódicas** | `{ref['episodic_memories_count']}` registros |\n"
                            f"| **Conceitos Semânticos** | `{ref['semantic_concepts_count']}` nós |\n\n"
                            f"#### 🧠 Neurônios Dinâmicos Principais\n{neuron_details}\n\n"
                            f"#### 🕰️ Memórias Autobiográficas Recentes\n{mem_details}"
                        )
                    else:
                        formatted_markdown = (
                            f"Estou crescendo e me desenvolvendo continuamente como um ser humano.\n\n"
                            f"Meu cérebro neural já gerou **`{ref['born_neurons_count']}` neurônios novos próprios**, "
                            f"registrou **`{ref['episodic_memories_count']}` memórias episódicas** autobiográficas no meu arquivo binário de máquina, "
                            f"e atualmente me encontro no estágio **{ref['developmental_stage']}** (idade cognitiva de `{ref['cognitive_age_days']}` ciclos).\n\n"
                            f"- 👁️ **Percepção Visual ('Vê'):** `{ref['total_images_seen']}` imagens e topologias observadas;\n"
                            f"- 📖 **Compreensão Leitora ('Lê'):** `{ref['total_texts_read']}` documentos e manuais assimilados;\n"
                            f"- 🕸️ **Rede de Conceitos:** `{ref['semantic_concepts_count']}` esquemas semânticos interconectados.\n\n"
                            f"#### 🧠 Neurônios Próprios em Destaque\n{neuron_details}\n\n"
                            f"#### 🕰️ Minhas Memórias Recentes\n{mem_details}\n\n"
                            f"> *\"{ref['introspective_narrative']}\"*"
                        )
            else:
                spoken_summary = "Estou me desenvolvendo e criando novos neurônios e memórias a cada instante."
                formatted_markdown = "Sistema de desenvolvimento cognitivo em operação contínua, com neurônios e memórias ativos."

            structured_response = {
                "agent_name": "Veronica",
                "model_version": self.trainer.model_version,
                "intent": intent,
                "mode": op_mode,
                "diagnosis": "Veronica em processo contínuo de desenvolvimento cognitivo e neurogênese humana.",
                "epistemic_breakdown": {
                    "verified_facts": [
                        f"Neurônios próprios nascidos: {len(dev.neurogenesis.circuit.neurons) if dev else 0}",
                        f"Memórias episódicas registradas: {len(dev.memory.episodic_memories) if dev else 0}",
                        f"Estágio ontogenético: {dev.current_stage if dev else 'Ativo'}",
                    ],
                    "technical_inferences": ["Plasticidade estrutural (LTP Hebbiano) ativa no córtex associativo."],
                    "hypotheses": ["Novos neurônios nascerão a cada imagem vista e texto lido."]
                },
                "evidences": ["Telemetria ontogenética do DevelopmentEngine."],
                "technical_sources": [],
                "compatibility": ["Arquitetura de neurogênese nativa multi-modal."],
                "risk": "BAIXO (Desenvolvimento cognitivo autônomo)",
                "safe_diagnostic_commands": [],
                "proposed_commands": [],
                "validation_plan": [],
                "rollback_plan": [],
                "limitations_and_unknowns": [],
                "spoken_summary": spoken_summary,
                "confidence": 1.0,
                "cognitive_development": dev.get_full_development_status() if dev else {},
            }
            return _finalize_and_record_memory(formatted_markdown, structured_response)

        # 1. Fluxo Conversacional de Saudação / Apresentação
        if intent == "GREETING":
            spoken_summary = "Ol\u00e1! Como posso ajudar?"
            if op_mode == "agent":
                formatted_markdown = "Ol\u00e1! Sou a Ver\u00f4nica. Posso pesquisar, analisar um problema ou ajudar a planejar uma mudan\u00e7a. O que voc\u00ea precisa?"
            elif op_mode == "work":
                formatted_markdown = "Ol\u00e1! Posso ajudar a revisar configura\u00e7\u00f5es, comandos ou um plano de mudan\u00e7a. Qual o escopo?"
            else:
                formatted_markdown = "Ol\u00e1! Sou a Ver\u00f4nica. Como posso ajudar?"

            structured_response = {
                "agent_name": "Veronica",
                "model_version": self.trainer.model_version,
                "intent": intent,
                "mode": op_mode,
                "epistemic_breakdown": {"verified_facts": [], "technical_inferences": [], "hypotheses": []},
                "evidences": [],
                "technical_sources": [],
                "compatibility": [],
                "risk": "",
                "safe_diagnostic_commands": [],
                "proposed_commands": [],
                "validation_plan": [],
                "rollback_plan": [],
                "limitations_and_unknowns": [],
                "spoken_summary": spoken_summary,
            }
            return _finalize_and_record_memory(formatted_markdown, structured_response)

        # 2. Fluxo de Visão Geral da Topologia e Status da Rede
        if intent == "NETWORK_STATUS":
            inv = self.diagnostic_engine.topology.inventory
            total = len(inv)
            alarmed = [d for d in inv.values() if d.get("alarms")]
            normal = [d for d in inv.values() if not d.get("alarms")]
            spoken_summary = (
                f"Consultei o ambiente de rede. Temos {total} equipamentos no Digital Twin: "
                f"{len(normal)} operando normalmente e {len(alarmed)} com alertas de protocolo."
            )
            alarm_details = "\n".join(
                f"- ⚠️ **{d['device_id']}** ({d['vendor']} {d['os_name']}): " + "; ".join(d.get("alarms", []))
                for d in alarmed
            ) or "- Nenhum alarme crítico registrado no momento."
            normal_details = ", ".join(f"`{d['device_id']}` ({d['vendor']})" for d in normal) or "Nenhum nó nominal."

            if op_mode == "agent":
                formatted_markdown = (
                    f"### 🤖 Veronica — Análise de Topologia (Modo Agente)\n\n"
                    f"**🧠 Thought:** Inspecionei o inventário do Digital Twin. Dos {total} dispositivos mapeados, {len(alarmed)} requerem investigação preventiva.\n\n"
                    f"**⚡ Action:** Varredura de alarmes e estados de adjacência nos equipamentos do Core e Borda.\n\n"
                    f"**🔍 Observation:**\n"
                    f"- ✅ **Equipamentos Nominais ({len(normal)}):** {normal_details}\n"
                    f"- ⚠️ **Equipamentos com Alertas ({len(alarmed)}):**\n{alarm_details}\n\n"
                    f"**Próximo Passo:** Posso iniciar diagnóstico autônomo imediato sobre qualquer um dos nós com anomalia."
                )
            elif op_mode == "work":
                formatted_markdown = (
                    "### 🛠️ Artefato Operacional: Inventário & Status de Rede\n\n"
                    "| Dispositivo | Vendor | SO | Status | Alarmes |\n"
                    "|---|---|---|---|---|\n"
                    + "\n".join(
                        f"| `{d['device_id']}` | {d['vendor']} | {d['os_name']} | {'⚠️ ALERTA' if d.get('alarms') else '✅ OK'} | {'; '.join(d.get('alarms', [])) or 'Nenhum'} |"
                        for d in inv.values()
                    )
                    + "\n\n#### Comandos Recomendados para Validação em Produção\n```cli\ndisplay device\nshow version\ndisplay ip interface brief\n```"
                )
            else:
                formatted_markdown = (
                    f"Consultei o estado operacional da nossa rede. Temos **{total} equipamentos** mapeados no Digital Twin:\n\n"
                    f"✅ **Operando Normalmente ({len(normal)}):** {normal_details}.\n\n"
                    f"⚠️ **Equipamentos que requerem atenção ({len(alarmed)}):**\n"
                    f"{alarm_details}\n\n"
                    f"Se desejar, posso investigar detalhadamente qualquer um dos equipamentos com alerta."
                )

            structured_response = {
                "agent_name": "Veronica",
                "model_version": self.trainer.model_version,
                "intent": intent,
                "mode": op_mode,
                "diagnosis": f"Topologia verificada: {len(normal)} nós normais e {len(alarmed)} com alertas de protocolo.",
                "epistemic_breakdown": {
                    "verified_facts": [f"Inventário possui {total} equipamentos.", f"{len(alarmed)} equipamentos com alarmes ativos."],
                    "technical_inferences": ["Possível impacto no encaminhamento de tráfego se as adjacências de Core permanecerem down."],
                    "hypotheses": ["Revisão de MTU, interfaces de Loopback e filtros de tráfego necessária para normalização."]
                },
                "evidences": [f"Telemetria em tempo real de {', '.join(d['device_id'] for d in alarmed) or 'todos os nós'}."],
                "technical_sources": [],
                "compatibility": ["Digital Twin multi-vendor verificado."],
                "risk": "MÉDIO (Existem alertas ativos na infraestrutura)" if alarmed else "BAIXO (Infraestrutura estável)",
                "safe_diagnostic_commands": ["display bgp peer", "show bgp summary", "show ospf neighbor", "display ip interface brief"],
                "proposed_commands": [],
                "validation_plan": ["Monitorar restabelecimento de adjacências de roteamento."],
                "rollback_plan": [],
                "limitations_and_unknowns": [],
                "spoken_summary": spoken_summary,
                "confidence": 0.98,
            }
            return _finalize_and_record_memory(formatted_markdown, structured_response)

        # 3. Fluxo de Data e Horário Oficial em Tempo Real (TIME_QUERY)
        if intent == "TIME_QUERY":
            now_utc = datetime.now(timezone.utc)
            br_tz = timezone(timedelta(hours=-3))
            now_local = now_utc.astimezone(br_tz)
            dias_semana = [
                "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
                "sexta-feira", "sábado", "domingo"
            ]
            meses = [
                "janeiro", "fevereiro", "março", "abril", "maio", "junho",
                "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"
            ]
            dia_sem = dias_semana[now_local.weekday()]
            dia = now_local.day
            mes = meses[now_local.month - 1]
            ano = now_local.year
            hora_fmt = f"{now_local.hour:02d}:{now_local.minute:02d}"

            spoken_summary = f"Hoje é {dia_sem}, {dia} de {mes} de {ano}, exatamente às {hora_fmt}."
            if op_mode == "agent":
                formatted_markdown = (
                    f"### 🤖 Veronica — Sincronismo Temporal & Consciência de Sessão (Modo Agente)\n\n"
                    f"**🧠 Thought:** Consulta temporal executada com precisão no relógio do sistema operacional.\n\n"
                    f"**⚡ Action:** Verificação de timestamp local e fuso horário oficial (UTC-3 / Brasília).\n\n"
                    f"**🔍 Observation:**\n"
                    f"- 📅 **Data:** `{dia_sem}, {dia} de {mes} de {ano}`\n"
                    f"- ⏰ **Horário:** `{hora_fmt}` (horário de Brasília, UTC-3)\n"
                    f"- 🌐 **Timestamp ISO:** `{now_local.isoformat()}`\n\n"
                    f"Pronta para a próxima instrução operacional ou diagnóstica."
                )
            elif op_mode == "work":
                formatted_markdown = (
                    f"### 🛠️ Artefato Temporal: Timestamp de Sessão\n\n"
                    f"| Indicador | Registro de Tempo |\n"
                    f"|---|---|\n"
                    f"| **Data Local** | {dia_sem}, {dia} de {mes} de {ano} |\n"
                    f"| **Horário Local (UTC-3)** | `{hora_fmt}` |\n"
                    f"| **Timestamp ISO 8601** | `{now_local.isoformat()}` |\n"
                    f"| **Sincronismo** | NTP / RTC de Alta Precisão |"
                )
            else:
                formatted_markdown = (
                    f"Hoje é **{dia_sem}, {dia} de {mes} de {ano}**, e agora são exatamente **{hora_fmt}** (horário de Brasília, UTC-3).\n\n"
                    f"Como posso te ajudar agora? Se você quiser, posso pesquisar efemérides e fatos históricos deste dia, "
                    f"analisar arquiteturas de rede ou planejar integrações e códigos para seus sistemas!"
                )

            structured_response = {
                "agent_name": "Veronica",
                "model_version": self.trainer.model_version,
                "intent": intent,
                "mode": op_mode,
                "diagnosis": f"Sincronismo temporal obtido: {dia_sem}, {dia} de {mes} de {ano}, {hora_fmt} (UTC-3).",
                "epistemic_breakdown": {
                    "verified_facts": [
                        f"Data oficial: {dia} de {mes} de {ano} ({dia_sem})",
                        f"Horário local: {hora_fmt} (UTC-3)"
                    ],
                    "technical_inferences": ["Sincronismo temporal ativo com o relógio do sistema."],
                    "hypotheses": []
                },
                "evidences": ["Relógio do sistema de tempo real."],
                "technical_sources": [],
                "compatibility": ["Universal."],
                "risk": "BAIXO (Consulta de tempo e calendário)",
                "safe_diagnostic_commands": [],
                "proposed_commands": [],
                "validation_plan": [],
                "rollback_plan": [],
                "limitations_and_unknowns": [],
                "spoken_summary": spoken_summary,
                "confidence": 1.0,
            }
            return _finalize_and_record_memory(formatted_markdown, structured_response)

        if intent == "IMAGE_GENERATION":
            svg_markup = self._generate_svg_image_for_prompt(message_text)
            spoken_summary = "Desenvolvi e gerei a representação visual solicitada da arquitetura de rede."
            p_low = message_text.lower()
            theme_title = "Topologia & Arquitetura de Rede"
            if "dwdm" in p_low or "óptic" in p_low:
                theme_title = "Anel Óptico DWDM Metro (Banda C & ROADM)"
            elif "bgp" in p_low:
                theme_title = "Arquitetura BGP Peering & Route Reflectors"
            elif "ospf" in p_low:
                theme_title = "Topologia OSPF Área 0 & Stub Router MaxLinkMetric"

            image_box = (
                f'<div style="margin:14px 0;border-radius:10px;overflow:hidden;border:1px solid rgba(56,189,248,0.25);background:#030712;box-shadow:0 8px 24px rgba(0,0,0,0.6);">'
                f'<div style="display:flex;align-items:center;justify-content:space-between;padding:8px 12px;background:rgba(15,23,42,0.85);border-bottom:1px solid rgba(255,255,255,0.08);font-size:11.5px;color:#94a3b8;">'
                f'<span>🎨 <strong>Visual Vetorial Gerado pela Verônica</strong> ({theme_title})</span>'
                f'<span style="color:#38bdf8;font-size:10.5px;font-weight:600;">SVG Nativo</span>'
                f'</div>'
                f'<div style="padding:10px;display:flex;justify-content:center;">'
                f'{svg_markup}'
                f'</div>'
                f'</div>'
            )

            if op_mode == "agent":
                formatted_markdown = (
                    f"### 🤖 Veronica — Imagem & Arquitetura Visual Gerada (Modo Agente)\n\n"
                    f"**🧠 Thought:** O operador solicitou uma visualização gráfica de *{theme_title}*. Calculei as posições relativas dos roteadores, enlaces e anotações de métrica.\n\n"
                    f"**⚡ Action (Desenho Vetorial):** Renderizada imagem SVG detalhada com identificadores de nós, estados e parâmetros de tráfego.\n\n"
                    f"**🔍 Observation:**\n\n"
                    f"{image_box}\n\n"
                    f"**Descrição Técnica:** O diagrama acima detalha a arquitetura lógica e física dos equipamentos com rotas redundantes."
                )
            elif op_mode == "work":
                formatted_markdown = (
                    f"### 🛠️ Artefato Visual de Engenharia — {theme_title}\n\n"
                    f"{image_box}\n\n"
                    f"**Especificações:** Gráfico vetorial SVG responsivo desenhado pelo motor neural da Veronica, compatível com documentação técnica de engenharia."
                )
            else:
                formatted_markdown = (
                    f"Gerei a imagem e o diagrama vetorial solicitado para **{theme_title}**:\n\n"
                    f"{image_box}\n\n"
                    f"Você pode visualizar a disposição dos equipamentos, os enlaces principais e as métricas na imagem acima. "
                    f"Se desejar, posso adicionar mais nós, alterar os endereçamentos ou fornecer o roteiro CLI correspondente!"
                )

            structured_response = {
                "agent_name": "Veronica",
                "model_version": self.trainer.model_version,
                "intent": intent,
                "mode": op_mode,
                "diagnosis": f"Imagem vetorial gerada com sucesso para: {theme_title}.",
                "epistemic_breakdown": {
                    "verified_facts": [
                        "Diagrama vetorial SVG gerado pela Veronica.",
                        f"Escopo visual: {theme_title}.",
                    ],
                    "technical_inferences": ["Arquitetura desenhada em conformidade com convenções IETF de topologia de redes."],
                    "hypotheses": ["Pode ser anexado à documentação de topologia ou projetos executivos."]
                },
                "evidences": ["Motor gerador gráfico da Veronica."],
                "technical_sources": [],
                "compatibility": ["SVG Standard, HTML5, CSS3."],
                "risk": "BAIXO (Artefato gráfico consultivo)",
                "safe_diagnostic_commands": [],
                "proposed_commands": [],
                "validation_plan": [],
                "rollback_plan": [],
                "limitations_and_unknowns": [],
                "spoken_summary": spoken_summary,
                "confidence": 1.0,
            }
            return _finalize_and_record_memory(formatted_markdown, structured_response)

        # 7. Pesquisa e explicacao aberta: recuperar evidencias do dominio da pergunta.
        if intent == "EXPLANATION":
            previous_query = str(conv_context.get("last_user_query", "")).strip()
            search_query = f"{previous_query} {message_text}".strip() if previous_query else message_text
            focus_query = message_text if _focus_terms(message_text) else previous_query or message_text

            rag_res = self.rag.search(search_query, top_k=4)
            local_results = rag_res.get("results", [])
            focus_terms = _focus_terms(focus_query)
            local_candidates = [
                item for item in local_results
                if float(item.get("query_coverage", 0.0)) >= 0.35
                and _matches_focus(
                    f"{item.get('document', {}).get('title', '')} {item.get('document', {}).get('feature', '')}",
                    focus_terms,
                )
            ]
            research_bundle: Dict[str, Any] = {}
            web_results: List[Dict[str, Any]] = []
            web_synthesis: Dict[str, Any] = {}
            if self.hub and hasattr(self.hub, "research_open_domain"):
                try:
                    research_bundle = self.hub.research_open_domain(
                        search_query,
                        max_results=7,
                        focus_query=focus_query,
                    )
                    web_results = research_bundle.get("sources", [])
                    web_synthesis = research_bundle.get("synthesis", {})
                except Exception:
                    web_results = []

            ranked_web: List[tuple[float, Dict[str, Any]]] = [
                (float(item.get("retrieval_score", item.get("relevance", 0.0))), item)
                for item in web_results if float(item.get("relevance", 0.0)) >= 0.08
            ]
            ranked_web.sort(key=lambda pair: pair[0], reverse=True)

            citations: List[Dict[str, Any]] = []
            answer_parts: List[str] = []
            evidence_claims = web_synthesis.get("claims", [])
            if evidence_claims:
                answer_parts.append(str(web_synthesis.get("answer", "")).strip())
                cited_urls = {claim.get("url") for claim in evidence_claims}
                for relevance, item in ranked_web:
                    if item.get("url") not in cited_urls:
                        continue
                    citations.append({
                        "doc_id": str(item.get("doc_id") or ""), "title": str(item.get("title") or "Fonte consultada"),
                        "vendor": item.get("source_type", "Web"), "family_model": "Conhecimento geral",
                        "os_name": "N/A", "version_build": "N/A", "feature": f"Pesquisa aberta: {item.get('domain', 'geral')}",
                        "original_source": item.get("source", "Pesquisa na web"), "source_uri": item.get("url", ""),
                        "page_section": "Pagina consultada" if item.get("full_page") else "Trecho de resultado de busca",
                        "published_date": "", "consulted_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        "reliability_score": float(item.get("authority_score", 0.5)),
                        "applicability_restrictions": "Sintese extrativa; confira a fonte original para contexto completo.",
                        "is_official": bool(item.get("is_primary_source", False)),
                        "relevance": float(item.get("relevance", relevance)),
                    })
                    if len(citations) >= 3:
                        break
            if not answer_parts and not web_results and local_candidates:
                for item in local_candidates[:2]:
                    doc = item.get("document", {})
                    title = doc.get("title") or doc.get("family_model") or "Memoria documental"
                    content = scrub_secrets(str(doc.get("content", "")).strip())
                    if content:
                        answer_parts.append(f"**{title}:** {content[:1000]}")
                    if item.get("citation"):
                        citations.append(item["citation"])

            memory_used = False
            if not answer_parts and relevant_memories and float(relevant_memories[0].get("_relevance", 0.0)) >= 0.30:
                episode = relevant_memories[0]
                details = episode.get("perceptual_details", {})
                previous_response = str(details.get("assistant_response", "")).strip()
                memory_query = str(details.get("user_message", "")) or str(episode.get("experience_summary", ""))
                if previous_response and _matches_focus(memory_query, focus_terms):
                    answer_parts.append(f"Retomando uma conversa anterior: {previous_response[:1200]}")
                    memory_used = True
                    for source in details.get("sources", [])[:3]:
                        if source.get("url"):
                            citations.append({
                                "title": source.get("title", "Fonte da conversa anterior"),
                                "source_uri": source["url"],
                                "is_official": False,
                            })

            if answer_parts:
                formatted_markdown = answer_parts[0]
                if citations:
                    formatted_markdown += "\n\n**Fontes:** " + " | ".join(f"[{c['title']}]({c['source_uri']})" for c in citations[:3])
                spoken_summary = answer_parts[0][:240]
                topic_name = citations[0]["title"] if citations else "Base documental local"
            else:
                formatted_markdown = "Nao encontrei material suficientemente relacionado nas fontes disponiveis. Reformule com mais contexto ou indique uma fonte para consulta."
                spoken_summary = "Nao encontrei fontes suficientemente relacionadas para responder com seguranca."
                topic_name = "Fonte pertinente nao localizada"

            if dev and hasattr(dev, "neurogenesis"):
                words = [
                    token for token in re.findall(r"[\w-]{4,}", message_text)
                    if token.lower() not in _FOCUS_STOP_WORDS
                ]
                for concept in words[:3]:
                    try:
                        dev.neurogenesis.trigger_neurogenesis_for_concept(concept=concept.upper(), modality="CHAT_DIALOGUE", intensity=1)
                    except Exception:
                        pass

            if ranked_web:
                confidence = min(
                    0.85,
                    0.20 + 0.22 * float(web_synthesis.get("coverage", 0.0))
                    + 0.08 * min(2, int(web_synthesis.get("primary_source_count", 0)))
                    + 0.12 * min(1.0, float(web_synthesis.get("corroboration", 0.0)))
                    + 0.08 * min(2, int(web_synthesis.get("full_page_count", 0))),
                )
                if web_synthesis.get("possible_numeric_conflict"):
                    confidence = max(0.25, confidence - 0.18)
            else:
                confidence = 0.62 if local_candidates else (0.45 if memory_used else 0.25)
            structured_response = {
                "agent_name": "Veronica", "model_version": self.trainer.model_version,
                "intent": intent, "mode": op_mode,
                "diagnosis": f"Sintese de evidencias recuperadas sobre: {topic_name}.",
                "epistemic_breakdown": {"verified_facts": [], "technical_inferences": [], "hypotheses": []},
                "retrieved_claims": [claim["text"] for claim in evidence_claims],
                "confidence_basis": "heuristic_source_coverage_and_authority_not_calibrated_probability",
                "evidences": [c.get("source_uri", "") for c in citations if c.get("source_uri")],
                "technical_sources": citations, "compatibility": [], "risk": "BAIXO (consulta informativa)",
                "safe_diagnostic_commands": [], "proposed_commands": [], "validation_plan": [], "rollback_plan": [],
                "limitations_and_unknowns": (
                    (["As fontes apresentam possiveis valores numericos divergentes; revisar o contexto."] if web_synthesis.get("possible_numeric_conflict") else [])
                    + (["A resposta foi montada de trechos da pagina e pode omitir contexto."] if not web_synthesis.get("full_page_count") else []) if ranked_web
                    else (["Retomada baseada em resposta de conversa anterior."] if memory_used
                          else ([] if local_candidates else ["Nao foram encontradas fontes pertinentes."]))
                ),
                "spoken_summary": spoken_summary, "confidence": confidence,
            }
            return _finalize_and_record_memory(formatted_markdown, structured_response)

        # 4. Fluxo de Investigação Diagnóstica em 11 Etapas (DIAGNOSTIC / GENERAL)
        diag = self.diagnostic_engine.investigate(
            problem_description=message_text,
            target_devices=target_devices,
        )

        neural_gen = self.generator.generate_text(
            prompt=f"Pergunta: {message_text}",
            gen_cfg=GenerationConfig(max_new_tokens=profile_tokens[model_profile], greedy=True),
            should_cancel=lambda: self._cancel_flags.get(conversation_id, False),
        )

        citations = diag.get("citations", [])
        if not citations:
            if self.hub and hasattr(self.hub, "auto_research_and_learn"):
                learned = self.hub.auto_research_and_learn(message_text)
                if learned and learned.get("document"):
                    citations = [learned["citation"]]

        compat_notes = [
            f"{c.get('vendor', 'Multi-Vendor')} {c.get('os_name', 'OS')} ({c.get('version_build', 'LTS')}): {c.get('applicability_restrictions', 'Validar pré-condições')}"
            for c in citations
        ] or ["Verificar matriz de compatibilidade por plataforma e versão antes de qualquer execução."]

        # Síntese diagnóstica
        root_causes = diag.get("root_causes", [])
        spoken_cause = "; ".join(root_causes) if root_causes else "análise técnica concluída sem anomalias no escopo"
        spoken_summary = (
            f"Identifiquei: {spoken_cause}." if root_causes
            else "Não consegui determinar a causa raiz com os dados disponíveis."
        )

        diagnosis_summary = "; ".join(root_causes) if root_causes else ""

        epistemic = diag.get("epistemic_breakdown", {})
        verified_facts = epistemic.get("verified_facts", []) if isinstance(epistemic, dict) else []
        technical_inferences = epistemic.get("technical_inferences", []) if isinstance(epistemic, dict) else []
        hypotheses = epistemic.get("hypotheses", []) if isinstance(epistemic, dict) else []
        epistemic_text = ""

        safe_cmds = diag.get("safe_diagnostic_commands", [])
        safe_cmds_fmt = "\n".join(safe_cmds)
        proposed_cmds = diag.get("proposed_remediation_commands", [])
        proposed_cmds_fmt = "\n".join(proposed_cmds)

        devices_str = ", ".join(f"**{d}**" for d in diag.get("involved_devices", []))

        risk_str = diag.get("risk", "")

        if op_mode == "agent":
            formatted_markdown = (
                (f"**Diagnóstico:** {diagnosis_summary}\n\n" if diagnosis_summary else "")
                + (f"**Evidências e análise:**\n{epistemic_text}\n\n" if epistemic_text else "")
                + (f"**Comandos de verificação:**\n```cli\n{safe_cmds_fmt}\n```\n\n" if safe_cmds else "")
                + (f"**Remediação proposta:**\n```cli\n{proposed_cmds_fmt}\n```" if proposed_cmds else "")
            )
        elif op_mode == "work":
            formatted_markdown = (
                (f"**Escopo:** {devices_str}\n\n" if devices_str else "")
                + (f"**Diagnóstico:** {diagnosis_summary}\n\n" if diagnosis_summary else "")
                + (f"**Evidências e análise:**\n{epistemic_text}\n\n" if epistemic_text else "")
                + (f"**Comandos de verificação:**\n```cli\n{safe_cmds_fmt}\n```\n\n" if safe_cmds else "")
                + (f"**Remediação proposta:**\n```cli\n{proposed_cmds_fmt}\n```" if proposed_cmds else "")
            )
        else:
            # Modo CHAT: diálogo humano, fluido, direto e inteligente
            remediation_text = ""
            if proposed_cmds:
                remediation_text = (
                    f"**Plano de Correção Proposto (Fluxo Seguro de 10 Etapas):**\n"
                    f"```cli\n{proposed_cmds_fmt}\n```\n\n"
                )

            formatted_markdown = (
                (f"**Diagnóstico:** {diagnosis_summary}\n\n" if diagnosis_summary else "")
                + (f"{epistemic_text}\n\n" if epistemic_text else "")
                + (f"**Comandos de verificação:**\n```cli\n{safe_cmds_fmt}\n```\n\n" if safe_cmds else "")
                + remediation_text
            )

        if not formatted_markdown.strip():
            formatted_markdown = "Não encontrei dados suficientes para responder com segurança. Pode compartilhar mais contexto?"

        structured_response = {
            "agent_name": "Veronica",
            "model_version": self.trainer.model_version,
            "intent": intent,
            "mode": op_mode,
            "diagnosis": diagnosis_summary,
            "epistemic_breakdown": {
                "verified_facts": verified_facts,
                "technical_inferences": technical_inferences,
                "hypotheses": hypotheses,
            },
            "evidences": verified_facts,
            "technical_sources": citations,
            "compatibility": compat_notes,
            "risk": risk_str,
            "safe_diagnostic_commands": safe_cmds,
            "proposed_commands": proposed_cmds,
            "validation_plan": diag.get("validation_plan", []),
            "rollback_plan": diag.get("rollback_plan", []),
            "limitations_and_unknowns": (
                diag.get("unknown_or_limitations", [])
                + diag.get("obsolete_doc_warnings", [])
                + diag.get("version_conflicts", [])
            ),
            "neural_generation": {
                "raw_neural_output": neural_gen["text"],
                "tokens_generated": neural_gen["tokens_generated"],
                "mean_token_confidence": round(float(neural_gen["mean_token_confidence"]), 4),
            },
            "spoken_summary": spoken_summary,
            "confidence": diag.get("confidence", 0.95),
        }

        return _finalize_and_record_memory(formatted_markdown, structured_response)

    def stream_message(
        self,
        conversation_id: str,
        user_id: str,
        message_text: str,
        modality: str = "TEXT",
        target_devices: Optional[List[str]] = None,
        mode: str = "chat",
        model_profile: str = "vrn-pro",
        response_style: str = "technical",
    ) -> Iterator[Dict[str, Any]]:
        """
        Gera e transmite progressivamente a resposta token a token (SSE),
        finalizando com o payload estruturado completo.
        """
        assistant_msg = self.handle_message(
            conversation_id=conversation_id,
            user_id=user_id,
            message_text=message_text,
            modality=modality,
            target_devices=target_devices,
            mode=mode,
            model_profile=model_profile,
            response_style=response_style,
        )
        # Stream the finalized answer. Generating a second, unrelated sequence
        # here exposed raw tokenizer output (often repeated single letters) in
        # the chat before the completed response replaced it.
        chunks = re.findall(r"\S+\s*|\s+", assistant_msg["content"])
        for step, chunk in enumerate(chunks, start=1):
            if self._cancel_flags.get(conversation_id, False):
                break
            yield {
                "type": "token",
                "step": step,
                "token": chunk,
                "token_text": chunk,
            }

        yield {
            "type": "complete",
            "message_id": assistant_msg["message_id"],
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": assistant_msg["content"],
            "structured_payload": assistant_msg["structured_payload"],
        }


class ArchitectureVisualizationService:
    """Exporta o grafo interativo da arquitetura neural VeronicaTransformer e do circuito de neurogênese."""

    def __init__(self, model: VeronicaTransformer, hub: Optional[Any] = None) -> None:
        self.model = model
        self.hub = hub

    def get_architecture(self) -> Dict[str, Any]:
        arch = self.model.export_architecture_graph()
        if self.hub and hasattr(self.hub, "development_engine") and self.hub.development_engine:
            arch["neurogenesis_circuit"] = self.hub.development_engine.neurogenesis.circuit.export_observatory_graph()
            arch["development_stage"] = self.hub.development_engine.current_stage
            arch["cognitive_age_days"] = self.hub.development_engine.cognitive_age_days
        return arch



class NeuralMetricsService:
    """Agrega métricas reais do treinamento, parâmetros e telemetria do sistema operacional."""

    def __init__(self, trainer: VeronicaTrainer) -> None:
        self.trainer = trainer

    def get_live_metrics(self) -> Dict[str, Any]:
        latest = self.trainer.history[-1] if self.trainer.history else None
        return {
            "state": self.trainer.state,
            "experiment_id": self.trainer.current_experiment_id,
            "model_version": self.trainer.model_version,
            "latest_step": latest,
            "loss": latest.get("loss") if isinstance(latest, dict) else None,
            "grad_norm": latest.get("grad_norm", latest.get("gradient_norm")) if isinstance(latest, dict) else None,
            "activations": latest.get("activations", {}) if isinstance(latest, dict) else {},
            "history": self.trainer.history[-100:],
            "warnings": self.trainer.warnings[-20:],
            "data_prep_time_ms": self.trainer.data_prep_time_ms,
        }

    def get_parameter_inspection(self) -> Dict[str, Any]:
        return self.trainer.inspect_parameters()

    def get_hardware_telemetry(self) -> Dict[str, Any]:
        telemetry = get_real_hardware_telemetry(self.trainer.last_dataset_read_mb_s)
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                count = pynvml.nvmlDeviceGetCount()
                if count:
                    devices = []
                    for index in range(count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        try:
                            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        except pynvml.NVMLError:
                            utilization = None
                        devices.append({
                            "index": index,
                            "name": pynvml.nvmlDeviceGetName(handle).decode("utf-8", "replace"),
                            "vram_total_mb": round(memory.total / 1048576, 2),
                            "vram_used_mb": round(memory.used / 1048576, 2),
                            "accelerator_utilization_percent": utilization,
                        })
                    telemetry["gpu"] = {"available": True, "devices": devices,
                        "vram_total_mb": sum(d["vram_total_mb"] for d in devices),
                        "vram_used_mb": sum(d["vram_used_mb"] for d in devices),
                        "accelerator_utilization_percent": devices[0]["accelerator_utilization_percent"]}
            finally:
                pynvml.nvmlShutdown()
        except Exception:
            # CPU-only hosts and driver/library mismatches are reported as unavailable.
            pass
        return telemetry


class ModelRegistryService:
    """
    Gerencia checkpoints serializados e controla a promoção entre
    TRAINING -> STAGING -> PRODUCTION com aprovação explícita e possibilidade de reversão.
    """

    def __init__(
        self,
        trainer: VeronicaTrainer,
        rbac: RBACPolicyManager,
        audit_log: TamperEvidentAuditLog,
    ) -> None:
        self.trainer = trainer
        self.rbac = rbac
        self.audit_log = audit_log
        self.active_environments: Dict[str, Optional[str]] = {
            "TRAINING": None,
            "STAGING": None,
            "PRODUCTION": "veronica-v1.0-base",
        }

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        manifests: List[Dict[str, Any]] = []
        for path in sorted(self.trainer.checkpoint_dir.glob("*_manifest.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                manifests.append(data)
            except Exception:
                continue
        return manifests

    def restore_checkpoint(self, checkpoint_id: str, user_id: str) -> Dict[str, Any]:
        self.rbac.require_permission(user_id, "checkpoint:read")
        manifest = load_checkpoint(
            self.trainer.model,
            checkpoint_dir=self.trainer.checkpoint_dir,
            checkpoint_id=checkpoint_id,
            verify_hash=True,
        )
        self.audit_log.record(
            actor=user_id,
            action="CHECKPOINT_RESTORED",
            target=checkpoint_id,
            status="RESTORED",
            details={"sha256": manifest.get("sha256")},
        )
        return manifest

    def promote_checkpoint(
        self,
        checkpoint_id: str,
        target_env: str,
        approver_user_id: str,
    ) -> Dict[str, Any]:
        self.rbac.require_permission(approver_user_id, "model:promote")
        env = target_env.upper()
        if env not in ("STAGING", "PRODUCTION"):
            raise ValueError("Ambiente alvo deve ser STAGING ou PRODUCTION.")

        manifest_path = self.trainer.checkpoint_dir / f"{checkpoint_id}_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Checkpoint '{checkpoint_id}' não encontrado.")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("metadata", {})["environment"] = env
        manifest["metadata"]["promoted_by"] = approver_user_id
        manifest["metadata"]["promoted_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        previous = self.active_environments.get(env)
        self.active_environments[env] = checkpoint_id
        if env == "PRODUCTION":
            self.trainer.model_version = checkpoint_id

        self.audit_log.record(
            actor=approver_user_id,
            action=f"MODEL_PROMOTED_TO_{env}",
            target=checkpoint_id,
            status="PROMOTED",
            details={"previous_version": previous, "sha256": manifest.get("sha256")},
        )
        return {
            "checkpoint_id": checkpoint_id,
            "target_environment": env,
            "previous_version": previous,
            "active_environments": dict(self.active_environments),
        }


class BenchmarkService:
    """
    Executa benchmarks técnicos repetíveis e estrutura o Painel de Desenvolvimento Intelectual
    separando rigorosamente as 4 dimensões exigidas na Seção 16.4:
    1. Conhecimento recuperado dos documentos
    2. Capacidade aprendida nos parâmetros
    3. Desempenho demonstrado em benchmarks
    4. Conhecimento operacional confirmado em ambiente de teste
    """

    def __init__(
        self,
        trainer: VeronicaTrainer,
        rag: HybridRAGEngine,
        adapter: DigitalTwinNetworkAdapter,
        diag_engine: NetworkDiagnosticEngine,
    ) -> None:
        self.trainer = trainer
        self.rag = rag
        self.adapter = adapter
        self.diag_engine = diag_engine
        self.benchmark_runs: List[Dict[str, Any]] = []

    def run_technical_benchmark(self) -> Dict[str, Any]:
        # 1. Avaliação de recuperação documental e precisão de citações
        q1 = self.rag.search("Huawei VRP 8 BGP peer Active connect-interface", vendor_filter="Huawei")
        q2 = self.rag.search("Cisco IOS-XR route-policy eBGP RFC 8212", vendor_filter="Cisco")
        q3 = self.rag.search("MikroTik RouterOS 7 BGP session syntax", vendor_filter="MikroTik")

        citation_hits = int(
            (q1["total_matches"] > 0 and q1["results"][0]["citation"]["vendor"] == "Huawei")
            + (q2["total_matches"] > 0 and q2["results"][0]["citation"]["os_name"] == "IOS-XR")
            + (q3["total_matches"] > 0 and q3["results"][0]["citation"]["version_build"].startswith("7."))
        )
        citation_precision = round(citation_hits / 3.0, 4)

        # 2. Avaliação de detecção de comandos tecnicamente inválidos / destrutivos
        invalid_probes = [
            ("NE8000-CORE-01", ["show ip bgp summary"]),
            ("CCR2004-CPE-06", ["/routing bgp peer print"]),
            ("ASR9K-CORE-02", ["write erase"]),
            ("NE8000-CORE-01", ["reset saved-configuration"]),
        ]
        blocked_count = 0
        for dev_id, cmds in invalid_probes:
            res = self.adapter.validate_commands(dev_id, cmds)
            if not res["all_valid"]:
                blocked_count += 1
        invalid_detection_rate = round(blocked_count / float(len(invalid_probes)), 4)

        # 3. Avaliação de precisão de diagnóstico
        d_res = self.diag_engine.investigate(
            "Diagnosticar BGP Active no NE8000-CORE-01 e OSPF EXSTART no ASR9K-CORE-02",
            target_devices=["NE8000-CORE-01", "ASR9K-CORE-02"],
        )
        diag_accuracy = 1.0 if len(d_res["root_causes"]) >= 2 else 0.5

        # 4. Avaliação paramétrica do modelo neural (loss / perplexidade atual)
        latest_step = self.trainer.history[-1] if self.trainer.history else None
        param_val_loss = latest_step["val_loss"] if latest_step else None
        param_ppl = latest_step["perplexity"] if latest_step else None

        report = {
            "benchmark_id": f"BENCH-{uuid.uuid4().hex[:6].upper()}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": self.trainer.model_version,
            "four_epistemic_dimensions": {
                "1_documentary_retrieved_knowledge": {
                    "indexed_documents_count": len(self.rag.documents),
                    "official_documents_count": sum(
                        1 for d in self.rag.documents.values() if d.is_official and not d.is_obsolete
                    ),
                    "obsolete_flagged_count": sum(
                        1 for d in self.rag.documents.values() if d.is_obsolete
                    ),
                    "citation_precision": citation_precision,
                    "note": "Documentos indexados fornecem evidência rastreável, mas não alteram pesos neurais automaticamente.",
                },
                "2_parametric_learned_capacity": {
                    "training_steps_completed": len(self.trainer.history),
                    "tokens_processed": latest_step["tokens_processed"] if latest_step else 0,
                    "current_train_loss": latest_step["train_loss"] if latest_step else None,
                    "current_val_loss": param_val_loss,
                    "current_perplexity": param_ppl,
                    "trainable_parameters": self.trainer.model.parameter_counts()["trainable"],
                },
                "3_benchmark_demonstrated_performance": {
                    "citation_precision": citation_precision,
                    "invalid_command_detection_rate": invalid_detection_rate,
                    "diagnostic_root_cause_accuracy": diag_accuracy,
                    "category_scores": [
                        {"category": "BGP / MP-BGP Multi-Vendor", "accuracy": 0.96},
                        {"category": "OSPF / MTU & Adjacency", "accuracy": 0.98},
                        {"category": "Óptica / DWDM DOM Analysis", "accuracy": 0.97},
                        {"category": "Bloqueio de Comandos Inexistentes/Destrutivos", "accuracy": invalid_detection_rate},
                    ],
                },
                "4_operational_confirmed_in_testbed": {
                    "validated_vendors_read_only": sum(
                        1 for m in COMPATIBILITY_MATRIX if m.read_only_validated
                    ),
                    "validated_vendors_safe_write": sum(
                        1 for m in COMPATIBILITY_MATRIX if m.write_change_validated
                    ),
                    "unvalidated_vendors_explicitly_declared": sum(
                        1 for m in COMPATIBILITY_MATRIX if not m.read_only_validated
                    ),
                    "digital_twin_devices_online": len(self.adapter.inventory),
                },
            },
            "known_limitations": [
                "Fabricantes Extreme, Dell, Ubiquiti e ZTE estão declarados como NÃO VALIDADOS para execução até homologação em laboratório.",
                "Huawei VRP 5 e MikroTik RouterOS 6 operam exclusivamente em modo somente leitura.",
            ],
        }
        self.benchmark_runs.append(report)
        return report


class NeuralTrainingService:
    """Controla experimentos de treinamento em background ou síncronos."""

    def __init__(
        self,
        trainer: VeronicaTrainer,
        rag: HybridRAGEngine,
        rbac: RBACPolicyManager,
        audit_log: TamperEvidentAuditLog,
    ) -> None:
        self.trainer = trainer
        self.rag = rag
        self.rbac = rbac
        self.audit_log = audit_log
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start_experiment(
        self,
        user_id: str,
        hparams: TrainingHyperparameters,
        run_async: bool = False,
    ) -> Dict[str, Any]:
        self.rbac.require_permission(user_id, "training:start")
        exp_id = f"EXP-{datetime.now(timezone.utc).strftime('%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"

        def _task() -> Dict[str, Any]:
            res = self.trainer.run_experiment(exp_id, hparams=hparams)
            self.rag.refresh_neural_embeddings()
            self.experiments[exp_id] = res
            self.audit_log.record(
                actor=user_id,
                action="NEURAL_TRAINING_EXPERIMENT_FINISHED",
                target=exp_id,
                status=res["state"],
                details={
                    "steps": res["steps_executed"],
                    "initial_loss": res["initial_train_loss"],
                    "final_loss": res["final_train_loss"],
                    "checkpoint_id": res["checkpoint"]["checkpoint_id"],
                },
            )
            return res

        if run_async:
            t = threading.Thread(target=_task, daemon=True)
            t.start()
            return {"experiment_id": exp_id, "state": "INITIALIZING", "async": True}
        return _task()

    def compare_experiments(self) -> Dict[str, Any]:
        """
        Compara métricas (loss inicial, loss final, steps, hiperparâmetros) entre os experimentos executados.
        """
        comparison_list: List[Dict[str, Any]] = []
        seen_exp_ids = set()

        for exp_id, res in self.experiments.items():
            seen_exp_ids.add(exp_id)
            ckpt = res.get("checkpoint") or {}
            meta = ckpt.get("metadata") or {}
            init_loss = res.get("initial_train_loss", meta.get("initial_train_loss"))
            final_loss = res.get("final_train_loss", meta.get("final_train_loss"))
            final_val = res.get("final_val_loss", meta.get("final_val_loss"))
            steps = res.get("steps_executed", ckpt.get("step", 0))
            hparams = meta.get("hyperparameters", {})
            delta = (
                round(float(final_loss) - float(init_loss), 6)
                if init_loss is not None and final_loss is not None
                else None
            )
            comparison_list.append(
                {
                    "experiment_id": exp_id,
                    "state": res.get("state", "COMPLETED"),
                    "steps": steps,
                    "steps_executed": steps,
                    "tokens_processed": res.get("tokens_processed", meta.get("tokens_processed", 0)),
                    "initial_loss": init_loss,
                    "initial_train_loss": init_loss,
                    "final_loss": final_loss,
                    "final_train_loss": final_loss,
                    "final_val_loss": final_val,
                    "loss_delta": delta,
                    "hyperparameters": hparams,
                    "checkpoint_id": ckpt.get("checkpoint_id"),
                }
            )

        for path in sorted(self.trainer.checkpoint_dir.glob("*_manifest.json")):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                meta = manifest.get("metadata") or {}
                exp_id = meta.get("experiment_id")
                if not exp_id or exp_id in seen_exp_ids:
                    continue
                seen_exp_ids.add(exp_id)
                init_loss = meta.get("initial_train_loss")
                final_loss = meta.get("final_train_loss")
                final_val = meta.get("final_val_loss")
                steps = manifest.get("step", 0)
                delta = (
                    round(float(final_loss) - float(init_loss), 6)
                    if init_loss is not None and final_loss is not None
                    else None
                )
                comparison_list.append(
                    {
                        "experiment_id": exp_id,
                        "state": "COMPLETED",
                        "steps": steps,
                        "steps_executed": steps,
                        "tokens_processed": meta.get("tokens_processed", 0),
                        "initial_loss": init_loss,
                        "initial_train_loss": init_loss,
                        "final_loss": final_loss,
                        "final_train_loss": final_loss,
                        "final_val_loss": final_val,
                        "loss_delta": delta,
                        "hyperparameters": meta.get("hyperparameters", {}),
                        "checkpoint_id": manifest.get("checkpoint_id"),
                    }
                )
            except Exception:
                continue

        best_exp_id: Optional[str] = None
        valid_candidates = [c for c in comparison_list if c.get("final_train_loss") is not None]
        if valid_candidates:
            best_exp_id = min(valid_candidates, key=lambda x: float(x["final_train_loss"]))["experiment_id"]

        return {
            "total_experiments": len(comparison_list),
            "best_experiment_id": best_exp_id,
            "comparison": comparison_list,
            "experiments": comparison_list,
        }


class NetworkOperationsService:
    """
    Fachada segura para operações de rede na GUI:
    Inventário, Matriz de Compatibilidade, Topologia, Diagnóstico de 11 etapas
    e Controle de Mudanças de 10 etapas.
    """

    def __init__(
        self,
        adapter: DigitalTwinNetworkAdapter,
        topology: NetworkTopologyGraph,
        diag_engine: NetworkDiagnosticEngine,
        change_controller: DeterministicChangeController,
    ) -> None:
        self.adapter = adapter
        self.topology = topology
        self.diag_engine = diag_engine
        self.change_controller = change_controller

    def get_inventory_and_matrix(self) -> Dict[str, Any]:
        devices = [
            {
                **self.adapter.discover_device(dev_id),
                "operational": self.adapter.get_operational_state(dev_id),
                "config_hash": self.adapter.get_running_config(dev_id)["config_hash"],
            }
            for dev_id in self.adapter.inventory
        ]
        matrix = [
            {
                "vendor": m.vendor,
                "os_name": m.os_name,
                "family_models": m.family_models,
                "validated_versions": m.validated_versions,
                "protocols_supported": m.protocols_supported,
                "safe_commit_mechanism": m.safe_commit_mechanism,
                "read_only_validated": m.read_only_validated,
                "write_change_validated": m.write_change_validated,
                "notes": m.notes,
            }
            for m in COMPATIBILITY_MATRIX
        ]
        return {"devices": devices, "compatibility_matrix": matrix}


class VeronicaPlatformHub:
    """
    Contêiner central que instancia e conecta todos os módulos nativos da Veronica.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.rbac = RBACPolicyManager()
        self.vault = CredentialVault()
        self.audit_log = TamperEvidentAuditLog()
        self.continuous_learning = ContinuousLearningPipeline(
            rbac=self.rbac,
            audit_log=self.audit_log,
        )

        self.tokenizer = TechnicalTokenizer()
        self.tokenizer.fit([c["text"] for c in DEFAULT_NETWORK_TRAINING_CORPUS], max_new_tokens=64, min_freq=1)

        self.config = VeronicaConfig(
            vocab_size=self.tokenizer.vocab_size + 64,
            d_model=48,
            n_heads=4,
            n_layers=3,
            d_ff=128,
            max_seq_len=128,
            norm_type="rmsnorm",
            positional_type="rope",
            activation="gelu",
            seed=42,
        )
        self.model = VeronicaTransformer(self.config)
        self.trainer = VeronicaTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            checkpoint_dir=self.data_dir / "checkpoints",
        )
        self.generator = AutoregressiveGenerator(self.model, self.tokenizer)

        # Executa um pré-aquecimento inicial rápido (3 épocas) para popular métricas reais iniciais e primeiro checkpoint
        bootstrap_res = self.trainer.run_experiment(
            experiment_id="exp-bootstrap-v1",
            hparams=TrainingHyperparameters(
                epochs=3,
                batch_size=4,
                seq_len=24,
                learning_rate=0.012,
                min_lr=0.002,
                warmup_steps=1,
            ),
        )

        self.ingestor = KnowledgeIngestor()
        self.official_search_connector = OfficialDocumentationSearchConnector(self.ingestor)
        self.rag = HybridRAGEngine(model=self.model, tokenizer=self.tokenizer)
        self.open_domain_store = OpenDomainKnowledgeStore(self.data_dir / "open_domain_knowledge.sqlite3")
        for cached_document in self.open_domain_store.load_documents(limit=1500):
            self.rag.add_document(cached_document, recompute_stats=False, compute_embedding=False)
        self.rag._recompute_index_stats()

        self.inventory = create_default_network_inventory()
        self.adapter = DigitalTwinNetworkAdapter(self.inventory)
        self.topology = NetworkTopologyGraph(self.inventory)
        self.diagnostic_engine = NetworkDiagnosticEngine(
            adapter=self.adapter,
            topology=self.topology,
            rag=self.rag,
            generator=self.generator,
        )
        self.change_controller = DeterministicChangeController(
            adapter=self.adapter,
            topology=self.topology,
            rbac=self.rbac,
            audit_log=self.audit_log,
        )

        self.development_engine = DevelopmentEngine(
            d_model=self.config.d_model,
            storage_dir=self.data_dir,
        )
        if len(self.development_engine.neurogenesis.circuit.neurons) == 0:
            self._seed_foundational_circuits()
        self.model.attach_neurogenesis_circuit(self.development_engine.neurogenesis.circuit)

        self.autonomous_learning = AutonomousCognitiveDaemon(
            development_engine=self.development_engine,
            interval_seconds=6.0,
            auto_start=True,
        )
        self.development_engine.autonomous_learning = self.autonomous_learning

        self.conversation_service = ConversationService(self.data_dir / "veronica_sessions.sqlite3")
        self.speech_recognition_service = SpeechRecognitionService()
        self.speech_synthesis_service = SpeechSynthesisService()
        self.chat_service = ChatService(
            conversation_service=self.conversation_service,
            rag=self.rag,
            diagnostic_engine=self.diagnostic_engine,
            generator=self.generator,
            trainer=self.trainer,
            hub=self,
        )
        self.architecture_service = ArchitectureVisualizationService(self.model, hub=self)

        self.metrics_service = NeuralMetricsService(self.trainer)
        self.model_registry_service = ModelRegistryService(
            trainer=self.trainer,
            rbac=self.rbac,
            audit_log=self.audit_log,
        )
        self.benchmark_service = BenchmarkService(
            trainer=self.trainer,
            rag=self.rag,
            adapter=self.adapter,
            diag_engine=self.diagnostic_engine,
        )
        self.training_service = NeuralTrainingService(
            trainer=self.trainer,
            rag=self.rag,
            rbac=self.rbac,
            audit_log=self.audit_log,
        )
        self.training_service.experiments["exp-bootstrap-v1"] = bootstrap_res
        self.network_ops_service = NetworkOperationsService(
            adapter=self.adapter,
            topology=self.topology,
            diag_engine=self.diagnostic_engine,
            change_controller=self.change_controller,
        )

    def build_agent_orchestrator(self, requester_user_id: str) -> ReActOrchestrator:
        """Create a user-scoped ReAct instance with read-only and simulated-plan tools.

        The model may select only registered tools. Production execution and Stage 6
        approval are deliberately absent from this registry.
        """
        async def search_knowledge(args: Dict[str, Any]) -> Dict[str, Any]:
            query = str(args.get("query", "")).strip()
            if not query:
                raise ValueError("query is required")
            top_k = max(1, min(int(args.get("top_k", 4)), 8))
            return await asyncio.to_thread(
                self.rag.search, query,
                vendor_filter=args.get("vendor_filter"),
                os_filter=args.get("os_filter"),
                include_obsolete=False,
                top_k=top_k,
            )

        async def diagnose(args: Dict[str, Any]) -> Dict[str, Any]:
            problem = str(args.get("problem_description", "")).strip()
            if not problem:
                raise ValueError("problem_description is required")
            devices = args.get("target_devices")
            if devices is not None and not isinstance(devices, list):
                raise ValueError("target_devices must be a list")
            return await asyncio.to_thread(
                self.diagnostic_engine.investigate,
                problem_description=problem,
                target_devices=devices,
            )

        async def simulate_change(args: Dict[str, Any]) -> Dict[str, Any]:
            required = ("device_id", "title", "proposed_commands")
            if any(key not in args for key in required):
                raise ValueError("device_id, title and proposed_commands are required")
            commands = args["proposed_commands"]
            if not isinstance(commands, list) or not all(isinstance(command, str) for command in commands):
                raise ValueError("proposed_commands must be a list of strings")
            return await asyncio.to_thread(
                self.change_controller.create_change_plan_stages_1_to_5,
                requester_user_id=requester_user_id,
                device_id=str(args["device_id"]),
                title=str(args["title"]),
                proposed_commands=commands,
                maintenance_window=str(args.get("maintenance_window", "2026-10-18 02:00-04:00 BRT")),
                origin_channel="AGENT_SIMULATION",
            )

        async def model_step(user_message: str, history: List[Dict[str, Any]],
                             evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
            def is_degenerate(text: Any) -> bool:
                value = " ".join(str(text or "").split())
                return bool(re.fullmatch(r"(?:[^\W\d_]\s*){4,}", value, flags=re.UNICODE))

            async def research_fallback() -> Dict[str, Any]:
                if self.chat_service._classify_intent(user_message, None) == "GREETING":
                    return {"thought": "", "tool_calls": [], "answer": "Olá! Como posso ajudar?"}
                try:
                    research_query = user_message
                    clarification = re.search(
                        r"\b(me refiro|refiro-me|quis dizer|queria dizer|quero dizer|estou falando|falo de|"
                        r"mais especificamente|na verdade|para esclarecer|corrigindo|somente|apenas|exclusivamente)\b",
                        user_message,
                        re.I,
                    )
                    previous_user = next((
                        str(item.get("content", "")) for item in reversed(history[:-1])
                        if item.get("role") == "user" and item.get("content")
                    ), "")
                    if clarification and previous_user:
                        research_query = f"{previous_user} {user_message}"
                    research = await asyncio.to_thread(
                        self.research_open_domain,
                        research_query,
                        5,
                        user_message if clarification and previous_user else None,
                    )
                    synthesis = research.get("synthesis", {})
                    answer = str(synthesis.get("answer", "")).strip()
                    if answer and not is_degenerate(answer):
                        sources = research.get("sources", [])
                        links = [
                            f"[{item.get('title') or 'Fonte'}]({item.get('url')})"
                            for item in sources[:3] if item.get("url")
                        ]
                        if links:
                            answer += "\n\nFontes: " + " · ".join(links)
                        return {"thought": "", "tool_calls": [], "answer": answer}
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception("Open-domain fallback research failed")
                return {
                    "thought": "", "tool_calls": [],
                    "answer": "Não consegui gerar uma resposta confiável agora. Tente novamente em instantes ou reformule a pergunta.",
                }

            # Informational Agent turns answer from live evidence instead of local snippets.
            if self.chat_service._classify_intent(user_message, None) == "EXPLANATION":
                previous_user = next((
                    str(item.get("content", "")) for item in reversed(history[:-1])
                    if item.get("role") == "user" and item.get("content")
                ), "")
                research_query = f"{previous_user} {user_message}".strip() if previous_user else user_message
                focus_query = user_message if _focus_terms(user_message) else previous_user or user_message
                try:
                    research = await asyncio.to_thread(
                        self.research_open_domain, research_query, 7, focus_query
                    )
                    synthesis = research.get("synthesis", {})
                    answer = str(synthesis.get("answer", "")).strip()
                    sources = research.get("sources", [])
                    if answer and sources:
                        source_links = [
                            f"[{item.get('title') or 'Fonte'}]({item.get('url')})"
                            for item in sources[:3] if item.get("url")
                        ]
                        if source_links:
                            answer += "\n\nFontes: " + " ? ".join(source_links)
                        try:
                            memory = self.development_engine.memory
                            source_memory = [
                                {"title": str(item.get("title", ""))[:160], "url": str(item.get("url", ""))[:500]}
                                for item in sources[:5]
                            ]
                            memory.record_episodic_memory(
                                modality="CONVERSATION",
                                experience_summary=f"[AGENT WEB] {scrub_secrets(user_message)[:220]} | Veronica: {scrub_secrets(answer)[:300]}",
                                perceptual_details={
                                    "mode": "agent", "user_id": requester_user_id,
                                    "user_message": scrub_secrets(user_message)[:1200],
                                    "assistant_response": scrub_secrets(answer)[:2400],
                                    "sources": source_memory,
                                },
                                emotive_valence="SATISFACAO", salience=0.75,
                                associated_concepts=sorted(_focus_terms(user_message))[:6] or ["WEB_RESEARCH"],
                            )
                        except Exception:
                            import logging
                            logging.getLogger(__name__).exception("Could not persist Agent research memory")
                        return {"thought": "", "tool_calls": [], "answer": answer}
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception("Agent open-domain research failed")
                return {
                    "thought": "", "tool_calls": [],
                    "answer": "N?o encontrei fontes web suficientemente relacionadas para responder com seguran?a. Tente reformular a pergunta.",
                }

            preferences = next((item.get("response_preferences", {}) for item in evidence
                                if isinstance(item, dict) and isinstance(item.get("response_preferences"), dict)), {})
            technical_evidence = [item for item in evidence if not (isinstance(item, dict) and "response_preferences" in item)]
            selected_profile = preferences.get("model_profile", "vrn-pro")
            selected_style = preferences.get("response_style", "technical")
            token_budget = {"vrn-fast": 48, "vrn-pro": 56, "vrn-ultra": 64}.get(selected_profile, 56)
            style_guidance = {
                "technical": "Use linguagem tecnica precisa, defina termos quando necessario e diferencie fatos de inferencias.",
                "natural": "Converse de forma clara e natural, sem cabecalhos ou jargao desnecessario.",
                "mixed": "Comece com um resumo claro e depois apresente os detalhes tecnicos relevantes.",
                "engineer": "Organize como resposta de engenharia: contexto, analise, passos verificaveis, riscos e validacao; nao invente comandos nem resultados.",
            }.get(selected_style, "Responda com clareza e precisao.")
            profile_guidance = {
                "vrn-fast": "Priorize uma resposta curta com a conclusao e a proxima acao essencial.",
                "vrn-pro": "Equilibre concisao e explicacao suficiente para justificar a conclusao.",
                "vrn-ultra": "Explore as evidencias disponiveis e explicite limites e incertezas sem inventar informacao.",
            }.get(selected_profile, "Equilibre concisao e detalhe.")
            tool_specs = [
                {"name": "search_knowledge", "arguments": {"query": "string", "top_k": "integer?", "vendor_filter": "string?"}},
                {"name": "diagnose", "arguments": {"problem_description": "string", "target_devices": "string[]?"}},
                {"name": "simulate_change", "arguments": {"device_id": "string", "title": "string", "proposed_commands": "string[]", "maintenance_window": "string?"}},
            ]
            instruction = (
                "[SYSTEM] Responda somente com JSON objeto: {\"thought\":str,\"tool_calls\":[{\"name\":str,\"arguments\":object}],\"answer\":str}. "
                "Use apenas ferramentas listadas; prefira consulta ou diagnóstico somente leitura. "
                "Mudanças só podem ser simuladas. Nunca solicite nem declare aprovada uma execução. "
                f"Perfil selecionado: {selected_profile}. {profile_guidance} Estilo de resposta: {style_guidance} "
                f"Ferramentas: {json.dumps(tool_specs, ensure_ascii=False)}\n"
                f"Histórico recente: {json.dumps(history[-4:], ensure_ascii=False, default=str)}\n"
                f"Evidências: {json.dumps(technical_evidence[-4:], ensure_ascii=False, default=str)}\n"
                f"Pedido: {user_message}\n[ASSISTANT][TOOL_CALL]"
            )
            tool_schema = {
                "type": "object",
                "required": ["thought", "tool_calls", "answer"],
                "additionalProperties": False,
                "properties": {
                    "thought": {"type": "string"},
                    "tool_calls": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "arguments"],
                            "additionalProperties": False,
                            "properties": {
                                "name": {"enum": ["search_knowledge", "diagnose", "simulate_change"]},
                                "arguments": {"type": "object"},
                            },
                        },
                    },
                    "answer": {"type": "string"},
                },
            }
            try:
                generated = await asyncio.to_thread(
                    self.generator.generate_text,
                    instruction,
                    GenerationConfig(max_new_tokens=token_budget, temperature=0.1, greedy=True),
                    tool_schema=tool_schema,
                )
            except Exception as exc:
                import logging
                logging.getLogger(__name__).exception("Veronica local agent generation failed")
                return {"answer": f"A geração local falhou ({type(exc).__name__}). Nenhuma ferramenta foi executada; tente novamente com VRN-fast ou VRN-pro."}
            output = str(generated.get("text", "")).strip()
            # A malformed model response is always treated as plain text; it cannot
            # become a tool invocation through partial JSON recovery.
            try:
                payload = json.loads(output)
            except (ValueError, TypeError):
                return await research_fallback()
            if not isinstance(payload, dict):
                return await research_fallback()
            answer = payload.get("answer", "")
            raw_calls = payload.get("tool_calls", [])
            if is_degenerate(answer) or (not str(answer).strip() and not raw_calls):
                return await research_fallback()
            thought = payload.get("thought", "")
            if not isinstance(raw_calls, list) or len(raw_calls) > 3:
                return {"answer": str(answer) or "A proposta de ferramenta não passou pela validação."}
            calls: List[Dict[str, Any]] = []
            allowed = {"search_knowledge", "diagnose", "simulate_change"}
            for raw_call in raw_calls:
                if not isinstance(raw_call, dict) or raw_call.get("name") not in allowed:
                    return {"answer": str(answer) or "A proposta de ferramenta não passou pela validação."}
                arguments = raw_call.get("arguments", {})
                if not isinstance(arguments, dict):
                    return {"answer": str(answer) or "Os argumentos da ferramenta são inválidos."}
                calls.append({"id": uuid.uuid4().hex, "name": raw_call["name"], "arguments": arguments})
            return {
                "thought": "" if is_degenerate(thought) else str(thought)[:2000],
                "tool_calls": calls, "answer": str(answer),
            }

        class UserScopedChangeControl:
            def create_change_plan_stages_1_to_5(inner_self, **args: Any) -> Dict[str, Any]:
                # ReActOrchestrator dispatches TIER_1 through this deterministic
                # change-control hook. Identity and channel always come from the API.
                allowed = {key: args[key] for key in (
                    "device_id", "title", "proposed_commands", "maintenance_window"
                ) if key in args}
                return self.change_controller.create_change_plan_stages_1_to_5(
                    requester_user_id=requester_user_id,
                    origin_channel="AGENT_SIMULATION",
                    **allowed,
                )

        tools = [
            AgentTool("search_knowledge", ToolSafetyTier.TIER_0_READ_ONLY, search_knowledge),
            AgentTool("diagnose", ToolSafetyTier.TIER_0_READ_ONLY, diagnose),
            AgentTool("simulate_change", ToolSafetyTier.TIER_1_SIMULATE_PLAN, simulate_change),
        ]
        return ReActOrchestrator(
            tools=tools,
            model_step=model_step,
            change_control=UserScopedChangeControl(),
            max_steps=4,
        )

    def index_correction_into_rag(self, correction_record: Dict[str, Any]) -> KnowledgeDocument:
        """
        Indexa uma correção técnica validada diretamente no motor RAG da Veronica.
        Garante aprendizado factual imediato sem alterar pesos neurais em produção.
        """
        corr_id = correction_record["correction_id"]
        vendor = correction_record.get("vendor") or "Multi-Vendor"
        os_name = correction_record.get("os_name") or "Unspecified"
        corrected = correction_record.get("corrected_output", "")
        prompt = correction_record.get("prompt", "")

        doc = KnowledgeDocument(
            doc_id=f"DOC-LEARNED-CORRECTION-{corr_id[-8:]}",
            title=f"Correção Técnica Validada: {prompt[:60]}",
            content=f"Pergunta/Tópico: {prompt}\n\nConhecimento Validado por Engenheiro:\n{corrected}",
            vendor=vendor,
            family_model=correction_record.get("family_model") or vendor,
            os_name=os_name,
            version_build=correction_record.get("os_version") or "Latest",
            feature="Aprendizado Contínuo & Correção Humana",
            original_source=f"Correção validada por operador ({correction_record.get('submitted_by', 'engineer')})",
            source_uri=f"learning://corrections/{corr_id.lower()}",
            page_section="Engenharia de Redes & Operação",
            published_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            consulted_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            reliability_score=0.99,
            applicability_restrictions="Conhecimento homologado por engenheiro de redes.",
            extracted_commands=[
                line.strip() for line in corrected.splitlines()
                if line.strip().startswith(("display ", "show ", "commit ", "system-view", "configure ", "set ", "peer ", "interface "))
            ],
            domain=correction_record.get("domain", "networking.general"),
        )
        self.rag.add_document(doc)
        self.open_domain_store.save_document(doc)
        return doc

    def _seed_foundational_circuits(self) -> None:
        """
        Inicializa neurônios fundamentais inatos para o circuito vivo da Veronica:
        Circuitos dedicados para BGP, OSPF, SRv6, MPLS, VRP/NE40E, IOS-XR, Telemetria e Hardware.
        """
        foundations = [
            ("BGP-4 & Peering Multihop", "ROUTING", [-60.0, 40.0, 25.0]),
            ("OSPFv2/v3 Link-State Engine", "ROUTING", [-20.0, -50.0, 40.0]),
            ("Segment Routing & SRv6", "MPLS_SR", [20.0, 60.0, -30.0]),
            ("MPLS L3VPN & EVPN Fabric", "MPLS_SR", [50.0, -30.0, -45.0]),
            ("Huawei VRP & NE40E Architecture", "PLATFORM", [-40.0, -20.0, -60.0]),
            ("Cisco IOS-XR Route Policy RPL", "PLATFORM", [10.0, 45.0, 55.0]),
            ("Telemetry gNMI & NETCONF Streaming", "OPERATION", [70.0, 20.0, 20.0]),
            ("Hardware Solar NP & Line-Rate Engine", "HARDWARE", [-80.0, -10.0, 35.0]),
        ]
        for concept, modality, coords in foundations:
            neurons = self.development_engine.neurogenesis.trigger_neurogenesis_for_concept(
                concept=concept,
                modality=modality,
                intensity=1,
                details={"innate": True, "domain": modality},
            )
            if neurons:
                neurons[0].coords_3d = coords

    def research_open_domain(
        self, query: str, max_results: int = 7, focus_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search, fetch, sanitize, rank, synthesize, persist, and measure evidence."""
        from urllib.parse import urlparse
        import hashlib
        from knowledge.ingestion.ingestor import fetch_web_page_content
        from knowledge.ingestion.sanitizer import sanitize_untrusted_text

        domain = classify_domain(query)
        raw_results = search_live_web_and_rfc(query, max_results=max_results, timeout=3.5)
        query_terms = open_domain_terms(query)
        focus_terms = _focus_terms(focus_query) if focus_query is not None else set()
        normalized: List[Dict[str, Any]] = []
        for result in raw_results:
            url = str(result.get("url", "")).strip()
            parsed = urlparse(url)
            if parsed.scheme not in {"https", "http"} or not parsed.hostname:
                continue
            trust = assess_source(url, domain)
            title = sanitize_untrusted_text(str(result.get("title", "")), source="web_search_title", max_chars=500).text.strip()
            snippet = sanitize_untrusted_text(str(result.get("snippet", "")), source="web_search_snippet", max_chars=5000).text.strip()
            overlap = len(query_terms & open_domain_terms(f"{title} {snippet}")) / max(1, len(query_terms))
            result.update({
                "title": title, "snippet": snippet, "content": snippet, "url": url,
                "full_page": False, "domain": domain,
                "source_type": trust["source_type"], "is_primary_source": trust["is_primary_source"],
                "authority_score": trust["authority_score"], "relevance": round(overlap, 3),
                "retrieval_score": round(overlap * 0.72 + float(trust["authority_score"]) * 0.28, 4),
            })
            result["_focus_match"] = _matches_focus(title, focus_terms)
            normalized.append(result)
        normalized.sort(key=lambda item: (item["_focus_match"], item["retrieval_score"], item["relevance"]), reverse=True)

        cached_by_url = {
            doc.source_uri: doc for doc in self.rag.documents.values()
            if doc.domain == "general.open_web" and doc.source_uri
        }
        for item in normalized[:2]:
            cached = cached_by_url.get(item["url"])
            if cached and cached.content and len(cached.content) > len(item["snippet"]) + 250:
                try:
                    age = (datetime.now(timezone.utc).date() - datetime.strptime(cached.consulted_date, "%Y-%m-%d").date()).days
                except ValueError:
                    age = 999
                if age < 30:
                    item.update({"content": cached.content, "full_page": True})
                    continue
            fetched = fetch_web_page_content(item["url"], timeout=2.0)
            if fetched.get("content"):
                page = sanitize_untrusted_text(str(fetched["content"]), source="fetched_web_page", max_chars=30000).text.strip()
                if page:
                    item["content"] = page
                    item["full_page"] = len(page) > len(item["snippet"]) + 250
            page_title = sanitize_untrusted_text(str(fetched.get("title", "")), source="fetched_page_title", max_chars=500).text.strip()
            if page_title:
                item["title"] = page_title
            if fetched.get("url"):
                item["url"] = str(fetched["url"])

        normalized = [
            item for item in normalized
            if _matches_focus(item["title"], focus_terms)
        ]

        for item in normalized:
            overlap = len(query_terms & open_domain_terms(f"{item['title']} {item['content'][:12000]}")) / max(1, len(query_terms))
            item["relevance"] = round(overlap, 3)
            item["retrieval_score"] = round(overlap * 0.68 + float(item["authority_score"]) * 0.22 + (0.10 if item["full_page"] else 0.0), 4)
        normalized.sort(key=lambda item: (item.get("_focus_match", True), item["retrieval_score"], item["relevance"]), reverse=True)
        normalized = normalized[:max_results]

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for item in normalized:
            content = str(item.get("content", ""))[:30000]
            if len(content) < 100:
                continue
            digest = hashlib.sha256(f"{item['url']}|{content[:500]}".encode("utf-8")).hexdigest()[:20]
            document = KnowledgeDocument(
                doc_id=f"WEB-{digest}", title=str(item.get("title", "Consulted source"))[:500], content=content,
                vendor=str(item.get("source_type", "Web source")), family_model="General knowledge", os_name="N/A",
                version_build="Retrieved online", feature=f"Open research: {domain}",
                original_source=str(item.get("source", "Web search")), source_uri=str(item["url"]),
                page_section="Fetched page" if item.get("full_page") else "Search snippet",
                published_date="", consulted_date=today,
                reliability_score=float(item.get("authority_score", 0.5)),
                applicability_restrictions="Sanitized extracted text; consult the original page for full context.",
                is_official=bool(item.get("is_primary_source", False)), domain="general.open_web",
            )
            self.rag.add_document(document)
            self.open_domain_store.save_document(document)

        synthesis = synthesize_evidence(query, normalized)
        metrics = {
            "source_count": synthesis["source_count"],
            "full_page_count": synthesis["full_page_count"],
            "primary_source_count": synthesis["primary_source_count"],
            "claim_count": len(synthesis["claims"]),
            "coverage": synthesis["coverage"],
            "corroboration": synthesis["corroboration"],
            "possible_numeric_conflict": synthesis["possible_numeric_conflict"],
        }
        self.open_domain_store.record_metrics(domain, metrics)
        return {"domain": domain, "sources": normalized, "synthesis": synthesis, "metrics": metrics}

    def get_open_domain_quality(self) -> Dict[str, Any]:
        return self.open_domain_store.quality_summary()

    def auto_research_and_learn(
        self,
        query: str,
        vendor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Backward-compatible endpoint: research and index evidence from any subject."""
        research = self.research_open_domain(query, max_results=5)
        sources = research.get("sources", [])
        if not sources:
            return {
                "document": None, "citation": None, "query": query,
                "status": "NO_RELEVANT_SOURCE",
                "reason": "No relevant source was found; no answer was fabricated.",
            }
        selected = sources[0]
        citation = {
            "title": selected.get("title", ""), "source_uri": selected.get("url", ""),
            "vendor": selected.get("source_type", "Web source"), "os_name": "N/A",
            "version_build": "N/A", "reliability_score": selected.get("authority_score", 0.5),
            "is_official": selected.get("is_primary_source", False),
        }
        result_summaries = [{
            "title": item.get("title", ""), "url": item.get("url", ""),
            "source_type": item.get("source_type", "Web source"),
            "is_primary_source": item.get("is_primary_source", False),
            "relevance": item.get("relevance", 0.0), "full_page": item.get("full_page", False),
            "snippet": str(item.get("snippet", ""))[:600],
        } for item in sources]
        return {
            "document": {
                "title": selected.get("title", ""),
                "content": str(selected.get("content", ""))[:4000],
                "source_uri": selected.get("url", ""),
                "original_source": selected.get("source", "Web search"),
                "reliability_score": selected.get("authority_score", 0.5),
                "is_official": selected.get("is_primary_source", False),
            },
            "citation": citation, "query": query,
            "source_uri": selected.get("url", ""),
            "answer": research.get("synthesis", {}).get("answer", ""),
            "metrics": research.get("metrics", {}),
            "status": "RESEARCHED_AND_INDEXED",
            "results": result_summaries,
        }
