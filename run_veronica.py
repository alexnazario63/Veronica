#!/usr/bin/env python3
"""
Script Principal de Execução da Plataforma VERONICA (Net Neural AI).
Permite:
1. Iniciar o servidor web FastAPI + Uvicorn (GUI Multimodal + Neural Observatory) na porta 8000:
   python run_veronica.py
   python run_veronica.py --host 0.0.0.0 --port 8000

2. Executar diagnóstico rápido de rede em modo CLI (somente leitura, 11 etapas + RAG + VeronicaTransformer):
   python run_veronica.py --diagnose "Sessão BGP em estado Active no NE8000-CORE-01"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional

import uvicorn


def run_cli_diagnostic(query: str, devices: Optional[List[str]] = None) -> int:
    """Executa o fluxo de investigação em 11 etapas e inferência neural via linha de comando."""
    from services.api.services_hub import VeronicaPlatformHub

    runtime_dir = Path(__file__).resolve().parent / "runtime_data"
    hub = VeronicaPlatformHub(runtime_dir)

    report = hub.diagnostic_engine.investigate(
        problem_description=query,
        target_devices=devices,
    )
    conv = hub.conversation_service.create_conversation(
        user_id="usr-eng-01",
        title=f"Diagnóstico CLI: {query[:40]}",
        model_version=hub.trainer.model_version,
    )
    reply = hub.chat_service.handle_message(
        conversation_id=conv["conversation_id"],
        user_id="usr-eng-01",
        message_text=query,
        modality="TEXT",
        target_devices=devices,
    )

    output = {
        "agent": "Veronica",
        "model_version": hub.trainer.model_version,
        "read_only_guarantee": report["read_only_guarantee"],
        "confidence": report["confidence"],
        "root_causes": report["root_causes"],
        "epistemic_breakdown": report["epistemic_breakdown"],
        "safe_diagnostic_commands": report["safe_diagnostic_commands"],
        "proposed_remediation_commands": report["proposed_remediation_commands"],
        "neural_generation": reply["structured_payload"].get("neural_generation", {}),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VERONICA — Inteligência Artificial Nativa para Engenharia e Operação Segura de Redes"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Endereço IP de bind do servidor web (padrão: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Porta TCP do servidor Uvicorn (padrão: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Habilita auto-reload para desenvolvimento",
    )
    parser.add_argument(
        "--diagnose",
        type=str,
        default=None,
        help="Executa um diagnóstico técnico imediato via CLI em modo somente leitura e encerra",
    )
    parser.add_argument(
        "--devices",
        nargs="*",
        default=None,
        help="Lista opcional de equipamentos alvo para o modo --diagnose (ex.: NE8000-CORE-01 ASR9K-CORE-02)",
    )

    args = parser.parse_args()

    if args.diagnose:
        return run_cli_diagnostic(args.diagnose, args.devices)

    print("=" * 78)
    print("  VERONICA — Inteligência Artificial Nativa para Engenharia de Redes")
    print(f"  Interface Gráfica Web & Neural Observatory: http://localhost:{args.port}")
    print(f"  Documentação OpenAPI / Swagger:             http://localhost:{args.port}/docs")
    print("=" * 78)

    uvicorn.run(
        "services.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
