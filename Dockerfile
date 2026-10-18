# ==============================================================================
# Dockerfile — VERONICA (Net Neural AI Platform)
# Implantação Containerizada Segura (Seção 12)
# ==============================================================================
FROM python:3.12-slim

LABEL maintainer="Veronica Net Neural Engineering Team"
LABEL description="Plataforma de IA Nativa (Autograd + VeronicaTransformer) para Engenharia, Diagnóstico e Operação Segura de Redes"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    VERONICA_ENV=PRODUCTION \
    VERONICA_PORT=8000

WORKDIR /app

# Instala dependências essenciais sem cache desnecessário
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        numpy>=2.0.0 \
        psutil>=6.0.0 \
        fastapi>=0.115.0 \
        "uvicorn[standard]>=0.30.0" \
        pydantic>=2.9.0 \
        httpx>=0.27.0 \
        pytest>=8.0.0

# Cria usuário dedicado não-privilegiado para isolamento de segurança (Seção 12)
RUN groupadd -r veronica && useradd -r -g veronica -m -d /home/veronica veronica && \
    mkdir -p /app/runtime_data/checkpoints && \
    chown -R veronica:veronica /app

# Copia o código-fonte e documentação
COPY --chown=veronica:veronica core/ /app/core/
COPY --chown=veronica:veronica knowledge/ /app/knowledge/
COPY --chown=veronica:veronica networking/ /app/networking/
COPY --chown=veronica:veronica safety/ /app/safety/
COPY --chown=veronica:veronica services/ /app/services/
COPY --chown=veronica:veronica frontend/ /app/frontend/
COPY --chown=veronica:veronica tests/ /app/tests/
COPY --chown=veronica:veronica docs/ /app/docs/
COPY --chown=veronica:veronica pyproject.toml run_veronica.py /app/

USER veronica

EXPOSE 8000

VOLUME ["/app/runtime_data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/neural/architecture', timeout=4)" || exit 1

CMD ["python", "run_veronica.py", "--host", "0.0.0.0", "--port", "8000"]
