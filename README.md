# VERONICA — Inteligência Artificial Nativa para Engenharia, Diagnóstico e Operação Segura de Redes

**Veronica** é uma plataforma completa de Inteligência Artificial construída desde os fundamentos matemáticos e neurais, especializada em engenharia, administração, troubleshooting estruturado e automação segura de redes de computadores multi-vendor.

Diferente de soluções baseadas em meros wrappers de APIs externas ou pesos comerciais de caixa-preta, a **Veronica** implementa seu próprio motor matemático de diferenciação automática (**Autograd sobre NumPy**), sua própria arquitetura neural (**`VeronicaTransformer`**), um sistema híbrido de recuperação documental (**RAG lexical BM25 + embeddings neurais próprios** com 12 metadados obrigatórios), adaptadores padronizados multi-vendor (**16 métodos**), diagnóstico estruturado em **11 etapas**, motor determinístico de mudanças seguras em **10 etapas** e uma interface gráfica web moderna com **Chat Técnico, Conversação por Voz e Neural Observatory (Neural Lab)**.

---

## 1. Pilares Arquiteturais

1. **Fundação Matemática e Neural Própria (`core/`)**:
   - Motor de tensores com grafo computacional dinâmico e backpropagation analítico validado por diferenças finitas (`gradcheck` com erro relativo $< 10^{-5}$).
   - Camadas parametrizadas (`Linear`, `Embedding`, `LayerNorm`, `RMSNorm`, `Dropout`), ativações (`ReLU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`), funções de perda (`MSELoss`, `CrossEntropyLoss`) e otimizadores (`SGD` com momentum, `AdamW` com weight decay desacoplado e `CosineWarmupScheduler`).
   - Tokenizador técnico (`TechnicalTokenizer`) que preserva átomos críticos de redes: endereços IPv4/IPv6, prefixos CIDR, ASNs, interfaces (`GigabitEthernet0/0/0`, `TenGigE0/0/0/1`, `10GE1/0/1`), VLANs, VRFs e comandos CLI.
   - Modelo **`VeronicaTransformer`** com RoPE (Rotary Positional Embeddings) ou codificação senoidal, Multi-Head Causal Self-Attention com KV-Cache e geração autoregressiva auditável.

2. **Conhecimento Híbrido com Epistemologia Transparente (`knowledge/`)**:
   - Separação estrita entre **conhecimento paramétrico** (aprendido nos pesos do modelo) e **conhecimento documental** (indexado com rastreabilidade exata).
   - Preservação obrigatória de **12 campos de metadados** por documento (`vendor`, `family_model`, `os_name`, `version_build`, `feature`, `original_source`, `source_uri`, `page_section`, `published_date`, `consulted_date`, `reliability_score`, `applicability_restrictions`).
   - Sanitização automática de credenciais (`scrub_secrets`) e bloqueio determinístico contra *Prompt Injection* documental.
   - Classificação explícita de toda afirmação em:
     - **Fato Comprovado (`VERIFIED_FACT`)**
     - **Inferência Técnica (`TECHNICAL_INFERENCE`)**
     - **Hipótese Verificável (`HYPOTHESIS`)**

3. **Engenharia Multi-Vendor e Diagnóstico em 11 Etapas (`networking/`)**:
   - Interface padronizada `BaseVendorAdapter` com **16 métodos obrigatórios** e Matriz de Compatibilidade explícita para **Huawei VRP (v8/v5)**, **Cisco (IOS-XE, IOS-XR, NX-OS)**, **Juniper Junos**, **MikroTik RouterOS (v7/v6)**, **Nokia SR OS**, **Arista EOS** e **Fortinet FortiOS** (além de declaração explícita de fabricantes não validados para escrita).
   - Grafo de topologia multi-camada (L1 Óptica/DWDM, L2 VLAN/STP/LACP, L3 OSPF/IS-IS/BGP, MPLS/EVPN) com cálculo de raio de impacto (*Blast Radius*).
   - Motor de investigação diagnóstica somente leitura em **11 etapas**, capaz de operar com telemetria completa ou declarar incertezas quando os dados são incompletos.

4. **Segurança Determinística e Fluxo de Mudança de 10 Etapas (`safety/`)**:
   - Controlador de mudanças 100% determinístico e independente do modelo neural:
     1. Descoberta e Identificação Exata
     2. Coleta de Baseline (hash SHA-256)
     3. Análise de Impacto e Risco de Lockout Administrativo
     4. Planejamento e Pré-condições
     5. Geração e Validação Sintática de Candidato + Diff
     6. Aprovação Humana Explícita (vinculada ao hash do plano e janela de manutenção; **proibida por voz**)
     7. Execução Controlada com Verificação de Concorrência
     8. Pós-validação Automática Imediata
     9. Rollback Automático Determinístico em Caso de Degradação
     10. Auditoria Imutável Encadeada por SHA-256
   - RBAC com separação de papéis (`VIEWER`, `NOC_OPERATOR`, `NETWORK_ENGINEER`, `CHANGE_APPROVER`, `AI_TRAINER`, `ADMIN`) e Cofre de Credenciais (`CredentialVault`).

5. **Aprendizado Contínuo Curado (`core/training/continuous_learning.py`)**:
   - Registro de incidentes operacionais validados com rollback testado (`record_validated_incident`).
   - Submissão de correções humanas em quarentena (`PENDING_VERIFICATION`) que só ingressam no dataset versionado após aprovação por engenheiro autorizado (`verify_and_approve_correction`).
   - Garantia determinística de que conteúdos externos jamais alteram políticas de segurança nem atualizam pesos em produção automaticamente.

---

## 2. Estrutura do Repositório

```text
Veronica/
├── core/
│   ├── neural/                      # Tensores com Autograd, Camadas, Ativações, Losses, Otimizadores, Clipping, Checkpoints, Gradcheck
│   ├── tokenizer/                   # Tokenizador técnico BPE + átomos de engenharia de redes (IPv4/IPv6/CIDR/CLI)
│   ├── transformer/                 # VeronicaConfig, RoPE/Positional, Multi-Head Causal Self-Attention, VeronicaTransformer
│   ├── training/                    # VeronicaTrainer, Telemetria de Hardware (psutil) e ContinuousLearningPipeline
│   └── inference/                   # Gerador autoregressivo com KV-Cache, Top-k/Top-p e cancelamento limpo
├── knowledge/
│   ├── ingestion/                   # Ingestão multi-formato, extração de CLI/tabelas, SecretScrubber e PromptInjectionGuard
│   ├── retrieval/                   # Motor RAG Híbrido (BM25 + Embeddings Neurais da Veronica) e detector de conflitos/obsolescência
│   └── sources/                     # Catálogo oficial multi-vendor com os 12 metadados obrigatórios
├── networking/
│   ├── vendors/                     # BaseVendorAdapter (16 métodos), CompatibilityMatrix e DigitalTwinNetworkAdapter
│   ├── protocols/                   # Analisadores determinísticos (BGP, OSPF, IS-IS, MPLS/EVPN, STP/LACP, Óptica DOM)
│   ├── topology/                    # Grafo de topologia e cálculo de Blast Radius
│   └── diagnostics/                 # Motor de diagnóstico estruturado em 11 etapas (somente leitura)
├── safety/
│   ├── policies/                    # RBACPolicyManager, CredentialVault e TamperEvidentAuditLog (cadeia SHA-256)
│   ├── change_control/              # DeterministicChangeController (Fluxo obrigatório de 10 etapas)
│   └── rollback/                    # RollbackOrchestrator
├── services/
│   └── api/                         # Aplicação FastAPI (app.py) e os 10 Serviços Modulares (services_hub.py)
├── frontend/
│   ├── index.html                   # Interface Web completa (Chat Técnico, Voz, Neural Observatory, NOC Center)
│   └── src/                         # Componentes e tipagens TypeScript da GUI
├── tests/
│   ├── test_phase1_neural_math.py
│   ├── test_phase2_transformer_and_training.py
│   ├── test_phase3_to_7_knowledge_networking_safety.py
│   ├── test_phase10_continuous_learning.py
│   └── test_phase16_api_gui_and_multimodal.py
├── docs/                            # Documentação detalhada (Arquitetura, Datasets, Interfaces, Métricas, Riscos, Roadmap)
├── run_veronica.py                  # Ponto de entrada executável (Servidor Web Uvicorn ou Diagnóstico CLI)
├── Dockerfile                       # Imagem de contêiner endurecida (usuário não-root + healthcheck)
├── docker-compose.yml               # Orquestração containerizada com volume persistente
└── pyproject.toml                   # Configuração do projeto e do pytest
```

---

## 3. Como Executar os Testes Automatizados

A suíte de testes valida rigorosamente:
- Gradientes analíticos vs numéricos por diferenças finitas (`gradcheck`), convergência real de perda, perplexidade e integridade SHA-256 de checkpoints;
- Tokenização de entidades de rede, máscara causal e geração autoregressiva do `VeronicaTransformer`;
- Precisão dos 12 metadados de citação no RAG, bloqueio de *Prompt Injection*, redator de segredos e detecção de comandos inválidos/destrutivos;
- Diagnóstico em 11 etapas (com dados completos e incompletos), fluxo seguro de mudança em 10 etapas, bloqueio de aprovação por voz, detecção de concorrência e rollback automático;
- Pipeline de aprendizado contínuo (`ContinuousLearningPipeline`) e todos os endpoints da API/GUI.

Para executar todos os testes no ambiente virtual do projeto:

```powershell
# No Windows PowerShell (usando o ambiente virtual .venv local)
.\.venv\Scripts\pytest.exe -v

# Ou em ambientes Linux/macOS
pytest -v
```

Para executar um módulo de teste específico:

```powershell
.\.venv\Scripts\pytest.exe tests/test_phase1_neural_math.py -v
.\.venv\Scripts\pytest.exe tests/test_phase10_continuous_learning.py -v
```

---

## 4. Como Iniciar a Interface Gráfica Web + Neural Observatory

### 4.1 Execução Direta Local (Python / Uvicorn)

```powershell
.\.venv\Scripts\python.exe run_veronica.py --host 0.0.0.0 --port 8000
```

Após iniciar, acesse no navegador:
- **Interface Gráfica Completa (Chat Técnico + Voz + Neural Observatory + NOC Center)**: [http://localhost:8000](http://localhost:8000)
- **Documentação Interativa da API (Swagger / OpenAPI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

Na interface web você encontrará as abas integradas:
1. **Chat Técnico & Voz Multimodal**: Conversação em português brasileiro com separação visual entre Fatos Comprovados (`VERIFIED_FACT`), Inferências Técnicas (`TECHNICAL_INFERENCE`) e Hipóteses (`HYPOTHESIS`), citações expansíveis com os 12 metadados, blocos de comandos seguros e controles de microfone/TTS com consentimento explícito.
2. **Neural Observatory (Neural Lab)**: Visualização gráfica interativa da arquitetura `VeronicaTransformer` (camadas, cabeças de atenção, dimensões, parâmetros treináveis/congelados), curvas reais de *Training Loss*, *Validation Loss*, *Perplexity*, normas de gradientes por camada, telemetria real de CPU/RAM/Disco (com status explícito de GPU) e painel das **4 Dimensões Epistêmicas de Conhecimento**.
3. **Laboratório de Experimentos & Checkpoints**: Configuração de hiperparâmetros (`epochs`, `batch_size`, `learning_rate`, `optimizer`, `max_grad_norm`), disparo/pausa/retomada/cancelamento de treinamentos reais e promoção controlada de checkpoints (`TRAINING -> STAGING -> PRODUCTION`).
4. **NOC Center (Inventário, Topologia, Diagnóstico 11 Etapas & Mudanças 10 Etapas)**: Inspeção do Digital Twin multi-vendor, matriz de compatibilidade e aprovação humana explícita de planos de mudança.

### 4.2 Diagnóstico Rápido via Linha de Comando (CLI)

Você também pode executar uma investigação diagnóstica somente leitura diretamente pelo terminal:

```powershell
.\.venv\Scripts\python.exe run_veronica.py --diagnose "Sessão BGP em estado Active no NE8000-CORE-01 e OSPF preso em EXSTART no ASR9K-CORE-02" --devices NE8000-CORE-01 ASR9K-CORE-02
```

### 4.3 Execução Containerizada (Docker & Docker Compose)

```bash
docker compose up --build -d
```

O contêiner roda sob usuário não-privilegiado (`veronica`), persiste checkpoints e sessões no volume `veronica_runtime_data` e expõe a plataforma em `http://localhost:8000`.

---

## 5. Documentação Complementar

Consulte a pasta [`docs/`](./docs/) para especificações detalhadas:
- [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) — Arquitetura em camadas e decisões de projeto
- [`docs/INTERFACES.md`](./docs/INTERFACES.md) — Contratos entre módulos, os 16 métodos de adaptadores e os 10 serviços da aplicação
- [`docs/DATASETS.md`](./docs/DATASETS.md) — Esquemas dos 12 metadados documentais, datasets JSONL e incidentes validados
- [`docs/METRICS_AND_ACCEPTANCE.md`](./docs/METRICS_AND_ACCEPTANCE.md) — Fórmulas matemáticas e critérios de aceitação
- [`docs/RISKS_AND_LIMITATIONS.md`](./docs/RISKS_AND_LIMITATIONS.md) — Limitações conhecidas e matriz de mitigação de riscos
- [`docs/ROADMAP.md`](./docs/ROADMAP.md) — Roadmap incremental dos trilhos Principal e de Interface/Observabilidade
