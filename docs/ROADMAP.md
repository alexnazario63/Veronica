# Roteiro Evolutivo (Roadmap) — VERONICA

Este documento apresenta o histórico de fases concluídas e o planejamento incremental da **Veronica**, organizado nos trilhos de **Engenharia Neural, Cognitiva e de Redes** e de **Interface, Multimodalidade e Observabilidade**.

---

## 1. Trilho Principal: Engenharia Neural, Cognitiva e Redes

### Fase 1 — Fundação Matemática e Autograd (`core/neural`) `[CONCLUÍDO]`
- [x] Motor de tensores com grafo computacional dinâmico sobre NumPy.
- [x] Diferenciação automática reversa (backpropagation topológico).
- [x] Camadas parametrizadas (`Linear`, `Embedding`, `LayerNorm`, `RMSNorm`, `Dropout`).
- [x] Ativações (`ReLU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax`).
- [x] Perdas (`CrossEntropyLoss`, `MSELoss`) e Otimizadores (`SGD` com momentum, `AdamW` desacoplado).
- [x] Verificação numérica analítica via diferenças finitas (`gradcheck` com erro $< 10^{-5}$).

### Fase 2 — Modelo Neural Próprio (`core/transformer`) `[CONCLUÍDO]`
- [x] Tokenizador técnico BPE com preservação de átomos de rede (IPv4, IPv6, CIDR, interfaces, CLIs).
- [x] Codificação posicional senoidal e Rotary Positional Embeddings (`RoPE`).
- [x] Multi-Head Causal Self-Attention com KV-Cache incremental (`PrefixKVCache`).
- [x] Arquitetura `VeronicaTransformer` parametrizável em número de camadas e cabeças.
- [x] Decodificador autoregressivo com amostragem top-k, top-p e temperatura.

### Fase 3 — Base de Conhecimento e RAG Híbrido (`knowledge/`) `[CONCLUÍDO]`
- [x] Ingestão multiformato (Markdown, TXT, HTML, JSON, XML).
- [x] Catálogo oficial com **12 metadados obrigatórios** preservados por fragmento.
- [x] RAG Híbrido combinando recuperação léxica BM25S e embeddings neurais próprios.
- [x] Filtros determinísticos: higienizador de credenciais (`SecretScrubber`) e bloqueio de injeção (`PromptInjectionGuard`).

### Fase 4 — Especialização em Redes e Analisadores (`networking/protocols`) `[CONCLUÍDO]`
- [x] Analisadores determinísticos para BGP-4, OSPFv2/v3, IS-IS, MPLS/LDP, EVPN/VXLAN.
- [x] Verificação de saúde de camadas físicas e enlaces ópticos (DOM - Digital Optical Monitoring).
- [x] Suporte a regras de STP (802.1D, RSTP, MSTP) e agregação de enlaces (LACP 802.3ad).

### Fase 5 — Investigação Diagnóstica em 11 Etapas (`networking/diagnostics`) `[CONCLUÍDO]`
- [x] Motor em 11 etapas estritamente somente leitura (*read-only*).
- [x] Grafo de topologia multi-camada com cálculo de raio de impacto (*Blast Radius*).
- [x] Formulação e ranqueamento de hipóteses técnicas com verificação de precedência causal.
- [x] Tratamento formal de incerteza e solicitação pontual de comandos complementares.

### Fase 6 — Adaptadores Multi-Vendor Padronizados (`networking/vendors`) `[CONCLUÍDO]`
- [x] Contrato unificado de **16 métodos padronizados** (`BaseVendorAdapter` e `AsyncVendorAdapter`).
- [x] Suporte multi-vendor homologado: Huawei VRP, Cisco (IOS-XE, IOS-XR, NX-OS), Juniper Junos, MikroTik RouterOS, Nokia SR OS, Arista EOS e Fortinet FortiOS.
- [x] Digital Twin para simulações e geração de diffs canônicos.

### Fase 7 — Controle Determinístico de Mudanças (`safety/change_control`) `[CONCLUÍDO]`
- [x] Fluxo obrigatório de **10 etapas** isolado dos pesos neurais.
- [x] Detecção de concorrência por hash SHA-256 de baseline antes da execução.
- [x] Travas de segurança contra lockout administrativo e exigência de `commit confirmed`.
- [x] Rollback automático determinístico pós-falha e auditoria encadeada imutável (`TamperEvidentAuditLog`).

### Fase 8 — Desenvolvimento Cognitivo e Neurogênese (`core/development`) `[CONCLUÍDO]`
- [x] Maturação por estágios ontogenéticos (`DevelopmentEngine` de `INFANT` a `EXPERT`).
- [x] Neurogênese estrutural baseada em novidade de estímulos (`NeurogenesisEngine` e `DynamicNeuron`).
- [x] Memória de trabalho, memória episódica temporal e grafo semântico de conceitos.
- [x] Armazenamento binário ultra-rápido de vetores de conceitos (`BinaryMachineMemoryStore`).

### Fase 9 — Aprendizado Autônomo em Segundo Plano (`core/development`) `[CONCLUÍDO]`
- [x] Processo daemon multi-worker (`AutonomousCognitiveDaemon`) para tarefas de fundo.
- [x] Ciclo de consolidação cognitiva (sono/sonho) com reorganização de grafos e poda sináptica.
- [x] Quarentena supervisionada de correções humanas (`ContinuousLearningPipeline`).

### Fase 10 — Orquestração de Agente ReAct e Decodificação Restrita (`core/agent`, `core/inference`) `[CONCLUÍDO]`
- [x] Ciclo deliberativo ReAct (*Thought $\rightarrow$ Action $\rightarrow$ Observation $\rightarrow$ Answer*).
- [x] Ferramentas especializadas com contratos de tipagem estrita (`AgentTool`).
- [x] Decodificador JSON guiado por esquemas semânticos (`ConstrainedJSONDecoder`).

### Fase 11 — Aceleração de Hardware Nativa e Drivers Diretos `[CONCLUÍDO]`
- [x] Backends otimizados para operações tensoriais aceleradas (CUDA/ROCm via CuPy e JIT Numba multi-core CPU) mantendo transparência matemática e erro relativo $< 10^{-5}$ no gradcheck.
- [x] Conectores assíncronos diretos para telemetria contínua via gNMI (`pygnmi`), streaming NETCONF com `commit confirmed` (`scrapli_netconf` / `ncclient`) e SSH assíncrono (`scrapli`) nos 16 métodos padronizados com fallback ao Digital Twin.

### Fase 12 — Resolução Multi-Agente e Injeção de Falhas Sintéticas `[CONCLUÍDO]`
- [x] Ecossistema cooperativo de sub-agentes Veronica especializados por protocolo (`BGPProtocolAgent`, `MPLSAndTrafficAgent`, `OpticalAndL1Agent`, `SecurityAndACLAgent`) coordenados pelo `ReActOrchestrator`.
- [x] Integração cognitiva contínua no loop agêntico (`MemorySystem.query_concepts`, `BinaryMachineMemoryStore.fast_knn_lookup` $< 2.0\text{ ms}$, `NeurogenesisEngine` com $\tau = 0.75$, `record_validated_incident`).
- [x] Gerador automático de falhas sintéticas (`FaultInjector`) em topologias virtuais nos 5 cenários críticos (DOM, LACP, BGP leak, MTU mismatch, partição OSPF/IS-IS) com cálculo exato de *Blast Radius*.

---

## 2. Trilho de Interface, Multimodalidade e Observabilidade

- **Fase A — Layout Web Responsivo e Chat Técnico** `[CONCLUÍDO]`: Interface moderna em React e TypeScript focada em engenharia de telecomunicações.
- **Fase B — Camada REST FastAPI e Streaming** `[CONCLUÍDO]`: Gestão de sessões conversacionais com persistência e streaming via SSE/WebSockets.
- **Fase C — Epistemologia Visual Transparente** `[CONCLUÍDO]`: Diferenciação clara entre Fatos Comprovados (`VERIFIED_FACT`), Inferências Técnicas (`TECHNICAL_INFERENCE`) e Hipóteses (`HYPOTHESIS`).
- **Fase D — Neural Observatory em Tempo Real** `[CONCLUÍDO]`: Inspeção gráfica da arquitetura, camadas, neurônios, normas de gradientes e telemetria de hardware (CPU/RAM/GPU/VRAM).
- **Fase E — Laboratório de Experimentos e Checkpoints** `[CONCLUÍDO]`: Controle de treinamento e esteira controlada de promoção (`TRAINING -> STAGING -> PRODUCTION`).
- **Fase F & G — Conversação por Voz Multimodal (ASR / TTS)** `[CONCLUÍDO]`: Reconhecimento de fala com normalização de termos técnicos, síntese e trava estrita contra comandos destrutivos por voz.
- **Fase H — NOC Center e Visualizador Topológico** `[CONCLUÍDO]`: Renderização de nós de rede, inspeção de Digital Twin e acompanhamento visual do diagnóstico em 11 etapas.
- **Fase I — Admin Cockpit e Grafo de Memória** `[CONCLUÍDO]`: Painel administrativo com visualização interativa dos nós conceituais e métricas de qualidade de pesquisa.
- **Fase J — Workspace de Artefatos Versionados** `[CONCLUÍDO]`: Visualização de documentos gerados, histórico de alterações e aplicação de propostas de diff.
- **Fase K — Gestão Multi-Tenant e Integração OIDC / SSO** `[CONCLUÍDO]`: Suporte a múltiplos clusters de rede corporativos isolados por `tenant_id`, autenticação via tokens JWT/OIDC e asserções SAML 2.0, e auditoria criptográfica contínua no `TamperEvidentAuditLog`.
- **Experiência Unificada Chat + Agent + Work (NetCanvas / Artifacts)** `[CONCLUÍDO]`: Interface split-screen dinâmica integrando ChatUI com Omnibox inteligente e chips compactos de logs, acordeão de raciocínio multi-agente (*Thinking & Tool Trace*), badges epistemológicas interativas, WorkUI / NetCanvas lateral com editor diff interativo, seleção parcial de linhas com co-edição no chat, console de 10 etapas de mudança com grafo Cytoscape de *Blast Radius*, modal de assinatura humana com bloqueio de voz (ADR-03 / R-09) e gaveta rápida de observabilidade/ontogênese.
