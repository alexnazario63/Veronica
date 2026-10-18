# Roadmap Incremental de Desenvolvimento — VERONICA

## Trilho Principal (Engenharia Neural e Redes — Seção 13)

- **Fase 1 — Fundação Matemática (`core/neural`)**:
  - Tensores com Autograd explícito, multiplicação matricial, neurônios parametrizados, camadas (`Linear`, `Embedding`, `LayerNorm`, `RMSNorm`), ativações (`ReLU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`), funções de perda (`MSE`, `CrossEntropy`), otimizadores (`SGD`, `AdamW`), gradient clipping, checkpoints com SHA-256 e verificação numérica de gradientes por diferenças finitas (`gradcheck`).
- **Fase 2 — Modelo Neural Próprio (`core/tokenizer`, `core/transformer`, `core/training`, `core/inference`)**:
  - Tokenizador técnico de redes, embeddings treináveis, RoPE/Positional Encoding, Scaled Dot-Product Attention, Multi-Head Causal Self-Attention com KV-Cache, blocos Transformer, decodificação autoregressiva e loop de treinamento supervisionado/next-token.
- **Fase 3 — Base de Conhecimento (`knowledge/`)**:
  - Ingestão multi-formato com extração de comandos/tabelas, sanitização contra prompt injection, preservação dos 12 metadados obrigatórios, RAG lexical (BM25) + vetorial nativo e detector de documentação obsoleta/conflitante.
- **Fase 4 — Especialização em Redes (`networking/protocols`, `knowledge/sources`)**:
  - Corpus curado e regras determinísticas para Fundamentos (VLAN, STP, LACP, MTU), Roteamento (OSPF, IS-IS, BGP), MPLS/EVPN/VXLAN, Infraestrutura Óptica (DWDM/potência), Segurança e Operação (NETCONF/gNMI/SNMP), além de benchmarks técnicos.
- **Fase 5 — Diagnóstico (`networking/topology`, `networking/diagnostics`)**:
  - Grafo de topologia, coleta operacional somente leitura e motor de investigação em 11 etapas com formulação de hipóteses verificáveis e cálculo de confiança.
- **Fase 6 — Configuração em Laboratório (`networking/vendors`)**:
  - Adaptadores multi-vendor implementando os 16 métodos padronizados, matriz de compatibilidade explícita e simulação de diffs/pré-condições em digital twin de laboratório.
- **Fase 7 — Operação Assistida e Segurança (`safety/`)**:
  - Motor determinístico de mudanças com as 10 etapas obrigatórias, controle de concorrência, RBAC, Cofre de Credenciais, rollback automático e auditoria imutável encadeada por SHA-256.
- **Fase 8 — Evolução Contínua**:
  - Pipeline de aprendizado operacional a partir de incidentes validados e promoção controlada de checkpoints (`TRAINING -> STAGING -> PRODUCTION`).

---

## Trilho de Interface, Multimodalidade e Observabilidade (Seção 16.10)

- **Fase A**: Layout da aplicação web moderna em português brasileiro e chat textual especializado.
- **Fase B**: API conversacional FastAPI e persistência estruturada de sessões e histórico.
- **Fase C**: Integração do chat com o modelo neural próprio Veronica + citações RAG + distinção Fato/Inferência/Hipótese.
- **Fase D**: **Neural Observatory** com visualização gráfica interativa da arquitetura e métricas reais do backend.
- **Fase E**: Laboratório de experimentos, controle de treinamento em tempo real e gestão de checkpoints/benchmarks.
- **Fase F & G**: Reconhecimento (ASR), síntese de voz (TTS), VAD, conversação contínua com interrupção e trava de segurança para comandos falados.
- **Fase H & I**: Integração visual com diagnóstico de redes em 11 etapas, topologia interativa e fluxo de mudanças de 10 etapas com aprovação humana explícita.
