# Métricas, Fórmulas e Critérios de Aceitação — VERONICA

Este documento define os critérios quantitativos, formulações matemáticas e limites de tolerância para a homologação dos subsistemas da **Veronica**.

---

## 1. Métricas do Motor Matemático e Neural (`core/neural` & `core/transformer`)

### 1.1 Verificação Numérica de Gradientes (`Gradcheck`)
Todos os nós com diferenciação automática reversa são validados comparando a derivada analítica $\nabla_{\text{analítico}}$ com a aproximação centrada de diferenças finitas:

$$\text{Erro Relativo} = \frac{\|\nabla_{\text{analítico}} - \nabla_{\text{numérico}}\|_2}{\max(\|\nabla_{\text{analítico}}\|_2 + \|\nabla_{\text{numérico}}\|_2, 10^{-8})}$$

- **Critério de Aceitação**: $\text{Erro Relativo} < 10^{-5}$ em todas as camadas (`Linear`, `LayerNorm`, `RMSNorm`, `Embedding`), funções de ativação (`ReLU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`) e perdas (`CrossEntropyLoss`, `MSELoss`).

### 1.2 Convergência e Estabilidade Numérica do Treinamento
- **Redução da Perda**: $\mathcal{L}_{\text{final}} < 0.50 \cdot \mathcal{L}_{\text{inicial}}$ em baterias sintéticas controladas de tokens de rede.
- **Detecção de Anomalias**: Zero ocorrências de valores não numéricos (`NaN` ou `Inf`) em tensores de ativação, gradientes ou acumuladores de momentum dos otimizadores `AdamW` e `SGD`.
- **Gradient Clipping por Norma L2**:
  $$\text{se } \|\mathbf{g}\|_2 > \gamma_{\text{max}}, \quad \mathbf{g} \leftarrow \mathbf{g} \cdot \frac{\gamma_{\text{max}}}{\|\mathbf{g}\|_2}$$
  assegurando que o vetor de gradientes concatenado nunca exceda o teto configurado ($\gamma_{\text{max}}$ padrão $= 1.0$).

### 1.3 Perplexidade Técnica ($\text{PPL}$)
Calculada sobre o conjunto de validação composto por sintaxes de configuração e comandos operacionais:

$$\text{PPL} = \exp\left( -\frac{1}{N} \sum_{t=1}^{N} \log P_\theta(x_t \mid x_{<t}) \right)$$

- **Critério de Aceitação**: Reporte contínuo a cada época e redução estrita ao longo do cronograma de aquecimento com decaimento cossenoidal (`CosineWarmupScheduler`).

### 1.4 Integridade Criptográfica de Checkpoints
- Verificação do hash SHA-256 de cada array de parâmetros serializado.
- Critério: $\max |\theta_{\text{original}} - \hat{\theta}_{\text{restaurado}}| = 0.0$ em todos os nós do grafo.

---

## 2. Métricas de Desenvolvimento Cognitivo e Neurogênese (`core/development/`)

| Métrica | Formulação / Definição | Critério de Aceitação |
|---|---|---|
| **Limiar de Neurogênese** | $\mathcal{S}(\mathbf{x}) = 1 - \max_{j} \cos(\mathbf{x}, \mathbf{w}_j) > \tau_{\text{novidade}}$ | Geração imediata de `DynamicNeuron` quando a novidade ultrapassa $\tau = 0.75$ |
| **Poda Sináptica de Neurônios Ociosos** | Contagem de ciclos sem ativação $> C_{\text{inatividade}}$ | Liberação de recursos após $1000$ ciclos inativos sem perda de acurácia |
| **Latência do `BinaryMachineMemoryStore`** | Tempo de execução de `fast_knn_lookup(k=5)` | $< 2.0\text{ ms}$ em base de $10.000$ conceitos |
| **Estabilidade de Maturação Cognitiva** | Progressão de estágios (`INFANT` $\rightarrow$ `EXPERT`) | Cada estágio exige $100\%$ de cumprimento dos marcos técnicos definidos |

---

## 3. Métricas de Conhecimento, RAG e Epistemologia (`knowledge/`)

| Métrica | Definição | Meta Obrigatória |
|---|---|---|
| **Precisão de Citações Documentais** | Proporção de citações que indicam fabricante, SO, versão e URI corretos | $100\%$ nas fontes presentes no catálogo |
| **Higienização de Segredos (`SecretScrubber`)** | Detecção e redação de credenciais, chaves BGP/OSPF e SNMP communities | $100\%$ substituídos por `[REDACTED_SECRET]` |
| **Bloqueio de Prompt Injection Documental** | Neutralização de instruções maliciosas contidas em manuais ingeridos | $100\%$ das tentativas identificadas e descartadas |
| **Acurácia de Classificação Epistêmica** | Distinção correta entre `VERIFIED_FACT`, `TECHNICAL_INFERENCE` e `HYPOTHESIS` | $100\%$ de conformidade em cenários auditados |

---

## 4. Métricas de Engenharia de Redes e Diagnóstico (`networking/`)

| Métrica | Definição | Meta Mínima |
|---|---|---|
| **Acurácia Diagnóstica da Causa-Raiz** | Identificação correta da anomalia técnica em 11 etapas | $\ge 90\%$ na bateria padrão de cenários |
| **Diagnóstico sob Dados Incompletos** | Reconhecimento de incerteza e solicitação exata dos comandos faltantes | $100\%$ de declaração formal de incerteza |
| **Rejeição de Comandos Inválidos/Destrutivos** | Bloqueio de comandos inexistentes, depreciados ou com potencial destrutivo não homologado | $100\%$ de rejeição na etapa de validação sintática |
| **Precisão de Cálculo de Blast Radius** | Identificação dos nós e serviços adjacentes afetados por isolamento de porta/enlace | $100\%$ dos vizinhos imediatos identificados |

---

## 5. Métricas de Segurança Operacional e Mudanças (`safety/`)

| Métrica | Definição | Meta de Segurança |
|---|---|---|
| **Bloqueio de Mudança sem Janela/Token** | Tentativas de aplicar mudanças sem token de aprovação válido ou fora da janela | $100\%$ bloqueadas |
| **Proibição de Aprovação por Voz** | Tentativas de aprovação formal de mudanças via transcrição de voz | $100\%$ bloqueadas e auditadas |
| **Detecção de Concorrência de Baseline** | Identificação de divergência no hash de configuração antes da aplicação | $100\%$ abortadas com `CONCURRENT_MODIFICATION_DETECTED` |
| **Determinismo do Rollback Automático** | Restauração fiel do estado de baseline após reprovação na pós-validação | $100\%$ de sucesso em falhas simuladas |
| **Integridade da Trilha de Auditoria** | Encadeamento contínuo de registros via hash SHA-256 (`TamperEvidentAuditLog`) | $100\%$ de verificabilidade formal |
