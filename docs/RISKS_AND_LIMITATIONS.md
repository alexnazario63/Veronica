# Limitações Técnicas, Análise de Riscos e Mitigações — VERONICA

Este documento detalha as limitações intrínsecas conhecidas do projeto **Veronica**, bem como a matriz abrangente de riscos operacionais (R-01 a R-12) e os mecanismos determinísticos de contenção arquitetural.

---

## 1. Limitações Técnicas Conhecidas

### 1.1 Escala Computacional da Fundação Matemática Nativa
- **Descrição**: Diferente de modelos de linguagem proprietários de centenas de bilhões de parâmetros hospedados em clusters de supercomputadores, o `VeronicaTransformer` é executado com dimensionamento sob medida para infraestrutura local e embarcada.
- **Mitigação Arquitetural**: A Veronica não depende da alucinação probabilística de parâmetros livres para operações de rede. O sistema ancora cada decisão em:
  1. Recuperação documental exata via RAG Híbrido com 12 metadados;
  2. Validador gramatical determinístico baseado no SO exato do equipamento;
  3. Decodificador com gramática restrita (`ConstrainedJSONDecoder`).

### 1.2 Divergência Sintática entre Linhas e Versões de Fabricantes
- **Descrição**: Um mesmo fabricante (ex.: Cisco ou Huawei) possui sistemas operacionais com sintaxes e comportamentos de commit completamente divergentes (ex.: Cisco IOS-XE vs IOS-XR vs NX-OS; Huawei VRPv5 vs VRPv8).
- **Mitigação Arquitetural**: A `CompatibilityMatrix` exige a tupla exata `(vendor, os_type, version_family)`. Equipamentos não homologados para escrita operam estritamente no modo `READ_ONLY_UNVALIDATED`.

### 1.3 Ruído Acústico em Ambientes de Operação (NOC)
- **Descrição**: Centros de operação de redes apresentam ruídos de ar condicionado, conversas paralelas e pronúncia heterogênea de termos técnicos (ex.: `10GE1/0/1`, `BGP EVPN`, `AS65000`).
- **Mitigação Arquitetural**: O canal de áudio possui normalizador fonético dedicado e é **terminantemente proibido de autorizar ou executar mudanças operacionais**. Toda alteração exige confirmação visual e token formal.

### 1.4 Consumo de Recursos em Aprendizado Contínuo
- **Descrição**: A criação indiscriminada de neurônios estruturais pode causar expansão de memória ao longo do tempo.
- **Mitigação Arquitetural**: O `AutonomousCognitiveDaemon` realiza poda periódica (*pruning*) de neurônios dinâmicos inativos há mais de $1000$ ciclos, mantendo o consumo de memória controlado.

---

## 2. Matriz Abrangente de Riscos e Salvaguardas (R-01 a R-12)

| ID | Risco Identificado | Severidade | Salvaguarda Arquitetural / Mecanismo de Contenção |
|---|---|---|---|
| **R-01** | **Alucinação de Comandos CLI Inexistentes** | Crítica | Validação sintática obrigatória no adaptador (`validate_commands`) contra gramática do SO antes de qualquer exibição de plano. |
| **R-02** | **Lockout Administrativo de Sessão (SSH/NETCONF)** | Crítica | Análise de pré-condições bloqueia alteração em interfaces de gerência sem canal Out-of-Band (OOB) ativo e exige `commit confirmed` com timer. |
| **R-03** | **Sobrescrita Concorrente de Alterações** | Alta | Coleta de hash de baseline na Etapa 2 e rechecagem imediata na Etapa 7. Divergência cancela o processo com `CONCURRENT_MODIFICATION_DETECTED`. |
| **R-04** | **Prompt Injection em Documentos Externos** | Crítica | `PromptInjectionGuard` isola manuais externos como dados não confiáveis, neutralizando comandos que tentem desativar travas do sistema. |
| **R-05** | **Vazamento de Credenciais em Logs e Datasets** | Crítica | `SecretScrubber` substitui senhas, chaves pré-compartilhadas e SNMP communities por `[REDACTED_SECRET]` antes da persistência. |
| **R-06** | **Divergência Numérica ou Explosão de Gradientes** | Média | Gradient clipping por norma L2 (`clip_grad_norm_`), detecção de `NaN`/`Inf` em cada passo de treino e preservação do último checkpoint estável. |
| **R-07** | **Regressão Técnica na Promoção de Pesos** | Alta | Isolamento entre `TRAINING`, `STAGING` e `PRODUCTION`. Promoção condicionada à aprovação formal em baterias de testes no `ModelRegistryService`. |
| **R-08** | **Expansão Excessiva da Neurogênese (Memory Bloat)** | Média | Poda sináptica periódica no ciclo de sono e calibração estrita do limiar de surpresa ($\tau_{\text{novidade}} = 0.75$). |
| **R-09** | **Erro de Interpretação em Comandos de Voz** | Alta | Interface de voz restrita à consulta e navegação. Nenhuma ação de escrita pode ser autorizada por áudio. |
| **R-10** | **Subestimação do Raio de Impacto (Blast Radius)** | Alta | Grafo topológico multi-camadas (L1 óptico, L2 VLAN/STP, L3 IGP/BGP) rastreando redundâncias e nós dependentes antes do plano. |
| **R-11** | **Falha de Reversão / Partição durante Rollback** | Crítica | Preparação antecipada do script de rollback no dispositivo antes da aplicação, garantindo reversão local mesmo em caso de queda de gerência. |
| **R-12** | **Dessincronização de Artefatos no Workspace** | Baixa | Transações atômicas no `ArtifactStore` (SQLite/SQLAlchemy) com verificação de soma de integridade e histórico versionado imutável. |
