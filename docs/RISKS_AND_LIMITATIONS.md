# Limitações, Riscos e Controles de Mitigação — VERONICA

## 1. Limitações Técnicas Conhecidas

1. **Escala Computacional do Motor Nativo**:
   - Como o motor neural da Veronica é desenvolvido desde os fundamentos matemáticos (sem usar pesos pré-treinados de LLMs comerciais de bilhões de parâmetros), modelos treinados localmente em CPU/GPU doméstica possuem capacidade paramétrica proporcional ao tamanho do corpus e ao número de parâmetros configurados (ex.: micro/small/medium).
   - **Mitigação**: A Veronica combina seu Transformer nativo com o motor RAG documental determinístico e analisadores de protocolo especialistas, garantindo que comandos CLI e diagnósticos críticos sejam sempre ancorados em documentação oficial validada e nunca na mera geração livre de tokens.

2. **Cobertura por Plataforma e Versão de Sistema Operacional**:
   - Nenhum fabricante possui CLI única em toda a sua linha (ex.: Cisco IOS vs IOS-XR vs NX-OS; Huawei VRPv5 vs VRPv8; MikroTik RouterOS v6 vs v7).
   - **Mitigação**: A matriz de compatibilidade (`CompatibilityMatrix`) exige correspondência exata de `(vendor, os_type, version_family)`. Quando um equipamento ou versão específica ainda não possui validação completa, a Veronica declara explicitamente `UNVALIDATED_VERSION` ou `UNSUPPORTED_PLATFORM` e bloqueia geração de candidatos de escrita.

3. **Conversação por Voz em Ambientes de Operação (NOC)**:
   - Ruído ambiente em centros de operação e pronúncia de termos técnicos (ex.: `OSPFv3`, `GigabitEthernet0/0/1`, `AS65001`) podem introduzir erros de transcrição no ASR.
   - **Mitigação**: A transcrição é exibida para revisão/edição antes ou durante o envio, possui normalizador fonético para termos de redes e **jamais pode autorizar mudanças em produção**.

---

## 2. Matriz de Riscos Operacionais e Controles

| ID | Risco Identificado | Severidade | Controle / Mitigação Arquitetural |
|---|---|---|---|
| **R-01** | Alucinação de comandos CLI inexistentes ou incompatíveis | Crítica | Validador sintático determinístico (`validate_commands`) no adaptador do fabricante + verificação cruzada com a base documental oficial antes de exibir comandos propostos. |
| **R-02** | Perda de sessão administrativa (lockout de SSH/NETCONF) durante alteração | Crítica | Análise de impacto bloqueia alterações na interface/ACL/rota de gerência sem canal Out-of-Band (OOB) verificado e exige mecanismo de `commit confirmed` / rollback automático por timer. |
| **R-03** | Sobrescrita de alterações concorrentes de outros operadores | Alta | Coleta de hash de configuração no Baseline (Etapa 2) e revalidação imediatamente antes da Execução Controlada (Etapa 7). Se o hash divergir, a execução é abortada (`CONCURRENT_MODIFICATION_DETECTED`). |
| **R-04** | Injeção de instruções maliciosas (Prompt Injection) via documentos externos | Crítica | Sanitizador de ingestão (`PromptInjectionGuard`) detecta tentativas de alterar regras do sistema, comandos destrutivos camuflados (`erase`, `format`, `write erase`, `reset saved-configuration`) e isola documentos externos como dados não confiáveis. |
| **R-05** | Vazamento de senhas, chaves BGP/OSPF, SNMP communities ou tokens em logs/datasets | Crítica | Redator automático de segredos (`SecretVault & Scrubber`) mascara credenciais antes de persistir logs, históricos de chat ou datasets de treinamento. |
| **R-06** | Divergência numérica ou explosão de gradientes durante o treinamento neural | Média | Gradient clipping por norma L2 (`clip_grad_norm_`), detecção de `NaN`/`Inf` por step e preservação intacta do checkpoint anterior homologado. |
| **R-07** | Promoção indevida de checkpoint recém-treinado com regressão técnica | Alta | Separação estrita entre ambientes (`TRAINING`, `STAGING`, `PRODUCTION`). Promoção exige execução de benchmark técnico e aprovação explícita no `ModelRegistryService`. |
