# Esquemas de Datasets, Metadados e Persistência — VERONICA

Este documento define formalmente os esquemas de dados, formatos de serialização e estruturas de metadados utilizados na **Veronica**, abrangendo a base documental (RAG), o treinamento supervisionado, o aprendizado contínuo com incidentes operacionais validados e a persistência binária do sistema cognitivo.

---

## 1. Esquema dos 12 Metadados Obrigatórios da Base Documental (`knowledge/`)

Todo fragmento de conhecimento (`KnowledgeChunk`) ingerido, processado e indexado na Veronica deve preservar e preencher estritamente **12 campos de metadados obrigatórios**. Fragmentos com campos faltantes são rejeitados pelo pipeline de ingestão:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "KnowledgeChunkMetadata",
  "type": "object",
  "required": [
    "chunk_id",
    "vendor",
    "family_model",
    "os_name",
    "version_build",
    "feature",
    "original_source",
    "source_uri",
    "page_section",
    "published_date",
    "consulted_date",
    "reliability_score",
    "applicability_restrictions"
  ],
  "properties": {
    "chunk_id": { "type": "string", "description": "Identificador único e imutável do fragmento" },
    "vendor": { "type": "string", "enum": ["Huawei", "Cisco", "Juniper", "MikroTik", "Nokia", "Arista", "Fortinet", "IETF"] },
    "family_model": { "type": "string", "description": "Linha de hardware ou família de equipamentos" },
    "os_name": { "type": "string", "description": "Nome canônico do sistema operacional de rede" },
    "version_build": { "type": "string", "description": "Versão exata e build do software documentado" },
    "feature": { "type": "string", "description": "Protocolo ou funcionalidade abordada no trecho" },
    "original_source": { "type": "string", "description": "Publicação ou manual oficial de origem" },
    "source_uri": { "type": "string", "format": "uri", "description": "Link permanente ou identificador oficial do documento" },
    "page_section": { "type": "string", "description": "Capítulo, seção ou número de página no documento oficial" },
    "published_date": { "type": "string", "format": "date", "description": "Data de publicação original (AAAA-MM-DD)" },
    "consulted_date": { "type": "string", "format": "date", "description": "Data de coleta e indexação pelo sistema (AAAA-MM-DD)" },
    "reliability_score": { "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Índice de confiabilidade da fonte" },
    "applicability_restrictions": { "type": "string", "description": "Restrições de hardware, licença ou limitações de versão" }
  }
}
```

### 1.1 Exemplo de Fragmento Ingerido (Huawei VRP 8)
```json
{
  "chunk_id": "chunk-hw-vrp8-bgp-evpn-001",
  "vendor": "Huawei",
  "family_model": "NetEngine 8000 Series",
  "os_name": "VRP",
  "version_build": "V800R022C00SPC600",
  "feature": "BGP EVPN / VXLAN Distributed Gateway",
  "original_source": "Huawei HedEx NE8000 Configuration Guide - IP Routing",
  "source_uri": "https://support.huawei.com/enterprise/en/doc/EDOC1100278435",
  "page_section": "Capítulo 5.4 - Configuring BGP EVPN Peers (p. 112)",
  "published_date": "2024-06-15",
  "consulted_date": "2026-10-17",
  "reliability_score": 0.99,
  "applicability_restrictions": "Aplicável apenas a placas de linha com chip Solar. Requer commit no modo two-phase."
}
```

---

## 2. Formato de Dataset para Treinamento Supervisionado (`JSONL`)

Armazenado em arquivos `.jsonl` versionados, onde cada linha representa uma amostra de treinamento sintaticamente validada e higienizada por [SecretScrubber](file:///c:/Users/alexn/Desktop/Veronica/knowledge/ingestion/sanitizer.py):

```json
{
  "sample_id": "smpl-cisco-xr-bgp-0042",
  "dataset_version": "v2.1.0-production-routing",
  "domain": "routing.bgp.troubleshooting",
  "vendor": "Cisco",
  "os_name": "IOS-XR",
  "os_version": "7.9.2",
  "verified_by_human": true,
  "author_id": "usr-eng-lead-01",
  "prompt": "Investigue sessão BGP entre PE-01 (AS 65001) e PE-02 (AS 65002) presa em estado Active.",
  "completion": "DIAGNÓSTICO: O estado Active indica tentativa ativa de estabelecer conexão TCP na porta 179 sem sucesso. AÇÕES READ-ONLY RECOMENDADAS: 1. Execute 'show bgp ipv4 unicast summary' para conferir IP de peer e ASN remoto; 2. Execute 'ping <peer_ip> source <interface>' para validar reachability e MTU; 3. Verifique se há 'route-policy' aplicada obrigatoriamente exigida pelo IOS-XR; 4. Cheque se TTL security ou ebgp-multihop estão configurados adequadamente caso o peer não seja diretamente conectado."
}
```

---

## 3. Esquema de Registro de Incidentes Operacionais Validados (`core/training/`)

Incidentes resolvidos em produção com sucesso de resolução e teste de reversão comprovado são capturados via `record_validated_incident()` para enriquecimento do aprendizado contínuo:

```json
{
  "incident_id": "INC-2026-1024",
  "timestamp": "2026-10-22T11:45:00Z",
  "devices": ["SW-AGG-01", "SW-AGG-02"],
  "vendor": "Cisco",
  "os_name": "NX-OS",
  "os_version": "10.3(2)F",
  "symptom": "LACP bundle Eth1/1-2 apresentando flapping intermitente e suspensão de portas membro.",
  "root_cause": "Incompatibilidade no valor de 'lacp rate' (fast em SW-AGG-01 vs normal em SW-AGG-02) sob carga de CPU.",
  "diagnostic_steps_executed": [
    "show port-channel summary",
    "show lacp neighbor interface Eth1/1",
    "show system internal lacp event-history detail"
  ],
  "resolution_plan_id": "CHG-2026-0812",
  "rollback_tested": true,
  "human_approved_for_training": true,
  "audited_by": "usr-noc-architect-03"
}
```

---

## 4. Esquema de Quarentena e Aprovação Humana de Correções (`ContinuousLearningPipeline`)

Correções enviadas por engenheiros operadores entram obrigatoriamente em estado de quarentena (`PENDING_VERIFICATION`), sendo promovidas apenas após aprovação formal de um treinador ou arquiteto:

```json
{
  "correction_id": "CORR-2026-0089",
  "submitted_at": "2026-10-22T13:10:00Z",
  "submitted_by": "usr-eng-02",
  "status": "PENDING_VERIFICATION",
  "target_domain": "mpls.ldp.session",
  "original_model_output": "Para habilitar LDP no Juniper Junos utilize 'set protocols ldp all'.",
  "human_correction": "No Junos, LDP deve ser configurado sob 'set protocols ldp interface <nome>' e as interfaces devem possuir 'family mpls' ativa em 'interfaces <nome> unit 0'.",
  "justification": "Comando sugerido anteriormente era sintaticamente inválido no Junos 22.4R1.",
  "verified_at": "2026-10-22T13:30:00Z",
  "verified_by": "usr-trainer-lead-01",
  "approved_for_training": true
}
```

---

## 5. Especificação do Armazenamento Binário de Memória (`BinaryMachineMemoryStore`)

Para garantir latência inferior a 2 milissegundos nas consultas do sistema cognitivo em produção, o [BinaryMachineMemoryStore](file:///c:/Users/alexn/Desktop/Veronica/core/development/memory_system.py) adota uma estrutura binária compacta com cabeçalho de integridade:

| Offset (Bytes) | Campo | Tipo | Descrição |
|---|---|---|---|
| `0x00 - 0x07` | **Magic Header** | `ASCII [8]` | `"VERONICA"` |
| `0x08 - 0x0B` | **File Version** | `uint32` | Versão do formato binário (ex.: `1`) |
| `0x0C - 0x0F` | **Embedding Dim**| `uint32` | Dimensão vetorial dos conceitos (ex.: `128` ou `256`) |
| `0x10 - 0x17` | **Total Records** | `uint64` | Quantidade total de registros indexados |
| `0x18 - 0x37` | **Payload SHA** | `Bytes [32]`| Hash SHA-256 de validação do corpo de registros |
| `0x38 - ...`  | **Vector Table** | `float32[]` | Matriz de embeddings contígua para cálculo rápido de produto interno |
| `EOF - Index` | **Metadata Map** | `BSON/JSON` | Tabela serializada de identificadores, categorias e timestamps |
