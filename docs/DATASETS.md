# Formatos de Datasets e Metadados — VERONICA

## 1. Metadados Obrigatórios da Base de Conhecimento (Seção 4)

Todo documento ou fragmento (`KnowledgeChunk`) ingerido pela Veronica deve preservar obrigatoriamente os **12 campos de metadados**:

```json
{
  "chunk_id": "doc-huawei-vrp8-bgp-001",
  "vendor": "Huawei",
  "family_model": "NetEngine 8000 / CloudEngine",
  "os_name": "VRP",
  "version_build": "V800R022C00SPC600",
  "feature": "BGP-4 / MP-BGP Peer Configuration",
  "original_source": "Huawei HedEx Official Documentation",
  "source_uri": "https://support.huawei.com/enterprise/en/doc/EDOC1100278435",
  "page_section": "Section 4.2 - Configuring Basic BGP Functions (p. 48)",
  "published_date": "2024-06-15",
  "consulted_date": "2026-10-17",
  "reliability_score": 0.99,
  "applicability_restrictions": "Aplicável somente a VRP 8.x com commit em two-phase; comandos diferem do VRP 5.x (S5720)."
}
```

---

## 2. Formato de Dataset de Treinamento Neural (`JSONL`)

Os datasets de treinamento supervisionado e previsão do próximo token são versionados separadamente da base documental e passam por higienização de credenciais (`SecretScrubber`) e validação técnica antes do treinamento:

```json
{
  "sample_id": "net-train-0001",
  "dataset_version": "v1.0.0-core-networking",
  "domain": "routing.bgp",
  "vendor": "Cisco",
  "os_name": "IOS-XR",
  "os_version": "7.9.2",
  "verified_by_human": true,
  "prompt": "Diagnostique sessao BGP em estado Active no Cisco IOS-XR 7.9.2 entre PE1 (AS 65001) e PE2 (AS 65002).",
  "completion": "DIAGNOSTICO: Estado Active indica tentativa de conexao TCP na porta 179 sem sucesso. VERIFICAR: show bgp ipv4 unicast summary, show ipv4 interface brief, ping com source interface, remote-as e ebgp-multihop/ttl-security ou route-policy obrigatoria no IOS-XR."
}
```

---

## 3. Formato de Registro de Aprendizado Operacional (Incidentes Validados)

```json
{
  "incident_id": "INC-2026-0042",
  "timestamp": "2026-10-17T13:30:00Z",
  "devices": ["CORE-RTR-01", "DIST-SW-02"],
  "symptom": "OSPF adjacency stuck in EXSTART/EXCHANGE on TenGigE0/0/0/1",
  "root_cause": "MTU mismatch (9216 on CORE-RTR-01 vs 1500 on DIST-SW-02)",
  "diagnostic_steps_executed": [
    "show ospf neighbor detail",
    "show imchains / show interfaces TenGigE0/0/0/1"
  ],
  "resolution_plan_id": "CHG-2026-0019",
  "rollback_tested": true,
  "human_approved_for_training": true
}
```
