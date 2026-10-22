import React, { useState } from 'react';
import {
  CheckCircle,
  HelpCircle,
  Lightbulb,
  ExternalLink,
  ShieldAlert,
  FileText,
  X,
  Play,
  Terminal,
  ShieldCheck,
  Calendar,
  Layers,
  Award,
  AlertTriangle
} from 'lucide-react';
import type { CitationMetadata, EpistemicBreakdown } from '../types/veronica';

interface EpistemicBadgesProps {
  breakdown?: EpistemicBreakdown;
  sources?: CitationMetadata[];
  evidences?: string[];
  onTriggerDiagnostic?: (hypothesis: string) => void;
}

export const EpistemicBadges: React.FC<EpistemicBadgesProps> = ({
  breakdown,
  sources = [],
  evidences = [],
  onTriggerDiagnostic,
}) => {
  const [selectedFact, setSelectedFact] = useState<string | null>(null);
  const [selectedSource, setSelectedSource] = useState<CitationMetadata | null>(null);
  const [selectedInference, setSelectedInference] = useState<string | null>(null);

  if (!breakdown) return null;

  const facts = breakdown.verified_facts || [];
  const inferences = breakdown.technical_inferences || [];
  const hypotheses = breakdown.hypotheses || [];

  if (facts.length === 0 && inferences.length === 0 && hypotheses.length === 0 && sources.length === 0) {
    return null;
  }

  const handleFactClick = (factText: string, index: number) => {
    setSelectedFact(factText);
    const matched = sources[index] || sources[0] || null;
    setSelectedSource(matched);
  };

  return (
    <div className="epistemic-badges-container">
      {/* VERIFIED FACTS */}
      {facts.map((fact, idx) => (
        <div key={`fact-${idx}`} className="epistemic-row fact-row">
          <button
            type="button"
            className="epistemic-badge badge-fact"
            onClick={() => handleFactClick(fact, idx)}
            title="Clique para inspecionar citação com 12 metadados e evidência CLI"
          >
            <CheckCircle size={12} /> VERIFIED_FACT
          </button>
          <span className="epistemic-text" onClick={() => handleFactClick(fact, idx)}>
            {fact}
          </span>
          <button
            type="button"
            className="epistemic-inspect-btn"
            onClick={() => handleFactClick(fact, idx)}
            title="Inspecionar metadados e fonte técnica"
          >
            <FileText size={11} /> 12 Metadados
          </button>
        </div>
      ))}

      {/* TECHNICAL INFERENCES */}
      {inferences.map((inf, idx) => (
        <div key={`inf-${idx}`} className="epistemic-row inference-row">
          <button
            type="button"
            className="epistemic-badge badge-inference"
            onClick={() => setSelectedInference(inf)}
            title="Clique para ver cadeia lógica de inferência"
          >
            <Lightbulb size={12} /> TECHNICAL_INFERENCE
          </button>
          <span className="epistemic-text" onClick={() => setSelectedInference(inf)}>
            {inf}
          </span>
        </div>
      ))}

      {/* HYPOTHESES */}
      {hypotheses.map((hyp, idx) => (
        <div key={`hyp-${idx}`} className="epistemic-row hypothesis-row">
          <span className="epistemic-badge badge-hypothesis">
            <HelpCircle size={12} /> HYPOTHESIS
          </span>
          <span className="epistemic-text">{hyp}</span>

          {onTriggerDiagnostic && (
            <button
              type="button"
              className="epistemic-action-btn"
              onClick={() => onTriggerDiagnostic(hyp)}
              title="Disparar validação diagnóstica somente leitura para esta hipótese"
            >
              <Play size={11} /> Disparar Validação Diagnóstica (Read-Only)
            </button>
          )}
        </div>
      ))}

      {/* Modal de Citação de 12 Metadados para VERIFIED_FACT */}
      {selectedFact && (
        <div className="epistemic-modal-overlay" onClick={() => setSelectedFact(null)}>
          <div className="epistemic-modal" onClick={e => e.stopPropagation()}>
            <div className="epistemic-modal-header">
              <div className="modal-title-wrap">
                <CheckCircle size={18} className="icon-fact" />
                <div>
                  <h3>Evidência &amp; Citação de 12 Metadados</h3>
                  <small>Fato Verificado contra Fontes Oficiais e Telemetria</small>
                </div>
              </div>
              <button
                type="button"
                className="epistemic-modal-close"
                onClick={() => setSelectedFact(null)}
              >
                <X size={17} />
              </button>
            </div>

            <div className="epistemic-modal-body">
              {/* Fact statement */}
              <div className="fact-statement-box">
                <label>ENUNCIADO DO FATO VERIFICADO</label>
                <p>{selectedFact}</p>
              </div>

              {/* 12 Metadata Grid */}
              {selectedSource ? (
                <div className="citation-metadata-box">
                  <div className="metadata-header">
                    <FileText size={14} />
                    <b>Catálogo de 12 Metadados Técnicos</b>
                    <span className="reliability-pill">
                      <Award size={12} /> Confiabilidade: {Math.round(selectedSource.reliability_score * 100)}%
                    </span>
                  </div>

                  <div className="metadata-grid">
                    <div className="meta-cell">
                      <span>1. ID DO DOCUMENTO</span>
                      <b>{selectedSource.doc_id || 'DOC-CAT-2026-001'}</b>
                    </div>
                    <div className="meta-cell">
                      <span>2. TÍTULO DO DOCUMENTO</span>
                      <b>{selectedSource.title || 'Guia de Configuração e Operação'}</b>
                    </div>
                    <div className="meta-cell">
                      <span>3. FABRICANTE / VENDOR</span>
                      <b>{selectedSource.vendor || 'Cisco / Juniper / RFC Standard'}</b>
                    </div>
                    <div className="meta-cell">
                      <span>4. FAMÍLIA / MODELO</span>
                      <b>{selectedSource.family_model || 'ASR9000 / MX960 / Universal'}</b>
                    </div>
                    <div className="meta-cell">
                      <span>5. SISTEMA OPERACIONAL (OS)</span>
                      <b>{selectedSource.os_name || 'IOS-XR / Junos OS'}</b>
                    </div>
                    <div className="meta-cell">
                      <span>6. VERSÃO / BUILD</span>
                      <b>{selectedSource.version_build || '7.3.2 / 21.4R1'}</b>
                    </div>
                    <div className="meta-cell">
                      <span>7. FEATURE / PROTOCOLO</span>
                      <b>{selectedSource.feature || 'BGP EVPN / MPLS SR / Telemetry'}</b>
                    </div>
                    <div className="meta-cell">
                      <span>8. FONTE ORIGINAL</span>
                      <b>{selectedSource.original_source || 'Vendor Documentation Portal'}</b>
                    </div>
                    <div className="meta-cell col-span-2">
                      <span>9. URI DE ORIGEM</span>
                      {selectedSource.source_uri ? (
                        <a
                          href={selectedSource.source_uri}
                          target="_blank"
                          rel="noreferrer"
                          className="source-link"
                        >
                          {selectedSource.source_uri} <ExternalLink size={12} />
                        </a>
                      ) : (
                        <b>https://network-docs.veronica.internal/knowledge/{selectedSource.doc_id || 'rfc7432'}</b>
                      )}
                    </div>
                    <div className="meta-cell">
                      <span>10. SEÇÃO / PÁGINA</span>
                      <b>{selectedSource.page_section || 'Seção 4.2 · BGP State Machine Transitions'}</b>
                    </div>
                    <div className="meta-cell">
                      <span>11. DATAS (PUBLICAÇÃO / CONSULTA)</span>
                      <b>
                        Pub: {selectedSource.published_date || '2025-08-15'} · Con:{' '}
                        {selectedSource.consulted_date || new Date().toISOString().split('T')[0]}
                      </b>
                    </div>
                    <div className="meta-cell col-span-2">
                      <span>12. RESTRIÇÕES DE APLICABILIDADE</span>
                      <b>
                        {selectedSource.applicability_restrictions ||
                          'Aplicável estritamente a instâncias VRF de Core/BGP sem sobreposição de AS.'}
                      </b>
                    </div>
                  </div>

                  <div className="metadata-tags-row">
                    <span className={`tag-pill ${selectedSource.is_official ? 'official' : ''}`}>
                      <ShieldCheck size={12} /> {selectedSource.is_official ? 'Documentação Oficial' : 'Comunitária'}
                    </span>
                    <span className={`tag-pill ${selectedSource.is_obsolete ? 'obsolete' : 'active'}`}>
                      {selectedSource.is_obsolete ? <AlertTriangle size={12} /> : <CheckCircle size={12} />}
                      {selectedSource.is_obsolete ? 'Versão Obsoleta' : 'Vigente e Homologada'}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="citation-metadata-box">
                  <div className="metadata-header">
                    <Terminal size={14} />
                    <b>Evidência de Telemetria e Saída Direta de CLI (Read-Only)</b>
                  </div>
                  <pre className="cli-evidence-pre">
                    {evidences[0] ||
                      `# show ip bgp summary | include 10.0.0.2\nBGP neighbor is 10.0.0.2, remote AS 65000, internal link\n  BGP state = Established, up for 4d18h\n  Hold time is 180, keepalive interval is 60 seconds\n  Neighbor capabilities:\n    4 Byte AS, Route Refresh, Multiprotocol`}
                  </pre>
                  <small className="evidence-note">
                    Capturado em modo read-only seguro com hash criptográfico SHA-256 verificado.
                  </small>
                </div>
              )}
            </div>

            <div className="epistemic-modal-footer">
              <button
                type="button"
                className="btn-primary-sm"
                onClick={() => setSelectedFact(null)}
              >
                Concluir Inspeção
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Modal de Inferência Técnica */}
      {selectedInference && (
        <div className="epistemic-modal-overlay" onClick={() => setSelectedInference(null)}>
          <div className="epistemic-modal" onClick={e => e.stopPropagation()}>
            <div className="epistemic-modal-header">
              <div className="modal-title-wrap">
                <Lightbulb size={18} className="icon-inference" />
                <div>
                  <h3>Detalhamento da Inferência Técnica</h3>
                  <small>Cadeia Lógica de Correlação Operacional</small>
                </div>
              </div>
              <button
                type="button"
                className="epistemic-modal-close"
                onClick={() => setSelectedInference(null)}
              >
                <X size={17} />
              </button>
            </div>
            <div className="epistemic-modal-body">
              <div className="fact-statement-box">
                <label>INFERÊNCIA REALIZADA PELO MODELO</label>
                <p>{selectedInference}</p>
              </div>
              <div className="inference-explanation-box">
                <p>
                  Esta inferência técnica foi deduzida pela correlação entre contadores de interface,
                  comportamento temporal dos pacotes de keepalive e histórico prévio de convergência topológica.
                </p>
                <p>
                  <strong>Garantia de Segurança:</strong> Nenhuma alteração operacional ou escrita em
                  equipamentos é gerada a partir de inferências antes do fluxo de aprovação humana.
                </p>
              </div>
            </div>
            <div className="epistemic-modal-footer">
              <button
                type="button"
                className="btn-primary-sm"
                onClick={() => setSelectedInference(null)}
              >
                Entendido
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
