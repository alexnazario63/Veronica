import React, { useState, useEffect, useRef } from 'react';
import cytoscape, { type ElementDefinition, type Core } from 'cytoscape';
import {
  ShieldAlert,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  Play,
  RotateCcw,
  KeyRound,
  Lock,
  Clock,
  Radio,
  Network,
  Maximize2,
  RefreshCw,
  X,
  FileCheck,
  AlertOctagon,
  MicOff
} from 'lucide-react';
import type { Artifact, ChangePlanRecord } from '../types/veronica';

interface ChangeConsoleProps {
  artifact: Artifact;
  onRefresh?: () => void;
  onAbortChange?: () => Promise<void>;
  onApproveStage6?: (approvalData: {
    approver_user_id: string;
    confirmed_maintenance_window: string;
    signature_ed25519: string;
    confirmed_plan_hash: string;
  }) => Promise<void>;
}

const STAGES = [
  { num: 1, id: '1_DISCOVERY', label: 'Descoberta', desc: 'Identificação de ativos e vizinhança' },
  { num: 2, id: '2_BASELINE', label: 'Baseline', desc: 'Coleta de snapshot operacional limpo' },
  { num: 3, id: '3_IMPACT_ANALYSIS', label: 'Análise de Impacto', desc: 'Cálculo de Blast Radius topológico' },
  { num: 4, id: '4_PLANNING', label: 'Planejamento', desc: 'Comandos atômicos e plano de rollback' },
  { num: 5, id: '5_SIMULATION', label: 'Simulação', desc: 'Verificação em dry-run virtual' },
  { num: 6, id: '6_HUMAN_APPROVAL', label: 'Aprovação Humana', desc: 'Assinatura Ed25519 via Web UI (R-09)' },
  { num: 7, id: '7_CONTROLLED_EXECUTION', label: 'Execução Controlada', desc: 'Aplicação de comandos com timer' },
  { num: 8, id: '8_POST_VALIDATION', label: 'Pós-Validação', desc: 'Checagem de invariantes operacionais' },
  { num: 9, id: '9_COMMIT_OR_ROLLBACK', label: 'Commit ou Rollback', desc: 'Confirmação ou reversão automática' },
  { num: 10, id: '10_IMMUTABLE_AUDIT', label: 'Auditoria Imutável', desc: 'Registro criptográfico append-only' },
];

function parseStageNumber(stageStr?: string | null): number {
  if (!stageStr) return 4; // default stage for draft change plans
  const match = stageStr.match(/(?:stage[_-]?)?(\d+)/i);
  return match ? parseInt(match[1], 10) : 4;
}

export const ChangeConsole: React.FC<ChangeConsoleProps> = ({
  artifact,
  onRefresh,
  onAbortChange,
  onApproveStage6,
}) => {
  const currentStageNum = parseStageNumber(artifact.change_stage);

  const [approvalModalOpen, setApprovalModalOpen] = useState(false);
  const [approverId, setApproverId] = useState('usr-app-01');
  const [maintenanceWindow, setMaintenanceWindow] = useState('2026-10-22 02:00-04:00 UTC');
  const [signatureEd25519, setSignatureEd25519] = useState('sig_ed25519_operator_approved_web_channel_valid');
  const [isApproving, setIsApproving] = useState(false);
  const [isAborting, setIsAborting] = useState(false);
  const [rollbackAlert, setRollbackAlert] = useState(false);
  const [timerSeconds, setTimerSeconds] = useState(300);

  const cyContainerRef = useRef<HTMLDivElement>(null);
  const cyInstanceRef = useRef<Core | null>(null);

  // Commit Confirmed countdown timer for stages 7-8
  useEffect(() => {
    if (currentStageNum < 7 || currentStageNum > 8) return;
    const interval = window.setInterval(() => {
      setTimerSeconds(s => Math.max(0, s - 1));
    }, 1000);
    return () => clearInterval(interval);
  }, [currentStageNum]);

  // Cytoscape Blast Radius Graph
  useEffect(() => {
    if (!cyContainerRef.current) return;

    const graphData = artifact.blast_radius_graph || {};
    const nested = (graphData as Record<string, unknown>).blast_radius_graph && typeof (graphData as Record<string, unknown>).blast_radius_graph === 'object'
      ? (graphData as Record<string, unknown>).blast_radius_graph as Record<string, unknown>
      : graphData;

    const rawNodes = Array.isArray(nested.nodes) ? (nested.nodes as Record<string, unknown>[]) : [];
    const rawEdges = Array.isArray(nested.edges) ? (nested.edges as Record<string, unknown>[]) : [];
    const rootId = typeof nested.root === 'string' ? nested.root : 'cisco-iosxr-01';

    // Fallback graph if empty
    const elements: ElementDefinition[] = [];

    if (rawNodes.length === 0) {
      elements.push(
        { data: { id: rootId, label: `${rootId}\n(Alvo da Mudança)`, impact: 'root' } },
        { data: { id: 'juniper-mx-02', label: 'juniper-mx-02\n(Direto)', impact: 'direct' } },
        { data: { id: 'arista-eos-03', label: 'arista-eos-03\n(Direto)', impact: 'direct' } },
        { data: { id: 'huawei-vrp-04', label: 'huawei-vrp-04\n(Transitivo)', impact: 'transitive' } },
        { data: { id: 'nokia-sr-05', label: 'nokia-sr-05\n(Transitivo)', impact: 'transitive' } },
        { data: { id: 'e1', source: rootId, target: 'juniper-mx-02' } },
        { data: { id: 'e2', source: rootId, target: 'arista-eos-03' } },
        { data: { id: 'e3', source: 'juniper-mx-02', target: 'huawei-vrp-04' } },
        { data: { id: 'e4', source: 'arista-eos-03', target: 'nokia-sr-05' } }
      );
    } else {
      rawNodes.forEach(node => {
        const id = String(node.id ?? node.device_id ?? '');
        if (id) {
          const impact = id === rootId ? 'root' : String(node.impact ?? 'transitive');
          elements.push({ data: { id, label: id, impact } });
        }
      });
      rawEdges.forEach((edge, idx) => {
        const source = String(edge.source ?? edge.from ?? '');
        const target = String(edge.target ?? edge.to ?? '');
        if (source && target) {
          elements.push({ data: { id: `edge-${idx}`, source, target } });
        }
      });
    }

    if (cyInstanceRef.current) {
      cyInstanceRef.current.destroy();
    }

    const cy = cytoscape({
      container: cyContainerRef.current,
      elements,
      layout: {
        name: 'breadthfirst',
        directed: false,
        padding: 24,
      },
      style: [
        {
          selector: 'node',
          style: {
            label: 'data(label)',
            'background-color': '#537b78',
            color: '#d8e4e4',
            'font-size': 10,
            'text-valign': 'bottom',
            'text-margin-y': 6,
            width: '32px',
            height: '32px',
            'text-wrap': 'wrap',
            'text-max-width': '110px',
          },
        },
        {
          selector: 'node[impact = "root"]',
          style: {
            'background-color': '#e75b5b',
            width: '42px',
            height: '42px',
            'border-width': '3px',
            'border-color': '#ff9b9b',
          },
        },
        {
          selector: 'node[impact = "direct"]',
          style: {
            'background-color': '#e59a44',
            width: '36px',
            height: '36px',
          },
        },
        {
          selector: 'node[impact = "transitive"]',
          style: {
            'background-color': '#4ea99b',
          },
        },
        {
          selector: 'edge',
          style: {
            width: '2px',
            'line-color': '#476366',
            'curve-style': 'bezier',
          },
        },
      ],
    });

    cyInstanceRef.current = cy;

    return () => {
      cy.destroy();
      cyInstanceRef.current = null;
    };
  }, [artifact.blast_radius_graph]);

  const handleApprove = async () => {
    if (!onApproveStage6) return;
    setIsApproving(true);
    try {
      await onApproveStage6({
        approver_user_id: approverId,
        confirmed_maintenance_window: maintenanceWindow,
        signature_ed25519: signatureEd25519,
        confirmed_plan_hash: artifact.sha256_hash,
      });
      setApprovalModalOpen(false);
      if (onRefresh) onRefresh();
    } finally {
      setIsApproving(false);
    }
  };

  const handleAbort = async () => {
    if (!window.confirm('ATENÇÃO: Deseja realmente abortar e acionar o Rollback imediato? Todos os comandos em execução serão interrompidos e a baseline original será restaurada.')) {
      return;
    }
    setIsAborting(true);
    try {
      if (onAbortChange) {
        await onAbortChange();
      }
      setRollbackAlert(true);
      if (onRefresh) onRefresh();
    } finally {
      setIsAborting(false);
    }
  };

  return (
    <div className="change-console-container">
      {/* Header Banner */}
      <div className="change-console-banner">
        <div className="banner-left">
          <ShieldCheck size={20} className="icon-shield" />
          <div>
            <h3>Deterministic Change Controller (NOCUI)</h3>
            <p>
              Fluxo Determinístico de 10 Etapas · Separação de Funções · Auditoria Append-Only
            </p>
          </div>
        </div>

        <div className="banner-actions">
          {/* Immediate Abort & Rollback Button */}
          <button
            type="button"
            className="btn-abort-rollback"
            onClick={handleAbort}
            disabled={isAborting}
            title="Acionar RollbackOrchestrator e abortar mudança de emergência"
          >
            <AlertOctagon size={14} /> {isAborting ? 'Revertendo...' : 'Abort & Rollback Imediato'}
          </button>
        </div>
      </div>

      {rollbackAlert && (
        <div className="rollback-notice animate__animated animate__fadeIn">
          <AlertTriangle size={16} />
          <span>Rollback acionado com sucesso. Configuração retornada à running baseline.</span>
          <button type="button" onClick={() => setRollbackAlert(false)}>×</button>
        </div>
      )}

      {/* 10-Stage Visual Stepper */}
      <div className="stage-stepper-wrap">
        <div className="stage-stepper-title">
          <span>PROGRESSO DETERMINÍSTICO (ETAPAS 1 A 10)</span>
          <span className="current-stage-tag">
            Etapa Atual: <strong>{currentStageNum}</strong> ({STAGES[currentStageNum - 1]?.label})
          </span>
        </div>

        <div className="stage-stepper-grid">
          {STAGES.map(stage => {
            const isDone = stage.num < currentStageNum;
            const isCurrent = stage.num === currentStageNum;
            const isPending = stage.num > currentStageNum;

            return (
              <div
                key={stage.id}
                className={`stepper-node ${isDone ? 'done' : ''} ${isCurrent ? 'current' : ''} ${
                  isPending ? 'pending' : ''
                }`}
              >
                <div className="node-icon-circle">
                  {isDone ? (
                    <CheckCircle2 size={13} />
                  ) : isCurrent ? (
                    <span className="current-pulse" />
                  ) : (
                    <Lock size={11} />
                  )}
                </div>
                <div className="node-text">
                  <div className="node-num-name">
                    <span className="node-num">{stage.num}</span>
                    <b className="node-label">{stage.label}</b>
                  </div>
                  <small className="node-desc">{stage.desc}</small>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Stage 6 Interactive Action Card */}
      {currentStageNum === 6 && (
        <div className="stage-action-card stage-6-card animate__animated animate__pulse">
          <div className="action-card-header">
            <KeyRound size={18} className="text-warning" />
            <div>
              <h4>Etapa 6: Aprovação Humana Obrigatória</h4>
              <p>A execução controlada (Etapa 7) exige assinatura criptográfica de operador habilitado.</p>
            </div>
            <button
              type="button"
              className="btn-approve-primary"
              onClick={() => setApprovalModalOpen(true)}
            >
              <FileCheck size={14} /> Assinar e Aprovar Mudança
            </button>
          </div>
        </div>
      )}

      {/* Commit Confirmed Timer (Stages 7 & 8) */}
      {currentStageNum >= 7 && currentStageNum <= 8 && (
        <div className="commit-timer-card">
          <div className="timer-info">
            <Clock size={18} className="icon-clock" />
            <div>
              <b>Janela de Commit Confirmed Ativa</b>
              <p>O equipamento reverterá para a baseline caso a pós-validação não aprove em tempo hábil.</p>
            </div>
          </div>
          <div className="timer-countdown">
            {String(Math.floor(timerSeconds / 60)).padStart(2, '0')}:
            {String(timerSeconds % 60).padStart(2, '0')}
          </div>
        </div>
      )}

      {/* Blast Radius Cytoscape Section */}
      <div className="blast-radius-section">
        <div className="section-head">
          <div className="head-title">
            <Network size={16} />
            <b>Grafo de Blast Radius &amp; Análise Topológica</b>
          </div>
          <div className="legend-pills">
            <span className="legend-item root">
              <span className="dot" /> Alvo Principal
            </span>
            <span className="legend-item direct">
              <span className="dot" /> Impacto Direto
            </span>
            <span className="legend-item transitive">
              <span className="dot" /> Impacto Transitivo
            </span>
          </div>
        </div>

        <div className="cytoscape-container-wrap">
          <div className="cytoscape-canvas" ref={cyContainerRef} />
        </div>
      </div>

      {/* Modal de Assinatura Humana para a Etapa 6 (com bloqueio explícito de microfone conforme ADR-03 / R-09) */}
      {approvalModalOpen && (
        <div className="approval-modal-overlay" onClick={() => setApprovalModalOpen(false)}>
          <div className="approval-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title-group">
                <ShieldAlert size={20} className="icon-alert" />
                <div>
                  <h3>Assinatura de Aprovação de Mudança · Etapa 6</h3>
                  <small>Garantia de Segurança Operacional e Separação de Funções</small>
                </div>
              </div>
              <button
                type="button"
                className="modal-close-btn"
                onClick={() => setApprovalModalOpen(false)}
              >
                <X size={17} />
              </button>
            </div>

            <div className="modal-body">
              {/* ADR-03 / R-09 Explicit Block Alert */}
              <div className="adr-voice-block-banner">
                <div className="adr-banner-icon">
                  <MicOff size={22} />
                </div>
                <div className="adr-banner-text">
                  <b>ADR-03 / R-09: PROIBIÇÃO EXPRESSA DE APROVAÇÃO POR VOZ</b>
                  <p>
                    Aprovação de Mudanças em Rede (Stage 6) é <strong>estritamente restrita</strong> a esta
                    interface Web autenticada. Comandos por voz, transcrições de microfone e fluxos automatizados
                    são bloqueados em conformidade com as normas de engenharia de rede.
                  </p>
                </div>
              </div>

              {/* Form Fields */}
              <div className="form-group">
                <label>ID DO OPERADOR APROVADOR (RBAC: change:approve)</label>
                <input
                  type="text"
                  value={approverId}
                  onChange={e => setApproverId(e.target.value)}
                  placeholder="usr-app-01"
                  className="modal-input"
                />
                <small className="field-hint">
                  A regra de separação de funções impede que o solicitante aprove sua própria mudança.
                </small>
              </div>

              <div className="form-group">
                <label>JANELA DE MANUTENÇÃO CONFIRMADA</label>
                <input
                  type="text"
                  value={maintenanceWindow}
                  onChange={e => setMaintenanceWindow(e.target.value)}
                  className="modal-input"
                />
              </div>

              <div className="form-group">
                <label>HASH CRIPTOGRÁFICO DO PLANO (SHA-256)</label>
                <div className="hash-display-box">
                  <code>{artifact.sha256_hash}</code>
                </div>
              </div>

              <div className="form-group">
                <label>TOKEN DE ASSINATURA DIGITAL (Ed25519)</label>
                <input
                  type="text"
                  value={signatureEd25519}
                  onChange={e => setSignatureEd25519(e.target.value)}
                  className="modal-input monospace"
                />
                <small className="field-hint">
                  Canal de Aprovação: <code>EXPLICIT_WEB_APPROVAL</code>
                </small>
              </div>
            </div>

            <div className="modal-footer">
              <button
                type="button"
                className="btn-cancel"
                onClick={() => setApprovalModalOpen(false)}
                disabled={isApproving}
              >
                Cancelar
              </button>
              <button
                type="button"
                className="btn-confirm-approve"
                onClick={handleApprove}
                disabled={isApproving || !approverId.trim()}
              >
                <FileCheck size={14} />
                {isApproving ? 'Validando Assinatura...' : 'Assinar & Liberar Etapa 7'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
