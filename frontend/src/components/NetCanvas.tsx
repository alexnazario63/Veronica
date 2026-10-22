import React, { useState } from 'react';
import {
  FileText,
  GitCompare,
  Network,
  ShieldCheck,
  Activity,
  Layers,
  Maximize2,
  Minimize2,
  X,
  AlertTriangle,
  FileCheck
} from 'lucide-react';
import type { Artifact, StoredArtifact } from '../types/veronica';
import { DiffEditor } from './DiffEditor';
import { ChangeConsole } from './ChangeConsole';
import { QuickObsAdminDrawer } from './QuickObsAdminDrawer';

interface NetCanvasProps {
  artifact: Artifact;
  versions?: StoredArtifact[];
  onClose: () => void;
  onSaved: (updated: Artifact) => void;
  onAskAdjustment: (startLine: number, endLine: number, selectedText: string) => void;
  onSelectVersion?: (version: number) => void;
  onAbortChange?: () => Promise<void>;
  onApproveStage6?: (data: {
    approver_user_id: string;
    confirmed_maintenance_window: string;
    signature_ed25519: string;
    confirmed_plan_hash: string;
  }) => Promise<void>;
  api: <T>(path: string, init?: RequestInit) => Promise<T>;
}

export const NetCanvas: React.FC<NetCanvasProps> = ({
  artifact,
  versions = [],
  onClose,
  onSaved,
  onAskAdjustment,
  onSelectVersion,
  onAbortChange,
  onApproveStage6,
  api,
}) => {
  const [activeTab, setActiveTab] = useState<'diff' | 'nocui' | 'obs' | 'rca'>('diff');
  const [isExpanded, setIsExpanded] = useState(false);

  const handleSaveProposed = async (newContent: string) => {
    const lines = (artifact.proposed_content || '').split('\n').length;
    const result = await api<{ artifact: Artifact }>(
      `/api/workspace/artifacts/${encodeURIComponent(artifact.id)}/versions/${artifact.version}/proposed-content`,
      {
        method: 'PATCH',
        body: JSON.stringify({
          start_line: 1,
          end_line: Math.max(1, lines),
          replacement: newContent,
        }),
      }
    );
    onSaved(result.artifact);
  };

  return (
    <aside className={`netcanvas-panel ${isExpanded ? 'netcanvas-expanded' : ''} animate__animated animate__fadeInRight`}>
      {/* NetCanvas Header */}
      <div className="netcanvas-header">
        <div className="netcanvas-header-left">
          <span className="netcanvas-eyebrow">NETCANVAS · WORKUI</span>
          <div className="netcanvas-title-row">
            <h2 className="netcanvas-title">{artifact.type}</h2>
            <span className="netcanvas-version-pill">v{artifact.version}</span>
            <span className="netcanvas-id-pill" title={artifact.id}>
              {artifact.id}
            </span>
          </div>
        </div>

        <div className="netcanvas-header-actions">
          <button
            type="button"
            className="netcanvas-btn-icon"
            onClick={() => setIsExpanded(!isExpanded)}
            title={isExpanded ? 'Restaurar split-screen' : 'Expandir NetCanvas'}
          >
            {isExpanded ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
          </button>
          <button
            type="button"
            className="netcanvas-btn-icon close-btn"
            onClick={onClose}
            title="Fechar WorkUI NetCanvas"
          >
            <X size={17} />
          </button>
        </div>
      </div>

      {/* Tabs Navigation */}
      <div className="netcanvas-tabs">
        <button
          type="button"
          className={`netcanvas-tab ${activeTab === 'diff' ? 'active' : ''}`}
          onClick={() => setActiveTab('diff')}
        >
          <GitCompare size={13} /> Editor Diff &amp; Proposta
        </button>

        <button
          type="button"
          className={`netcanvas-tab ${activeTab === 'nocui' ? 'active' : ''}`}
          onClick={() => setActiveTab('nocui')}
        >
          <ShieldCheck size={13} /> NOCUI (10 Etapas &amp; Blast Radius)
        </button>

        <button
          type="button"
          className={`netcanvas-tab ${activeTab === 'obs' ? 'active' : ''}`}
          onClick={() => setActiveTab('obs')}
        >
          <Activity size={13} /> Gaveta ObsUI &amp; AdminUI
        </button>

        <button
          type="button"
          className={`netcanvas-tab ${activeTab === 'rca' ? 'active' : ''}`}
          onClick={() => setActiveTab('rca')}
        >
          <FileText size={13} /> Diagnóstico &amp; RCA
        </button>
      </div>

      {/* Tab Content */}
      <div className="netcanvas-content">
        {activeTab === 'diff' && (
          <DiffEditor
            artifact={artifact}
            versions={versions}
            onSelectVersion={onSelectVersion}
            onSaveProposed={handleSaveProposed}
            onAskAdjustment={onAskAdjustment}
          />
        )}

        {activeTab === 'nocui' && (
          <ChangeConsole
            artifact={artifact}
            onRefresh={() => {}}
            onAbortChange={onAbortChange}
            onApproveStage6={onApproveStage6}
          />
        )}

        {activeTab === 'obs' && (
          <QuickObsAdminDrawer api={api} />
        )}

        {activeTab === 'rca' && (
          <div className="rca-tab-content">
            <div className="rca-header">
              <FileCheck size={17} />
              <div>
                <b>Relatório de Diagnóstico e Causa Raiz (RCA)</b>
                <small>Investigação determinística guiada por telemetria e catálogo de 12 metadados</small>
              </div>
            </div>

            <div className="rca-card">
              <h4>1. Trilha de Evidências Coletadas</h4>
              <div className="rca-evidence-item">
                <span className="rca-badge fact">FATO VERIFICADO</span>
                <p>Queda de adjacência BGP com neighbor 10.0.0.2 registrada nos contadores de interface.</p>
              </div>
              <div className="rca-evidence-item">
                <span className="rca-badge inf">INFERÊNCIA</span>
                <p>A perda de pacotes ocorre no enlace de agregação antes do nó Core 01.</p>
              </div>
              <div className="rca-evidence-item">
                <span className="rca-badge hyp">HIPÓTESE</span>
                <p>MTU mismatch ou flap no enlace óptico após manutenção recente de fibra.</p>
              </div>
            </div>

            <div className="rca-card">
              <h4>2. Plano de Mitigação e Recomendações</h4>
              <p>
                1. Executar verificação de MTU via comandos somente leitura (Tier 1): <code>ping 10.0.0.2 size 1500 df-bit</code>.<br />
                2. Iniciar plano de mudança controlado via /mudanca caso seja necessária alteração de interface.
              </p>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};
