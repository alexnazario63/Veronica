import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  GitCommit,
  Check,
  Save,
  MessageSquare,
  Copy,
  ChevronDown,
  Columns,
  FileCode,
  Hash,
  ExternalLink,
  Edit3,
  RotateCcw
} from 'lucide-react';
import type { Artifact, StoredArtifact } from '../types/veronica';

interface DiffEditorProps {
  artifact: Artifact;
  versions?: StoredArtifact[];
  onSelectVersion?: (version: number) => void;
  onSaveProposed: (newContent: string) => Promise<void>;
  onAskAdjustment: (startLine: number, endLine: number, selectedText: string) => void;
}

interface DiffLine {
  type: 'same' | 'added' | 'removed' | 'empty';
  leftLineNum?: number;
  rightLineNum?: number;
  leftText?: string;
  rightText?: string;
}

export const DiffEditor: React.FC<DiffEditorProps> = ({
  artifact,
  versions = [],
  onSelectVersion,
  onSaveProposed,
  onAskAdjustment,
}) => {
  const [proposedText, setProposedText] = useState(artifact.proposed_content || '');
  const [isEditing, setIsEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [copied, setCopied] = useState(false);

  // Line selection state
  const [selectedStartLine, setSelectedStartLine] = useState<number | null>(null);
  const [selectedEndLine, setSelectedEndLine] = useState<number | null>(null);
  const [selectionAnchorY, setSelectionAnchorY] = useState<number>(0);

  const baselineText = artifact.baseline_content || '# Baseline indisponível no repositório local';

  useEffect(() => {
    setProposedText(artifact.proposed_content || '');
    setSelectedStartLine(null);
    setSelectedEndLine(null);
  }, [artifact.id, artifact.version, artifact.proposed_content]);

  // Compute simple side-by-side line diff
  const diffLines = useMemo<DiffLine[]>(() => {
    const baseLines = baselineText.split('\n');
    const candLines = proposedText.split('\n');
    const result: DiffLine[] = [];

    const maxLines = Math.max(baseLines.length, candLines.length);
    for (let i = 0; i < maxLines; i++) {
      const left = baseLines[i];
      const right = candLines[i];

      if (left === undefined) {
        result.push({
          type: 'added',
          rightLineNum: i + 1,
          leftText: '',
          rightText: right,
        });
      } else if (right === undefined) {
        result.push({
          type: 'removed',
          leftLineNum: i + 1,
          leftText: left,
          rightText: '',
        });
      } else if (left === right) {
        result.push({
          type: 'same',
          leftLineNum: i + 1,
          rightLineNum: i + 1,
          leftText: left,
          rightText: right,
        });
      } else {
        result.push({
          type: 'added',
          leftLineNum: i + 1,
          rightLineNum: i + 1,
          leftText: left,
          rightText: right,
        });
      }
    }
    return result;
  }, [baselineText, proposedText]);

  // Handle line click/selection on right pane
  const handleLineClick = (lineNum: number, event: React.MouseEvent) => {
    if (selectedStartLine === null || (selectedStartLine !== null && selectedEndLine !== null)) {
      setSelectedStartLine(lineNum);
      setSelectedEndLine(lineNum);
    } else {
      const start = Math.min(selectedStartLine, lineNum);
      const end = Math.max(selectedStartLine, lineNum);
      setSelectedStartLine(start);
      setSelectedEndLine(end);
    }
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
    setSelectionAnchorY(rect.top);
  };

  const handleAskVeronica = () => {
    if (selectedStartLine === null || selectedEndLine === null) return;
    const candLines = proposedText.split('\n');
    const snippet = candLines.slice(selectedStartLine - 1, selectedEndLine).join('\n');
    onAskAdjustment(selectedStartLine, selectedEndLine, snippet);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await onSaveProposed(proposedText);
      setIsEditing(false);
    } finally {
      setSaving(false);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(proposedText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const availableVersions = versions.length > 0
    ? versions.map(v => v.artifact.version)
    : [artifact.version];

  const hasModifications = proposedText !== (artifact.proposed_content || '');

  return (
    <div className="diff-editor-container">
      {/* Top Toolbar */}
      <div className="diff-toolbar">
        <div className="diff-toolbar-left">
          {/* Version Selector */}
          <div className="version-selector-wrap">
            <span className="diff-label">Versão:</span>
            <select
              value={artifact.version}
              onChange={e => onSelectVersion && onSelectVersion(Number(e.target.value))}
              className="version-dropdown"
            >
              {availableVersions.map(v => (
                <option key={v} value={v}>
                  v{v} {v === artifact.version ? '(ativa)' : ''}
                </option>
              ))}
            </select>
          </div>

          <span className="sha-hash-pill" title={`SHA-256: ${artifact.sha256_hash}`}>
            <Hash size={11} /> {artifact.sha256_hash.slice(0, 8)}...{artifact.sha256_hash.slice(-6)}
          </span>
        </div>

        <div className="diff-toolbar-right">
          <button
            type="button"
            className={`diff-tool-btn ${isEditing ? 'active' : ''}`}
            onClick={() => setIsEditing(!isEditing)}
            title="Alternar modo de edição da configuração proposta"
          >
            <Edit3 size={13} /> {isEditing ? 'Visualizar Diff' : 'Editar Proposta'}
          </button>

          <button
            type="button"
            className="diff-tool-btn"
            onClick={handleCopy}
            title="Copiar configuração proposta"
          >
            {copied ? <Check size={13} className="text-success" /> : <Copy size={13} />}
            {copied ? 'Copiado!' : 'Copiar'}
          </button>

          {hasModifications && (
            <button
              type="button"
              className="btn-primary-sm diff-save-btn"
              onClick={handleSave}
              disabled={saving}
            >
              <Save size={13} /> {saving ? 'Salvando...' : 'Salvar Nova Versão'}
            </button>
          )}
        </div>
      </div>

      {/* Floating Selection Banner / Ask Veronica Button */}
      {selectedStartLine !== null && selectedEndLine !== null && (
        <div className="diff-selection-bar animate__animated animate__fadeIn">
          <div className="diff-selection-info">
            <FileCode size={14} />
            <span>
              Linhas selecionadas: <strong>L{selectedStartLine}</strong> a <strong>L{selectedEndLine}</strong>
            </span>
          </div>

          <div className="diff-selection-actions">
            <button
              type="button"
              className="btn-ask-veronica"
              onClick={handleAskVeronica}
            >
              <MessageSquare size={13} /> Pedir ajuste à Veronica
            </button>
            <button
              type="button"
              className="diff-cancel-sel"
              onClick={() => {
                setSelectedStartLine(null);
                setSelectedEndLine(null);
              }}
            >
              Desmarcar
            </button>
          </div>
        </div>
      )}

      {/* Editor Body */}
      {isEditing ? (
        <div className="diff-raw-editor-wrap">
          <div className="raw-editor-header">
            <span>EDITANDO CONFIGURAÇÃO CANDIDATA (v{artifact.version})</span>
            <small>As edições salvas criam um novo snapshot imutável no ArtifactStore</small>
          </div>
          <textarea
            className="diff-raw-textarea"
            value={proposedText}
            onChange={e => setProposedText(e.target.value)}
            placeholder="# Digite ou ajuste a configuração candidata..."
            spellCheck={false}
          />
        </div>
      ) : (
        <div className="diff-grid-layout">
          {/* Left Column: Running Baseline */}
          <div className="diff-column baseline-column">
            <div className="column-header">
              <span className="col-tag baseline-tag">RUNNING BASELINE (ATIVO)</span>
              <small>Snapshot original coletado</small>
            </div>
            <div className="column-lines">
              {diffLines.map((line, idx) => (
                <div
                  key={`left-${idx}`}
                  className={`diff-line-row line-type-${line.type === 'removed' ? 'removed' : 'neutral'}`}
                >
                  <span className="line-num">{line.leftLineNum || ''}</span>
                  <pre className="line-content">{line.leftText || ''}</pre>
                </div>
              ))}
            </div>
          </div>

          {/* Right Column: Candidate Config */}
          <div className="diff-column candidate-column">
            <div className="column-header">
              <span className="col-tag candidate-tag">CANDIDATE CONFIG (PROPOSTA)</span>
              <small>Clique nas linhas para pedir ajustes à Veronica</small>
            </div>
            <div className="column-lines">
              {diffLines.map((line, idx) => {
                const isSelected =
                  line.rightLineNum !== undefined &&
                  selectedStartLine !== null &&
                  selectedEndLine !== null &&
                  line.rightLineNum >= selectedStartLine &&
                  line.rightLineNum <= selectedEndLine;

                return (
                  <div
                    key={`right-${idx}`}
                    className={`diff-line-row line-type-${line.type === 'added' ? 'added' : 'neutral'} ${
                      isSelected ? 'line-selected' : ''
                    }`}
                    onClick={e => line.rightLineNum && handleLineClick(line.rightLineNum, e)}
                  >
                    <span className="line-num">{line.rightLineNum || ''}</span>
                    <pre className="line-content">{line.rightText || ''}</pre>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
