import React, { useState, useRef, useEffect, useMemo } from 'react';
import {
  FileText,
  Server,
  Terminal,
  Paperclip,
  X,
  Eye,
  Check,
  Send,
  Square,
  Sparkles,
  Zap,
  ShieldCheck,
  AlertCircle
} from 'lucide-react';
import type { DeviceItem, KnowledgeDocumentItem, AttachedChip } from '../types/veronica';

interface OmniboxProps {
  draft: string;
  setDraft: React.Dispatch<React.SetStateAction<string>>;
  busy: boolean;
  onSend: (content: string, chips: AttachedChip[]) => void;
  onCancel: () => void;
  devices: DeviceItem[];
  documents: KnowledgeDocumentItem[];
  placeholder?: string;
  externalChips?: AttachedChip[];
  onRemoveExternalChip?: (id: string) => void;
}

const COMMAND_LIST = [
  {
    cmd: '/diagnostico',
    label: 'Diagnóstico Read-Only',
    desc: 'Investigação de incidentes, flaps de BGP e telemetria (Tier 1)',
    badge: 'Tier 1',
  },
  {
    cmd: '/mudanca',
    label: 'Planejamento de Mudança',
    desc: 'Iniciar fluxo determinístico de 10 etapas com blast radius e aprovação',
    badge: 'Stage 1-10',
  },
  {
    cmd: '/simular-falha',
    label: 'Simular Falha',
    desc: 'Simular queda de enlace transitivo e avaliar convergência de rotas',
    badge: 'Simulação',
  },
  {
    cmd: '/benchmark',
    label: 'Benchmark Neural',
    desc: 'Avaliar tempo de inferência, taxa de tokens/s e precisão de raciocínio',
    badge: 'Performance',
  },
  {
    cmd: '/simular',
    label: 'Simulação Topológica',
    desc: 'Calcular raio de impacto direto e transitivo em grafos de rede',
    badge: 'Blast Radius',
  },
];

export const Omnibox: React.FC<OmniboxProps> = ({
  draft,
  setDraft,
  busy,
  onSend,
  onCancel,
  devices,
  documents,
  placeholder = 'Pergunte sobre sua rede, use @ para ativos/docs ou / para comandos...',
  externalChips = [],
  onRemoveExternalChip,
}) => {
  const [chips, setChips] = useState<AttachedChip[]>([]);
  const [activeChipPreview, setActiveChipPreview] = useState<AttachedChip | null>(null);
  const [autocompleteOpen, setAutocompleteOpen] = useState(false);
  const [autocompleteType, setAutocompleteType] = useState<'command' | 'mention' | null>(null);
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Combine local chips and external chips (if any)
  const allChips = useMemo(() => [...chips, ...externalChips], [chips, externalChips]);

  // Autocomplete filtering
  const filteredCommands = useMemo(() => {
    if (autocompleteType !== 'command') return [];
    const q = query.toLowerCase().replace(/^\//, '');
    return COMMAND_LIST.filter(c => c.cmd.toLowerCase().includes(q) || c.label.toLowerCase().includes(q));
  }, [autocompleteType, query]);

  const filteredMentions = useMemo(() => {
    if (autocompleteType !== 'mention') return [];
    const q = query.toLowerCase().replace(/^@/, '');
    const matchedDevices = devices
      .filter(d => d.device_id.toLowerCase().includes(q) || d.vendor.toLowerCase().includes(q) || d.management_ip.includes(q))
      .map(d => ({
        type: 'device' as const,
        id: `@${d.device_id}`,
        label: d.device_id,
        sub: `${d.vendor} · ${d.management_ip}`,
        tag: 'Equipamento',
      }));

    const matchedDocs = documents
      .filter(doc => doc.title.toLowerCase().includes(q) || (doc.vendor && doc.vendor.toLowerCase().includes(q)) || (doc.feature && doc.feature.toLowerCase().includes(q)))
      .map(doc => ({
        type: 'doc' as const,
        id: `@${doc.doc_id || doc.title.toLowerCase().replace(/[^a-z0-9_-]/g, '-').slice(0, 24)}`,
        label: doc.title,
        sub: `${doc.vendor || 'RFC'} · ${doc.os_name || 'Standard'}`,
        tag: 'Doc Metadados',
      }));

    return [...matchedDevices, ...matchedDocs];
  }, [autocompleteType, query, devices, documents]);

  const activeOptionsCount = autocompleteType === 'command' ? filteredCommands.length : filteredMentions.length;

  useEffect(() => {
    setSelectedIndex(0);
  }, [autocompleteType, query]);

  // Handle typing & trigger detection
  const handleDraftChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setDraft(val);

    const cursorPos = e.target.selectionStart;
    const textBeforeCursor = val.slice(0, cursorPos);
    const lastWordMatch = textBeforeCursor.match(/(?:^|\s)([/@][a-zA-Z0-9_-]*)$/);

    if (lastWordMatch) {
      const matchText = lastWordMatch[1];
      if (matchText.startsWith('/')) {
        setAutocompleteType('command');
        setQuery(matchText);
        setAutocompleteOpen(true);
      } else if (matchText.startsWith('@')) {
        setAutocompleteType('mention');
        setQuery(matchText);
        setAutocompleteOpen(true);
      }
    } else {
      setAutocompleteOpen(false);
      setAutocompleteType(null);
    }
  };

  // Detect and transform extensive paste of configs or logs (> 5 lines)
  const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const pastedText = e.clipboardData.getData('text');
    if (!pastedText) return;

    const lines = pastedText.split('\n');
    const isConfigOrLog =
      lines.length > 5 ||
      pastedText.includes('interface ') ||
      pastedText.includes('router bgp') ||
      pastedText.includes('syslog') ||
      pastedText.includes('%SYS-') ||
      pastedText.includes('show ip ') ||
      lines.some(l => /^\s*(set|set\s+system|show|display|ip\s+route|neighbor)\s+/i.test(l));

    if (lines.length > 5 && isConfigOrLog) {
      e.preventDefault();

      // Guess type and name
      let chipType: 'config' | 'log' = 'config';
      let title = 'Configuração de Rede';
      if (pastedText.includes('syslog') || pastedText.includes('%') || pastedText.includes('INFO') || pastedText.includes('ERROR')) {
        chipType = 'log';
        title = 'Log do Sistema';
      } else if (pastedText.includes('bgp') || pastedText.includes('BGP')) {
        title = 'Bloco BGP';
      } else if (pastedText.includes('interface')) {
        title = 'Configuração de Interfaces';
      }

      const newChip: AttachedChip = {
        id: `chip-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        title: `${title} (${lines.length} linhas)`,
        type: chipType,
        content: pastedText,
        lineCount: lines.length,
        sizeBytes: new Blob([pastedText]).size,
      };

      setChips(prev => [...prev, newChip]);

      // If textarea is empty, leave a friendly hint
      if (!draft.trim()) {
        setDraft('Analise o bloco anexado ');
      }
    }
  };

  const insertSuggestion = (insertion: string) => {
    if (!textareaRef.current) return;
    const cursorPos = textareaRef.current.selectionStart;
    const textBefore = draft.slice(0, cursorPos);
    const textAfter = draft.slice(cursorPos);

    const replacedTextBefore = textBefore.replace(/(?:^|\s)([/@][a-zA-Z0-9_-]*)$/, (match, prefix) => {
      const leadingSpace = match.startsWith(' ') ? ' ' : '';
      return `${leadingSpace}${insertion} `;
    });

    const newDraft = replacedTextBefore + textAfter;
    setDraft(newDraft);
    setAutocompleteOpen(false);

    setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.focus();
        const nextPos = replacedTextBefore.length;
        textareaRef.current.setSelectionRange(nextPos, nextPos);
      }
    }, 10);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (autocompleteOpen && activeOptionsCount > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex(prev => (prev + 1) % activeOptionsCount);
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex(prev => (prev - 1 + activeOptionsCount) % activeOptionsCount);
        return;
      }
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        if (autocompleteType === 'command') {
          const item = filteredCommands[selectedIndex];
          if (item) insertSuggestion(item.cmd);
        } else if (autocompleteType === 'mention') {
          const item = filteredMentions[selectedIndex];
          if (item) insertSuggestion(item.id);
        }
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        setAutocompleteOpen(false);
        return;
      }
    }

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if (busy) {
      onCancel();
      return;
    }
    const trimmed = draft.trim();
    if (!trimmed && allChips.length === 0) return;

    onSend(trimmed, allChips);
    setChips([]);
  };

  const removeChip = (id: string) => {
    setChips(prev => prev.filter(c => c.id !== id));
    if (onRemoveExternalChip) {
      onRemoveExternalChip(id);
    }
  };

  return (
    <div className="omnibox-container" ref={containerRef}>
      {/* Autocomplete Dropdown Popup */}
      {autocompleteOpen && activeOptionsCount > 0 && (
        <div className="omnibox-dropdown animate__animated animate__fadeInUp">
          <div className="omnibox-dropdown-header">
            <span>
              {autocompleteType === 'command' ? (
                <>
                  <Terminal size={13} /> Comandos Rápidos da Veronica
                </>
              ) : (
                <>
                  <Server size={13} /> Inventário Multi-Vendor & Documentação
                </>
              )}
            </span>
            <small>↑↓ navegar · [Enter] / [Tab] selecionar</small>
          </div>

          <div className="omnibox-dropdown-list">
            {autocompleteType === 'command' &&
              filteredCommands.map((item, idx) => (
                <div
                  key={item.cmd}
                  className={`omnibox-dropdown-item ${idx === selectedIndex ? 'selected' : ''}`}
                  onClick={() => insertSuggestion(item.cmd)}
                  onMouseEnter={() => setSelectedIndex(idx)}
                >
                  <div className="item-icon cmd-icon">/</div>
                  <div className="item-body">
                    <div className="item-title">
                      <b>{item.cmd}</b>
                      <span>{item.label}</span>
                      <span className="omnibox-item-badge">{item.badge}</span>
                    </div>
                    <div className="item-desc">{item.desc}</div>
                  </div>
                </div>
              ))}

            {autocompleteType === 'mention' &&
              filteredMentions.map((item, idx) => (
                <div
                  key={`${item.type}-${item.id}`}
                  className={`omnibox-dropdown-item ${idx === selectedIndex ? 'selected' : ''}`}
                  onClick={() => insertSuggestion(item.id)}
                  onMouseEnter={() => setSelectedIndex(idx)}
                >
                  <div className={`item-icon ${item.type === 'device' ? 'device-icon' : 'doc-icon'}`}>
                    {item.type === 'device' ? <Server size={14} /> : <FileText size={14} />}
                  </div>
                  <div className="item-body">
                    <div className="item-title">
                      <b>{item.id}</b>
                      <span className="omnibox-item-badge">{item.tag}</span>
                    </div>
                    <div className="item-desc">{item.sub}</div>
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Quick Command Pills above Omnibox */}
      <div className="quick-command-bar">
        <button
          type="button"
          className="quick-chip"
          onClick={() => {
            setDraft(prev => `${prev ? prev + ' ' : ''}/diagnostico `);
            textareaRef.current?.focus();
          }}
          title="Executar diagnóstico somente leitura (Tier 1)"
        >
          <Zap size={12} /> /diagnostico
        </button>
        <button
          type="button"
          className="quick-chip"
          onClick={() => {
            setDraft(prev => `${prev ? prev + ' ' : ''}/mudanca `);
            textareaRef.current?.focus();
          }}
          title="Planejar mudança determinística de 10 etapas"
        >
          <ShieldCheck size={12} /> /mudanca
        </button>
        <button
          type="button"
          className="quick-chip"
          onClick={() => {
            setDraft(prev => `${prev ? prev + ' ' : ''}/simular-falha `);
            textareaRef.current?.focus();
          }}
          title="Simulação de failover e blast radius"
        >
          <AlertCircle size={12} /> /simular-falha
        </button>
        <button
          type="button"
          className="quick-chip"
          onClick={() => {
            setDraft(prev => `${prev ? prev + ' ' : ''}/benchmark `);
            textareaRef.current?.focus();
          }}
          title="Benchmark de inferência neural local"
        >
          <Sparkles size={12} /> /benchmark
        </button>
        <div className="quick-chip-hint">
          <span>@equipamento</span> · <span>@documento</span>
        </div>
      </div>

      {/* Attached Chips Area (pasted configs/logs) */}
      {allChips.length > 0 && (
        <div className="attached-chips-row">
          <div className="attached-chips-label">
            <Paperclip size={12} /> Anexos Compactos ({allChips.length}):
          </div>
          <div className="attached-chips-list">
            {allChips.map(chip => (
              <div key={chip.id} className={`compact-chip type-${chip.type}`}>
                <span className="chip-icon">
                  {chip.type === 'config' ? <FileText size={13} /> : <Terminal size={13} />}
                </span>
                <span className="chip-name" title={chip.title}>
                  {chip.title}
                </span>
                <button
                  type="button"
                  className="chip-btn preview"
                  title="Inspecionar conteúdo colado"
                  onClick={() => setActiveChipPreview(chip)}
                >
                  <Eye size={12} />
                </button>
                <button
                  type="button"
                  className="chip-btn remove"
                  title="Remover anexo"
                  onClick={() => removeChip(chip.id)}
                >
                  <X size={12} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Main Omnibox Textarea and Footer */}
      <div className="omnibox-box">
        <textarea
          ref={textareaRef}
          value={draft}
          onChange={handleDraftChange}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          placeholder={placeholder}
          rows={Math.min(6, Math.max(1, draft.split('\n').length))}
          disabled={busy}
        />

        <div className="omnibox-footer">
          <div className="omnibox-footer-info">
            <span className="safety-indicator">
              <ShieldCheck size={13} /> Somente leitura ativa · Alterações exigem aprovação humana
            </span>
          </div>

          <div className="omnibox-footer-actions">
            <button
              type="button"
              className={`send-action-btn ${busy ? 'btn-abort' : 'btn-send'}`}
              onClick={handleSend}
              disabled={!busy && !draft.trim() && allChips.length === 0}
              aria-label={busy ? 'Interromper geração' : 'Enviar mensagem'}
              title={busy ? 'Interromper processamento da Veronica' : 'Enviar comando'}
            >
              {busy ? <Square size={14} fill="currentColor" /> : <Send size={15} />}
            </button>
          </div>
        </div>
      </div>

      <div className="omnibox-disclaimer">
        Veronica Transformer · Operação de rede estritamente segura e auditada.
      </div>

      {/* Chip Preview Modal */}
      {activeChipPreview && (
        <div className="chip-preview-overlay" onClick={() => setActiveChipPreview(null)}>
          <div className="chip-preview-modal" onClick={e => e.stopPropagation()}>
            <div className="chip-modal-header">
              <div className="chip-modal-title">
                <FileText size={16} />
                <b>{activeChipPreview.title}</b>
                <span className="chip-modal-tag">{activeChipPreview.type.toUpperCase()}</span>
                <span className="chip-modal-count">{activeChipPreview.lineCount} linhas</span>
              </div>
              <button
                type="button"
                className="chip-modal-close"
                onClick={() => setActiveChipPreview(null)}
              >
                <X size={16} />
              </button>
            </div>
            <div className="chip-modal-content">
              <pre>{activeChipPreview.content}</pre>
            </div>
            <div className="chip-modal-footer">
              <small>Este bloco será enviado de forma estruturada para contextualizar a Veronica.</small>
              <button
                type="button"
                className="btn-primary-sm"
                onClick={() => setActiveChipPreview(null)}
              >
                <Check size={14} /> Fechar Prévia
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
