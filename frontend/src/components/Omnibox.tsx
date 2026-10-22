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
  AlertCircle,
  Code,
  FileCode,
  Binary,
  Upload,
  Cpu,
  Database,
  Layers,
  Calculator
} from 'lucide-react';
import type { DeviceItem, KnowledgeDocumentItem, AttachedChip, AttachedChipType } from '../types/veronica';
import { detectLanguage, SUPPORTED_LANGUAGES, SupportedLanguage } from '../utils/languageDetector';

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
    cmd: '/run',
    label: 'Executar no Sandbox',
    desc: 'Execução hermética de código em Python / Polyglot Sandbox com stdout, stderr e AST',
    badge: 'Sandbox',
  },
  {
    cmd: '/codigo',
    label: 'Codex & Refatoração',
    desc: 'Geração de código de alta performance em Rust, Python, TS, C++, Go e SQL',
    badge: 'Polyglot',
  },
  {
    cmd: '/matematica',
    label: 'Matemática & Ciências',
    desc: 'Resolução de cálculo, equações diferenciais, álgebra linear e física computacional',
    badge: 'Math',
  },
  {
    cmd: '/dados',
    label: 'Análise de Dados',
    desc: 'Processamento de datasets, estatística descritiva e telemetria temporal',
    badge: 'Data',
  },
  {
    cmd: '/diagnostico',
    label: 'Diagnóstico Read-Only',
    desc: 'Investigação de falhas de sistemas, flaps de rede e análise de logs',
    badge: 'Tier 1',
  },
  {
    cmd: '/mudanca',
    label: 'Planejamento Determinístico',
    desc: 'Iniciar fluxo de 10 etapas com blast radius e aprovação auditada',
    badge: 'Stage 1-10',
  },
  {
    cmd: '/simular',
    label: 'Simulação de Sistemas',
    desc: 'Simulação de falhas, convergência de grafos e impacto transitivo',
    badge: 'Simulation',
  },
  {
    cmd: '/benchmark',
    label: 'Benchmark de Performance',
    desc: 'Medição de tokens/s, latência de inferência e consumo de memória',
    badge: 'Benchmark',
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
  placeholder = 'Pergunte qualquer questão de código, matemática, dados ou redes... Arraste arquivos ou use / para comandos',
  externalChips = [],
  onRemoveExternalChip,
}) => {
  const [chips, setChips] = useState<AttachedChip[]>([]);
  const [activeChipPreview, setActiveChipPreview] = useState<AttachedChip | null>(null);
  const [autocompleteOpen, setAutocompleteOpen] = useState(false);
  const [autocompleteType, setAutocompleteType] = useState<'command' | 'mention' | null>(null);
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [isDraggingOver, setIsDraggingOver] = useState(false);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
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
        tag: 'Ativo / Infra',
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

  // Helper to process uploaded or pasted file items
  const processFiles = async (fileList: FileList | File[]) => {
    const newChips: AttachedChip[] = [];

    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      const filename = file.name;
      const extension = filename.includes('.') ? `.${filename.split('.').pop()?.toLowerCase()}` : '';

      try {
        if (filename.toLowerCase().endsWith('.pdf')) {
          // PDF preview chip with size
          const sizeKb = (file.size / 1024).toFixed(1);
          newChips.push({
            id: `chip-pdf-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
            title: `${filename} (${sizeKb} KB)`,
            type: 'doc',
            extension: '.pdf',
            language: 'pdf',
            content: `[DOCUMENTO PDF ANEXADO: ${filename} - Tamanho: ${sizeKb} KB - O documento foi carregado e está pronto para análise textual e extração semântica.]`,
            lineCount: 1,
            sizeBytes: file.size,
          });
        } else {
          // Read as text
          const text = await file.text();
          const lines = text.split('\n');
          const detectedLang = detectLanguage(filename, text);

          let chipType: AttachedChipType = 'code';
          if (extension === '.md' || extension === '.txt') chipType = 'doc';
          else if (extension === '.json' || extension === '.csv') chipType = 'data';
          else if (detectedLang === 'network-cli') chipType = 'config';

          newChips.push({
            id: `chip-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
            title: filename,
            type: chipType,
            language: detectedLang,
            extension,
            content: text,
            lineCount: lines.length,
            sizeBytes: file.size,
          });
        }
      } catch (err) {
        console.error('Falha ao processar arquivo:', filename, err);
      }
    }

    if (newChips.length > 0) {
      setChips(prev => [...prev, ...newChips]);
      if (!draft.trim()) {
        setDraft('Analise o arquivo anexado: ');
      }
    }
  };

  // Handle native file selection
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      void processFiles(e.target.files);
      e.target.value = '';
    }
  };

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      void processFiles(e.dataTransfer.files);
    }
  };

  // Detect and transform extensive paste of code, configs, or logs (> 5 lines)
  const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const pastedText = e.clipboardData.getData('text');
    if (!pastedText) return;

    const lines = pastedText.split('\n');
    if (lines.length > 5) {
      const detectedLang = detectLanguage('', pastedText);

      // Check if it's substantial code, config, or log
      const isCodeOrConfig =
        detectedLang !== 'markdown' ||
        pastedText.includes('def ') ||
        pastedText.includes('class ') ||
        pastedText.includes('fn ') ||
        pastedText.includes('import ') ||
        pastedText.includes('SELECT ') ||
        pastedText.includes('interface ') ||
        pastedText.includes('router bgp');

      if (isCodeOrConfig) {
        e.preventDefault();

        const langMeta = SUPPORTED_LANGUAGES[detectedLang];
        const title = `Snippet ${langMeta?.label || detectedLang} (${lines.length} linhas)`;

        const newChip: AttachedChip = {
          id: `chip-paste-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
          title,
          type: detectedLang === 'network-cli' ? 'config' : 'code',
          language: detectedLang,
          extension: langMeta?.extension,
          content: pastedText,
          lineCount: lines.length,
          sizeBytes: new Blob([pastedText]).size,
        };

        setChips(prev => [...prev, newChip]);

        if (!draft.trim()) {
          setDraft(`Analise e otimize o código ${langMeta?.label || detectedLang} anexado: `);
        }
      }
    }
  };

  const insertSuggestion = (insertion: string) => {
    if (!textareaRef.current) return;
    const cursorPos = textareaRef.current.selectionStart;
    const textBefore = draft.slice(0, cursorPos);
    const textAfter = draft.slice(cursorPos);

    const replacedTextBefore = textBefore.replace(/(?:^|\s)([/@][a-zA-Z0-9_-]*)$/, (match) => {
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
    <div
      className={`omnibox-container ${isDraggingOver ? 'dragging-over' : ''}`}
      ref={containerRef}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Hidden File Input for Paperclip / Anexar button */}
      <input
        type="file"
        ref={fileInputRef}
        multiple
        accept=".py,.ts,.tsx,.js,.jsx,.rs,.cpp,.c,.h,.hpp,.go,.sql,.json,.txt,.md,.pdf,.yaml,.yml,.sh,.html,.css,.diff,.patch"
        onChange={handleFileInputChange}
        style={{ display: 'none' }}
      />

      {/* Autocomplete Dropdown Popup */}
      {autocompleteOpen && activeOptionsCount > 0 && (
        <div className="omnibox-dropdown animate__animated animate__fadeInUp">
          <div className="omnibox-dropdown-header">
            <span>
              {autocompleteType === 'command' ? (
                <>
                  <Terminal size={13} /> Comandos &amp; Ações da Veronica
                </>
              ) : (
                <>
                  <Server size={13} /> Contexto, Ativos &amp; Documentação
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
            setDraft(prev => `${prev ? prev + ' ' : ''}/run `);
            textareaRef.current?.focus();
          }}
          title="Executar código hermético no Sandbox Python / Polyglot"
        >
          <Zap size={12} className="text-amber" /> /run Sandbox
        </button>
        <button
          type="button"
          className="quick-chip"
          onClick={() => {
            setDraft(prev => `${prev ? prev + ' ' : ''}/codigo `);
            textareaRef.current?.focus();
          }}
          title="Gerar e refatorar código com tipagem estrita"
        >
          <Code size={12} className="text-mint" /> /codigo Codex
        </button>
        <button
          type="button"
          className="quick-chip"
          onClick={() => {
            setDraft(prev => `${prev ? prev + ' ' : ''}/matematica `);
            textareaRef.current?.focus();
          }}
          title="Cálculo simbólico, equações e física computacional"
        >
          <Calculator size={12} className="text-blue" /> /matematica
        </button>
        <button
          type="button"
          className="quick-chip"
          onClick={() => {
            setDraft(prev => `${prev ? prev + ' ' : ''}/diagnostico `);
            textareaRef.current?.focus();
          }}
          title="Diagnóstico avançado e investigação de telemetria"
        >
          <ShieldCheck size={12} /> /diagnostico
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
          <Layers size={12} /> /mudanca
        </button>

        <div className="quick-chip-hint">
          <span>Arraste código</span> · <span>.py .ts .rs .cpp .sql .pdf</span>
        </div>
      </div>

      {/* Attached Chips Area (uploaded/pasted code, configs, docs) */}
      {allChips.length > 0 && (
        <div className="attached-chips-row">
          <div className="attached-chips-label">
            <Paperclip size={12} /> Anexos ({allChips.length}):
          </div>
          <div className="attached-chips-list">
            {allChips.map(chip => (
              <div key={chip.id} className={`compact-chip type-${chip.type}`}>
                <span className="chip-icon">
                  {chip.type === 'code' ? (
                    <FileCode size={13} />
                  ) : chip.type === 'data' ? (
                    <Database size={13} />
                  ) : chip.type === 'doc' ? (
                    <FileText size={13} />
                  ) : (
                    <Terminal size={13} />
                  )}
                </span>
                <span className="chip-name" title={chip.title}>
                  {chip.title}
                </span>
                {chip.language && (
                  <span className="chip-lang-badge">{chip.language}</span>
                )}
                <button
                  type="button"
                  className="chip-btn preview"
                  title="Inspecionar conteúdo do anexo"
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
            {/* Attachment Button */}
            <button
              type="button"
              className="btn-attach-file"
              onClick={() => fileInputRef.current?.click()}
              title="Anexar arquivos de código (.py, .ts, .rs, .cpp, .sql) ou documentos (.txt, .md, .pdf, .json)"
            >
              <Paperclip size={14} /> Anexar Código / Doc
            </button>
            <span className="safety-indicator">
              <ShieldCheck size={13} /> Sandbox hermético ativo · Tipagem estrita &amp; AST
            </span>
          </div>

          <div className="omnibox-footer-actions">
            <button
              type="button"
              className={`send-action-btn ${busy ? 'btn-abort' : 'btn-send'}`}
              onClick={handleSend}
              disabled={!busy && !draft.trim() && allChips.length === 0}
              aria-label={busy ? 'Interromper geração' : 'Enviar mensagem'}
              title={busy ? 'Interromper processamento da Veronica' : 'Enviar mensagem'}
            >
              {busy ? <Square size={14} fill="currentColor" /> : <Send size={15} />}
            </button>
          </div>
        </div>
      </div>

      <div className="omnibox-disclaimer">
        Veronica Universal Intelligence · Raciocínio poliglota, sandbox hermético e engenharia auditada.
      </div>

      {/* Chip Preview Modal */}
      {activeChipPreview && (
        <div className="chip-preview-overlay" onClick={() => setActiveChipPreview(null)}>
          <div className="chip-preview-modal" onClick={e => e.stopPropagation()}>
            <div className="chip-modal-header">
              <div className="chip-modal-title">
                <FileCode size={16} />
                <b>{activeChipPreview.title}</b>
                <span className="chip-modal-tag">{activeChipPreview.language?.toUpperCase() || activeChipPreview.type.toUpperCase()}</span>
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
              <small>Este arquivo será enviado de forma estruturada para contextualizar a Veronica.</small>
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
