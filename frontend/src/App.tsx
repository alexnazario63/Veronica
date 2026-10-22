import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  ArrowDown,
  ArrowUp,
  Bot,
  BrainCircuit,
  Check,
  ChevronDown,
  CircleHelp,
  ClipboardList,
  Command,
  FileText,
  Gauge,
  GitBranch,
  Menu,
  Moon,
  Network,
  Plus,
  Search,
  Settings2,
  ShieldCheck,
  Square,
  Sun,
  Trash2,
  Wrench,
  Sparkles,
  Building2,
  Maximize2,
  Code,
  Zap
} from 'lucide-react';
import { create } from 'zustand';
import type {
  ArchitectureNode,
  Artifact,
  AttachedChip,
  CitationMetadata,
  DeviceItem,
  KnowledgeDocumentItem,
  StoredArtifact,
  StructuredAssistantPayload,
  TrainingStepMetric
} from './types/veronica';

import { Omnibox } from './components/Omnibox';
import { ReasoningAccordion } from './components/ReasoningAccordion';
import { EpistemicBadges } from './components/EpistemicBadges';
import { NetCanvas } from './components/NetCanvas';
import { TenantSelector } from './components/TenantSelector';

const Admin = lazy(() => import('./Admin'));

type View = 'work' | 'observatory' | 'admin';
type ModelProfile = 'vrn-fast' | 'vrn-pro' | 'vrn-ultra';
type ResponseStyle = 'technical' | 'natural' | 'mixed' | 'engineer';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  payload?: StructuredAssistantPayload;
  profile?: ModelProfile;
  style?: ResponseStyle;
  thoughts?: string[];
  tools?: string[];
}

interface Conversation {
  conversation_id: string;
  title: string;
}

interface Store {
  view: View;
  setView: (view: View) => void;
  artifact: Artifact | null;
  setArtifact: (artifact: Artifact | null) => void;
  tenantId: string;
  setTenantId: (id: string) => void;
}

const viewFromPath = (): View => (window.location.pathname === '/admin' ? 'admin' : 'work');

export const useUi = create<Store>((set) => ({
  view: viewFromPath(),
  setView: (view) => {
    const path = view === 'admin' ? '/admin' : '/';
    if (window.location.pathname !== path) window.history.pushState({}, '', path);
    set({ view });
  },
  artifact: null,
  setArtifact: (artifact) => set({ artifact }),
  tenantId: (() => {
    try {
      return localStorage.getItem('veronica-tenant-id') || 'tenant-default';
    } catch {
      return 'tenant-default';
    }
  })(),
  setTenantId: (tenantId) => {
    try {
      localStorage.setItem('veronica-tenant-id', tenantId);
    } catch {
      // storage unavailable
    }
    set({ tenantId });
  },
}));

export const api = async <T,>(path: string, init?: RequestInit): Promise<T> => {
  const tenant = useUi.getState().tenantId;
  const response = await fetch(path, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      'X-User-Id': 'usr-eng-01',
      'X-Tenant-Id': tenant,
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `API ${response.status}`);
  }
  return response.json();
};

export default function VeronicaApp() {
  const { view, setView, artifact, setArtifact, tenantId, setTenantId } = useUi();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversationId, setConversationId] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [devices, setDevices] = useState<DeviceItem[]>([]);
  const [documents, setDocuments] = useState<KnowledgeDocumentItem[]>([]);
  const [workspaceArtifacts, setWorkspaceArtifacts] = useState<StoredArtifact[]>([]);
  const [artifactVersions, setArtifactVersions] = useState<StoredArtifact[]>([]);
  const [draft, setDraft] = useState('');
  const [mode, setMode] = useState<'chat' | 'agent'>('agent');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [thoughts, setThoughts] = useState<string[]>([]);
  const [tools, setTools] = useState<string[]>([]);
  const [historyOpen, setHistoryOpen] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);

  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    try {
      return localStorage.getItem('veronica-theme') === 'dark' ? 'dark' : 'light';
    } catch {
      return 'light';
    }
  });

  const [modelProfile, setModelProfile] = useState<ModelProfile>(() => {
    try {
      const saved = localStorage.getItem('veronica-model-profile');
      return saved === 'vrn-fast' || saved === 'vrn-ultra' ? saved : 'vrn-pro';
    } catch {
      return 'vrn-pro';
    }
  });

  const [responseStyle, setResponseStyle] = useState<ResponseStyle>(() => {
    try {
      const saved = localStorage.getItem('veronica-response-style');
      return saved === 'natural' || saved === 'mixed' || saved === 'engineer' ? saved : 'technical';
    } catch {
      return 'technical';
    }
  });

  const aborter = useRef<AbortController | null>(null);
  const chatScrollRef = useRef<HTMLElement | null>(null);
  const stickToBottom = useRef(true);
  const restoredLatestConversation = useRef(false);

  const activeConversation = conversations.find((item) => item.conversation_id === conversationId);

  useEffect(() => {
    const syncRoute = () => setView(viewFromPath());
    window.addEventListener('popstate', syncRoute);
    return () => window.removeEventListener('popstate', syncRoute);
  }, [setView]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem('veronica-theme', theme);
    } catch {
      /* storage can be unavailable */
    }
  }, [theme]);

  useEffect(() => {
    try {
      localStorage.setItem('veronica-model-profile', modelProfile);
      localStorage.setItem('veronica-response-style', responseStyle);
    } catch {
      /* storage can be unavailable */
    }
  }, [modelProfile, responseStyle]);

  const refresh = async () => {
    try {
      const [c, inventory, docsRes, saved] = await Promise.all([
        api<{ conversations: Conversation[] }>('/api/conversations'),
        api<{ devices: DeviceItem[] }>('/api/network/inventory').catch(() => ({
          devices: [
            { device_id: 'cisco-iosxr-01', vendor: 'Cisco', management_ip: '10.0.0.1' },
            { device_id: 'juniper-mx-02', vendor: 'Juniper', management_ip: '10.0.0.2' },
            { device_id: 'arista-eos-03', vendor: 'Arista', management_ip: '10.0.0.3' },
            { device_id: 'huawei-vrp-04', vendor: 'Huawei', management_ip: '10.0.0.4' },
            { device_id: 'nokia-sr-05', vendor: 'Nokia', management_ip: '10.0.0.5' },
          ],
        })),
        api<{ documents: KnowledgeDocumentItem[] }>('/api/knowledge/documents').catch(() => ({
          documents: [
            {
              doc_id: 'doc-rfc7432',
              title: 'RFC 7432: BGP MPLS-Based Ethernet VPN',
              vendor: 'IETF',
              os_name: 'Standard',
              feature: 'BGP EVPN',
              reliability_score: 0.99,
            },
            {
              doc_id: 'doc-cisco-routing',
              title: 'Cisco IOS-XR Routing Configuration Guide',
              vendor: 'Cisco',
              os_name: 'IOS-XR',
              version_build: '7.3.2',
              feature: 'BGP / MPLS',
              reliability_score: 0.98,
            },
            {
              doc_id: 'doc-junos-peering',
              title: 'Juniper Junos OS BGP Peering and Policy',
              vendor: 'Juniper',
              os_name: 'Junos OS',
              version_build: '21.4R1',
              feature: 'BGP Policy',
              reliability_score: 0.97,
            },
            {
              doc_id: 'doc-telecom-security',
              title: 'Norma Operacional de Mudanças em Core Telecom (ADR-03 / R-09)',
              vendor: 'Telecom Corp',
              os_name: 'Corporate Policy',
              feature: 'Change Control',
              reliability_score: 1.0,
            },
          ],
        })),
        api<{ artifacts: StoredArtifact[] }>('/api/workspace/artifacts').catch(() => ({ artifacts: [] })),
      ]);

      const rows = c.conversations || [];
      setConversations(rows);
      setDevices(inventory.devices || []);
      setDocuments(docsRes.documents || []);
      setWorkspaceArtifacts(saved.artifacts || []);

      if (!restoredLatestConversation.current) {
        restoredLatestConversation.current = true;
        if (rows[0]) void selectConversation(rows[0].conversation_id);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'API indisponível');
    }
  };

  useEffect(() => {
    void refresh();
  }, [tenantId]);

  useEffect(() => {
    if (!stickToBottom.current) return;
    const frame = requestAnimationFrame(() => {
      const scroller = chatScrollRef.current;
      if (scroller) scroller.scrollTop = scroller.scrollHeight;
    });
    return () => cancelAnimationFrame(frame);
  }, [messages, thoughts, tools]);

  function onChatScroll() {
    const scroller = chatScrollRef.current;
    if (!scroller) return;
    const atBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 120;
    stickToBottom.current = atBottom;
    setShowScrollToBottom(!atBottom && messages.length > 0);
  }

  function scrollToLatest() {
    stickToBottom.current = true;
    setShowScrollToBottom(false);
    chatScrollRef.current?.scrollTo({ top: chatScrollRef.current.scrollHeight, behavior: 'smooth' });
  }

  async function selectConversation(id: string) {
    setError('');
    setSidebarOpen(false);
    stickToBottom.current = true;
    setShowScrollToBottom(false);
    try {
      const data = await api<{
        messages?: {
          role: string;
          content: string;
          structured_payload?: StructuredAssistantPayload & { model_profile?: string; response_style?: string };
        }[];
      }>(`/api/conversations/${id}`);
      setConversationId(id);
      setMessages(
        (data.messages || []).map((m) => ({
          role: m.role === 'user' ? 'user' : 'assistant',
          content: m.content,
          payload: m.structured_payload,
          profile: ['vrn-fast', 'vrn-pro', 'vrn-ultra'].includes(m.structured_payload?.model_profile || '')
            ? (m.structured_payload?.model_profile as ModelProfile)
            : undefined,
          style: ['technical', 'natural', 'mixed', 'engineer'].includes(m.structured_payload?.response_style || '')
            ? (m.structured_payload?.response_style as ResponseStyle)
            : undefined,
        }))
      );
    } catch (e) {
      setConversationId('');
      setMessages([]);
      setError(e instanceof Error ? e.message : 'Não foi possível abrir a conversa');
    }
  }

  async function newConversation() {
    restoredLatestConversation.current = true;
    setConversationId('');
    setMessages([]);
    setThoughts([]);
    setTools([]);
    setShowScrollToBottom(false);
    setSidebarOpen(false);
    try {
      const c = await api<Conversation>('/api/conversations', {
        method: 'POST',
        body: JSON.stringify({ title: 'Nova conversa' }),
      });
      await refresh();
      setConversationId(c.conversation_id);
      setMessages([]);
      stickToBottom.current = true;
      setShowScrollToBottom(false);
      setSidebarOpen(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Não foi possível criar conversa');
    }
  }

  async function deleteConversation(event: React.MouseEvent, conversation: Conversation) {
    event.stopPropagation();
    if (!window.confirm(`Excluir a conversa “${conversation.title || 'Conversa sem título'}”? Esta ação não pode ser desfeita.`))
      return;
    try {
      await api<{ deleted: boolean }>(`/api/conversations/${encodeURIComponent(conversation.conversation_id)}`, {
        method: 'DELETE',
      });
      setConversations((current) => current.filter((item) => item.conversation_id !== conversation.conversation_id));
      if (conversationId === conversation.conversation_id) {
        aborter.current?.abort();
        setConversationId('');
        setMessages([]);
        setThoughts([]);
        setTools([]);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Não foi possível excluir a conversa');
    }
  }

  async function openStoredArtifact(id: string, version: number) {
    try {
      const result = await api<{ versions: StoredArtifact[] }>(
        `/api/workspace/artifacts/${encodeURIComponent(id)}/versions`
      );
      setArtifactVersions(result.versions || []);
      const selected = result.versions.find((row) => row.artifact.version === version)?.artifact;
      if (selected) setArtifact(selected);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Não foi possível abrir o artefato');
    }
  }

  // Handle Omnibox send with serialized chips
  async function handleOmniboxSend(content: string, chips: AttachedChip[]) {
    if ((!content && chips.length === 0) || busy) return;

    // Serialize attached chips neatly for Veronica
    let finalPrompt = content;
    if (chips.length > 0) {
      const attachmentsText = chips
        .map(
          (c) =>
            `\n\n--- [ARQUIVO ANEXADO: ${c.title} (${c.language || c.type}, ${c.lineCount} linhas)] ---\n\`\`\`${c.language || c.type}\n${c.content}\n\`\`\`\n`
        )
        .join('\n');
      finalPrompt = `${content || 'Analise o arquivo anexado:'}${attachmentsText}`;
    }

    const title = conversationTitle(content || (chips[0] ? chips[0].title : 'Conversa'));
    let id = conversationId;

    if (!id) {
      try {
        const c = await api<Conversation>('/api/conversations', {
          method: 'POST',
          body: JSON.stringify({ title }),
        });
        id = c.conversation_id;
        setConversationId(id);
        restoredLatestConversation.current = true;
        await refresh();
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Falha ao criar conversa');
        return;
      }
    } else {
      try {
        await api<Conversation>(`/api/conversations/${encodeURIComponent(id)}`, {
          method: 'PATCH',
          body: JSON.stringify({ title }),
        });
        setConversations((current) => [
          { conversation_id: id!, title },
          ...current.filter((item) => item.conversation_id !== id),
        ]);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Não foi possível atualizar o título da conversa');
        return;
      }
    }

    stickToBottom.current = true;
    setShowScrollToBottom(false);
    setDraft('');
    setError('');
    setThoughts([]);
    setTools([]);

    // Display message shows user content plus chip indicators
    const displayMessage =
      chips.length > 0
        ? `${content ? content + '\n\n' : ''}${chips.map((c) => `[📋 ${c.title}]`).join(' ')}`
        : content;

    setMessages((prev) => [
      ...prev,
      { role: 'user', content: displayMessage, profile: modelProfile, style: responseStyle },
      { role: 'assistant', content: '', profile: modelProfile, style: responseStyle },
    ]);
    setBusy(true);

    const controller = new AbortController();
    aborter.current = controller;

    try {
      const endpoint = mode === 'agent' ? 'agent-stream' : 'stream';
      const response = await fetch(`/api/conversations/${id}/${endpoint}`, {
        method: 'POST',
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          'X-User-Id': 'usr-eng-01',
          'X-Tenant-Id': tenantId,
        },
        body: JSON.stringify({
          content: finalPrompt,
          mode,
          model_profile: modelProfile,
          response_style: responseStyle,
        }),
      });

      if (!response.ok || !response.body) {
        const failure = await response.json().catch(() => ({}));
        throw new Error(failure.detail || `Falha no stream (${response.status})`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split('\n\n');
        buffer = chunks.pop() || '';

        for (const chunk of chunks) {
          const line = chunk.split('\n').find((s) => s.startsWith('data: '));
          if (!line) continue;
          const ev = JSON.parse(line.slice(6));

          if (ev.type === 'error') {
            throw new Error(ev.message || 'Falha ao gerar a resposta');
          } else if (ev.type === 'token') {
            appendAssistant(ev.token_text || ev.token || '');
          } else if (ev.type === 'complete') {
            setMessages((prev) =>
              prev.map((m, i) =>
                i === prev.length - 1
                  ? {
                      ...m,
                      content: ev.content || m.content,
                      payload: ev.structured_payload,
                    }
                  : m
              )
            );
          } else if (ev.type === 'message_delta') {
            appendAssistant(ev.delta || '');
          } else if (ev.type === 'thought_delta') {
            setThoughts((prev) => [...prev, ev.delta || '']);
          } else if (ev.type === 'tool_start') {
            setTools((prev) => [
              ...prev,
              `Iniciado: ${ev.tool_name || 'ferramenta'} · ${ev.tier || 'tier desconhecido'}`,
            ]);
          } else if (ev.type === 'tool_end') {
            setTools((prev) => [
              ...prev,
              `Concluído: ${ev.tool_call_id || 'ferramenta'} · ${ev.status || 'ok'}${
                ev.duration_ms ? ` · ${Math.round(ev.duration_ms)} ms` : ''
              }`,
            ]);
          } else if (ev.type === 'artifact_open' && ev.artifact) {
            setArtifact(ev.artifact);
            void refresh();
          } else if (ev.type === 'artifact_delta') {
            const current = useUi.getState().artifact;
            if (current && current.id === ev.artifact_id) {
              setArtifact({
                ...current,
                version: ev.version || current.version,
                proposed_content: `${current.proposed_content || ''}${ev.delta || ''}`,
              });
            }
          } else if (ev.type === 'safety_gate_interrupt') {
            setTools((prev) => [
              ...prev,
              `Aprovação humana necessária · ${ev.change_stage || ''} · ${ev.reason || ''}`,
            ]);
          }
        }
      }
      await refresh();
    } catch (e) {
      if (!(e instanceof DOMException && e.name === 'AbortError')) {
        setError(e instanceof Error ? e.message : 'Erro ao gerar resposta');
      }
    } finally {
      setBusy(false);
      aborter.current = null;
    }
  }

  function appendAssistant(delta: string) {
    if (!delta) return;
    setMessages((prev) =>
      prev.map((m, i) => (i === prev.length - 1 ? { ...m, content: m.content + delta } : m))
    );
  }

  // Handle "Pedir ajuste à Veronica" from Diff Editor line selection
  const handleAskAdjustment = (startLine: number, endLine: number, snippet: string) => {
    if (!artifact) return;
    const adjustmentPrompt = `Na linha ${startLine} a ${endLine} da configuração proposta [${artifact.type} v${artifact.version}], por favor ajuste:\n`;
    setDraft((prev) => `${prev ? prev + '\n' : ''}${adjustmentPrompt}`);
  };

  // Handle hypothesis one-click diagnostic trigger
  const handleTriggerDiagnostic = (hypothesis: string) => {
    const diagnosticCmd = `/diagnostico Testar hipótese: ${hypothesis}`;
    setDraft(diagnosticCmd);
  };

  // Stage 6 Human Approval API call
  const handleApproveStage6 = async (approvalData: {
    approver_user_id: string;
    confirmed_maintenance_window: string;
    signature_ed25519: string;
    confirmed_plan_hash: string;
  }) => {
    if (!artifact) return;
    await api(`/api/network/changes/${encodeURIComponent(artifact.id)}/approve`, {
      method: 'POST',
      body: JSON.stringify({
        ...approvalData,
        approval_channel: 'EXPLICIT_WEB_APPROVAL',
        approval_timestamp: new Date().toISOString(),
      }),
    });
    setArtifact({
      ...artifact,
      change_stage: '7_CONTROLLED_EXECUTION',
    });
    await refresh();
  };

  // Abort and immediate rollback
  const handleAbortChange = async () => {
    if (!artifact) return;
    await api(`/api/conversations/${conversationId}/cancel`, {
      method: 'POST',
      body: JSON.stringify({ reason: 'Manual abort from ChangeConsole' }),
    }).catch(() => {});
    setArtifact({
      ...artifact,
      change_stage: '9_COMMIT_OR_ROLLBACK',
    });
    await refresh();
  };

  if (view === 'admin') {
    return (
      <Suspense fallback={<div className="admin-loading">Abrindo administração…</div>}>
        <Admin
          onBack={() => setView('work')}
          onToggleTheme={() => setTheme((current) => (current === 'light' ? 'dark' : 'light'))}
          theme={theme}
        />
      </Suspense>
    );
  }

  return (
    <div className={`shell ${sidebarOpen ? 'sidebar-open' : ''}`}>
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <BrainCircuit size={21} />
          </div>
          <div>
            <b>VERONICA</b>
            <small>GENERAL INTELLIGENCE &amp; CODEX</small>
          </div>
        </div>

        <button className="new-chat" onClick={newConversation}>
          <Plus size={17} /> Nova conversa <Command size={14} />
        </button>

        <div className="side-label">Workspace</div>
        <button
          className={`side-link ${view === 'work' ? 'selected' : ''}`}
          onClick={() => setView('work')}
        >
          <Code size={16} /> Chat &amp; Workspace Poliglota
        </button>
        <button
          className={`side-link ${view === 'observatory' ? 'selected' : ''}`}
          onClick={() => setView('observatory')}
        >
          <Activity size={16} /> Neural Observatory
        </button>
        <button className="side-link" onClick={() => setView('admin')}>
          <Settings2 size={16} /> Administração
        </button>

        {/* Tenant Selector in Sidebar */}
        <div className="side-label">Tenant / Cluster (Fase K)</div>
        <div className="sidebar-tenant-wrap">
          <TenantSelector currentTenantId={tenantId} onSelectTenant={setTenantId} />
        </div>

        {/* Conversations List */}
        <button
          className="side-link history-toggle"
          onClick={() => setHistoryOpen(!historyOpen)}
          aria-expanded={historyOpen}
        >
          <ChevronDown size={15} className={historyOpen ? '' : 'rotated'} /> Conversas{' '}
          <span className="count">{conversations.length}</span>
        </button>
        {historyOpen && (
          <div className="conversation-list">
            {conversations.map((c) => (
              <div
                key={c.conversation_id}
                className={`conversation-row ${c.conversation_id === conversationId ? 'selected' : ''}`}
              >
                <button
                  className="conversation"
                  title={c.title || 'Conversa sem título'}
                  onClick={() => void selectConversation(c.conversation_id)}
                >
                  {c.title || 'Conversa sem título'}
                </button>
                <button
                  className="delete-conversation"
                  title="Excluir conversa"
                  aria-label={`Excluir conversa ${c.title || ''}`}
                  onClick={(event) => void deleteConversation(event, c)}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
            {!conversations.length && <small className="muted pad">Sem conversas recentes</small>}
          </div>
        )}

        {/* Quick Devices */}
        <div className="side-label">Ativos Rápidos</div>
        <div className="device-list">
          {devices.slice(0, 5).map((d) => (
            <button
              key={d.device_id}
              className="device"
              onClick={() => setDraft((v) => `${v}${v ? ' ' : ''}@${d.device_id} `)}
            >
              <span className="online-dot" />
              <span>
                {d.device_id}
                <small>
                  {d.vendor} · {d.management_ip}
                </small>
              </span>
            </button>
          ))}
        </div>

        {/* Stored Artifacts */}
        <div className="side-label">Artefatos &amp; Código</div>
        <div className="device-list">
          {workspaceArtifacts.map(({ artifact: item }) => (
            <button
              key={`${item.id}-${item.version}`}
              className="device artifact-link"
              onClick={() => void openStoredArtifact(item.id, item.version)}
            >
              <FileText size={14} />
              <span>
                {item.type}
                <small>
                  v{item.version} · {item.id}
                </small>
              </span>
            </button>
          ))}
          {!workspaceArtifacts.length && <small className="muted pad">Nenhum artefato salvo</small>}
        </div>

        <div className="sidebar-bottom">
          <ShieldCheck size={16} />
          <span>
            Safety Control ativo
            <small>Alterações exigem aprovação humana</small>
          </span>
          <button
            className="sidebar-theme"
            onClick={() => setTheme((current) => (current === 'light' ? 'dark' : 'light'))}
            aria-label={`Ativar tema ${theme === 'light' ? 'escuro' : 'claro'}`}
            title={`Tema ${theme === 'light' ? 'escuro' : 'claro'}`}
          >
            {theme === 'light' ? <Moon size={15} /> : <Sun size={15} />}
          </button>
        </div>
      </aside>

      {/* Main Experience: Split-Screen Chat + WorkUI NetCanvas */}
      {view === 'observatory' ? (
        <Observatory />
      ) : (
        <>
          <main className="main">
            {/* Topbar */}
            <header className="topbar">
              <div>
                <button
                  className="mobile-menu"
                  onClick={() => setSidebarOpen((open) => !open)}
                  aria-label={sidebarOpen ? 'Fechar menu lateral' : 'Abrir menu lateral'}
                >
                  <Menu size={17} />
                </button>
                <span className="status-dot" /> Veronica Intelligence Core{' '}
                <span className="version">UNIVERSAL · {mode === 'agent' ? 'AGENT' : 'CHAT'}</span>
              </div>

              <div className="top-actions">
                {/* Tenant / Organization pill in topbar */}
                <div className="top-tenant-wrap">
                  <TenantSelector currentTenantId={tenantId} onSelectTenant={setTenantId} />
                </div>

                <label className="preference-control">
                  <span>Modelo</span>
                  <select
                    aria-label="Modelo Veronica"
                    title="Perfil local: rápido, equilibrado ou resposta ampliada"
                    disabled={busy}
                    value={modelProfile}
                    onChange={(event) => setModelProfile(event.target.value as ModelProfile)}
                  >
                    <option value="vrn-fast">VRN-fast</option>
                    <option value="vrn-pro">VRN-pro</option>
                    <option value="vrn-ultra">VRN-ultra</option>
                  </select>
                </label>

                <label className="preference-control response-control">
                  <span>Resposta</span>
                  <select
                    aria-label="Estilo da resposta"
                    title="Escolha como a Veronica apresenta a resposta"
                    disabled={busy}
                    value={responseStyle}
                    onChange={(event) => setResponseStyle(event.target.value as ResponseStyle)}
                  >
                    <option value="technical">Técnica</option>
                    <option value="natural">Natural</option>
                    <option value="mixed">Mista</option>
                    <option value="engineer">Engenheira</option>
                  </select>
                </label>

                <button
                  className={mode === 'chat' ? 'mode active' : 'mode'}
                  onClick={() => setMode('chat')}
                >
                  <Bot size={15} /> Chat
                </button>
                <button
                  className={mode === 'agent' ? 'mode active' : 'mode'}
                  onClick={() => setMode('agent')}
                >
                  <Wrench size={15} /> Agent
                </button>

                <button
                  className="theme-toggle"
                  onClick={() => setTheme((current) => (current === 'light' ? 'dark' : 'light'))}
                  aria-label={`Ativar tema ${theme === 'light' ? 'escuro' : 'claro'}`}
                  title={`Tema ${theme === 'light' ? 'escuro' : 'claro'}`}
                >
                  {theme === 'light' ? <Moon size={16} /> : <Sun size={16} />}
                  <span>{theme === 'light' ? 'Escuro' : 'Claro'}</span>
                </button>

                <span className="secure">
                  <ShieldCheck size={14} /> Ambiente seguro
                </span>
              </div>
            </header>

            {/* Chat Scroll Feed */}
            <section className="chat-scroll" ref={chatScrollRef} onScroll={onChatScroll}>
              {messages.length === 0 ? (
                <div className="welcome">
                  <div className="welcome-icon">
                    <BrainCircuit size={32} />
                  </div>
                  <p className="eyebrow">UNIVERSAL INTELLIGENCE · POLYGLOT CODEX · WORKSPACE</p>
                  <h1>Como posso ajudar você hoje?</h1>
                  <p>
                    Especialista em programação poliglota (Python, TypeScript, Rust, C++, Go, SQL), matemática,
                    ciências, análise de dados e engenharia determinística com execução em sandbox.
                  </p>

                  <div className="suggestions">
                    <button onClick={() => setDraft('/codigo Implemente um servidor HTTP assíncrono em Rust com Axum e SQLite')}>
                      <Code size={17} /> Programação &amp; Codex
                      <small>/codigo</small>
                    </button>
                    <button onClick={() => setDraft('/run Execute uma análise estatística e cálculo matricial em Python com NumPy')}>
                      <Zap size={17} /> Sandbox Python &amp; Dados
                      <small>/run</small>
                    </button>
                    <button onClick={() => setDraft('/diagnostico Analise os logs do cluster e verifique estabilidade de enlaces')}>
                      <Activity size={17} /> Engenharia de Sistemas &amp; Redes
                      <small>/diagnostico</small>
                    </button>
                  </div>
                </div>
              ) : (
                <div className="messages">
                  <div className="conversation-heading">
                    <div>
                      <small>CONVERSA ATUAL · {tenantId}</small>
                      <h2>{activeConversation?.title || 'Nova conversa'}</h2>
                    </div>
                    <span>{messages.length} mensagens</span>
                  </div>

                  {messages.map((m, i) => {
                    const isLastAssistant = m.role === 'assistant' && i === messages.length - 1;

                    return (
                      <article key={i} className={`message ${m.role}`}>
                        <div className="message-avatar">
                          {m.role === 'user' ? 'A' : <BrainCircuit size={18} />}
                        </div>

                        <div className="message-body">
                          <div className="message-author">
                            {m.role === 'user' ? 'Você' : 'Veronica'}
                            {m.role === 'assistant' && m.payload && (
                              <span className="epistemic fact">Resposta Técnica Verificada</span>
                            )}
                            {m.profile && (
                              <span className={`message-profile ${m.profile}`}>
                                {m.profile.toUpperCase()}
                              </span>
                            )}
                            {m.style && (
                              <span className="message-style">
                                {
                                  (
                                    {
                                      technical: 'Técnica',
                                      natural: 'Natural',
                                      mixed: 'Mista',
                                      engineer: 'Engenheira',
                                    } as Record<ResponseStyle, string>
                                  )[m.style]
                                }
                              </span>
                            )}
                          </div>

                          {/* Reasoning Accordion (Above Veronica's response) */}
                          {m.role === 'assistant' && isLastAssistant && (thoughts.length > 0 || tools.length > 0) && (
                            <ReasoningAccordion
                              thoughts={thoughts}
                              actions={tools}
                              isStreaming={busy}
                              defaultOpen={busy}
                            />
                          )}

                          <div className="message-text">
                            {m.content ||
                              (busy && isLastAssistant ? (
                                <span className="typing">
                                  Pensando, executando código e validando hipóteses…
                                </span>
                              ) : (
                                ''
                              ))}
                          </div>

                          {/* Epistemic Badges (VERIFIED_FACT, TECHNICAL_INFERENCE, HYPOTHESIS) */}
                          {m.role === 'assistant' && m.payload && (
                            <EpistemicBadges
                              breakdown={m.payload.epistemic_breakdown}
                              sources={m.payload.technical_sources}
                              evidences={m.payload.evidences}
                              onTriggerDiagnostic={handleTriggerDiagnostic}
                            />
                          )}

                          {/* Legacy Payload Fallback if needed */}
                          {m.role === 'assistant' && m.payload && (
                            <PayloadFallback payload={m.payload} />
                          )}
                        </div>
                      </article>
                    );
                  })}
                </div>
              )}

              {showScrollToBottom && messages.length > 0 && (
                <button className="scroll-to-latest" onClick={scrollToLatest}>
                  <ArrowDown size={15} /> Última resposta
                </button>
              )}
            </section>

            {/* Omnibox de Comando no ChatUI */}
            <div className="composer-wrap">
              <Omnibox
                draft={draft}
                setDraft={setDraft}
                busy={busy}
                onSend={handleOmniboxSend}
                onCancel={() => aborter.current?.abort()}
                devices={devices}
                documents={documents}
              />
            </div>
          </main>

          {/* NetCanvas (WorkUI) Split-Screen Panel */}
          {artifact && (
            <NetCanvas
              artifact={artifact}
              versions={artifactVersions}
              onClose={() => setArtifact(null)}
              onSaved={(updated) => {
                setArtifact(updated);
                void refresh();
              }}
              onAskAdjustment={handleAskAdjustment}
              onSelectVersion={(v) => void openStoredArtifact(artifact.id, v)}
              onAbortChange={handleAbortChange}
              onApproveStage6={handleApproveStage6}
              api={api}
            />
          )}
        </>
      )}

      {error && (
        <button className="error-toast" onClick={() => setError('')}>
          <CircleHelp size={16} />
          {error}
          <span>×</span>
        </button>
      )}
    </div>
  );
}

function conversationTitle(prompt: string) {
  const title = prompt.replace(/^\s*\/\w+\s*/, '').replace(/\s+/g, ' ').trim();
  return title.length > 52 ? `${title.slice(0, 51).trimEnd()}…` : title || 'Conversa';
}

function PayloadFallback({ payload }: { payload: StructuredAssistantPayload }) {
  const sources = payload.technical_sources || [];
  const commands = payload.proposed_commands || [];
  if (!payload.diagnosis && sources.length === 0 && commands.length === 0) return null;
  return (
    <div className="payload">
      {payload.diagnosis && <p>{payload.diagnosis}</p>}
      {sources.length > 0 && (
        <div className="source-list">
          <b>Fontes Oficiais</b>
          {sources.slice(0, 3).map((source) => (
            <a key={source.doc_id} href={source.source_uri} target="_blank" rel="noreferrer">
              <FileText size={13} />
              {source.title}
              <ArrowUp size={11} />
            </a>
          ))}
        </div>
      )}
      {commands.length > 0 && payload.risk && (
        <span className="risk-tag">
          <ShieldCheck size={14} />
          {payload.risk}
        </span>
      )}
    </div>
  );
}

function Observatory() {
  const [nodes, setNodes] = useState<ArchitectureNode[]>([]);
  const [metrics, setMetrics] = useState<TrainingStepMetric[]>([]);
  const [selected, setSelected] = useState<ArchitectureNode | null>(null);
  const [telemetry, setTelemetry] = useState<Record<string, unknown>>({});

  async function loadObservatory() {
    try {
      const [a, m, t] = await Promise.all([
        api<{ nodes: ArchitectureNode[] }>('/api/neural/architecture'),
        api<{ history: TrainingStepMetric[] }>('/api/neural/metrics'),
        api<Record<string, unknown>>('/api/neural/telemetry'),
      ]);
      setNodes(a.nodes || []);
      setMetrics(m.history || []);
      setTelemetry(t);
    } catch {
      /* Observatory displays its offline state. */
    }
  }

  useEffect(() => {
    void loadObservatory();
    const id = window.setInterval(() => void loadObservatory(), 5000);
    return () => clearInterval(id);
  }, []);

  const chart = metrics.slice(-36);
  const values = chart.map((m) => Number(m.train_loss || 0));
  const max = Math.max(...values, 1);
  const points = values
    .map((v, i) => `${chart.length < 2 ? 0 : (i * 500) / (chart.length - 1)},${140 - (v / max) * 122}`)
    .join(' ');

  return (
    <main className="observatory">
      <header className="topbar">
        <div>
          <span className="status-dot" /> Neural Observatory{' '}
          <span className="version">VERONICA TRANSFORMER</span>
        </div>
        <button className="mode" onClick={() => useUi.getState().setView('work')}>
          Voltar ao workspace
        </button>
      </header>
      <div className="obs-scroll">
        <div className="obs-title">
          <div>
            <p className="eyebrow">LIVE MODEL TELEMETRY</p>
            <h1>Observatório Neural</h1>
            <p>Arquitetura, gradientes e atividade de treinamento em tempo real.</p>
          </div>
          <span className="live">
            <i /> LIVE · 5s
          </span>
        </div>

        <div className="metric-cards">
          <Metric title="LOSS DE TREINO" value={chart.at(-1)?.train_loss?.toFixed(4) || '—'} icon={<Activity />} />
          <Metric title="GRAD NORM" value={chart.at(-1)?.grad_norm?.toFixed(3) || '—'} icon={<Gauge />} />
          <Metric title="CAMADAS" value={String(nodes.length)} icon={<GitBranch />} />
          <Metric
            title="HARDWARE"
            value={String(telemetry.cpu_percent ?? '—') + (telemetry.cpu_percent !== undefined ? '%' : '')}
            icon={<BrainCircuit />}
          />
        </div>

        <div className="obs-grid">
          <section className="panel architecture">
            <div className="panel-heading">
              <h2>Arquitetura VeronicaTransformer</h2>
              <span>{nodes.length} nós · selecione para inspecionar</span>
            </div>
            <div className="architecture-flow">
              {nodes.length ? (
                nodes.map((n, i) => (
                  <div className="flow-wrap" key={n.id}>
                    <button
                      className={`architecture-node type-${n.type} ${selected?.id === n.id ? 'picked' : ''}`}
                      onClick={() => setSelected(n)}
                    >
                      <span>{n.type}</span>
                      <b>{n.label}</b>
                      <small>{n.tensor_shape?.join(' × ')}</small>
                    </button>
                    {i < nodes.length - 1 && <div className="flow-arrow">→</div>}
                  </div>
                ))
              ) : (
                <div className="empty-state">Conectando à API de arquitetura…</div>
              )}
            </div>
            {selected && (
              <div className="node-detail">
                <b>{selected.label}</b>
                <span>Dimensões: {selected.tensor_shape?.join(' × ')}</span>
                <span>
                  Parâmetros: {selected.stats?.params?.toLocaleString() || 0} · Grad norm:{' '}
                  {selected.stats?.grad_l2_norm?.toFixed(4) || '—'}
                </span>
                <code>{selected.formula}</code>
              </div>
            )}
          </section>

          <section className="panel chart-panel">
            <div className="panel-heading">
              <h2>Loss de treinamento</h2>
              <span>últimos {chart.length} pontos</span>
            </div>
            <svg className="loss-chart" viewBox="0 0 500 160" preserveAspectRatio="none">
              <defs>
                <linearGradient id="fill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0" stopColor="#60d8ca" stopOpacity=".24" />
                  <stop offset="1" stopColor="#60d8ca" stopOpacity="0" />
                </linearGradient>
              </defs>
              <path
                d={points ? `M ${points.replaceAll(' ', ' L ')} L 500 160 L 0 160 Z` : ''}
                fill="url(#fill)"
              />
              <polyline
                points={points}
                fill="none"
                stroke="#60d8ca"
                strokeWidth="2.5"
                vectorEffect="non-scaling-stroke"
              />
            </svg>
            <div className="chart-labels">
              <span>Loss</span>
              <span>{chart.length ? new Date(chart[0].timestamp).toLocaleTimeString() : 'Aguardando métricas'}</span>
              <span>{chart.length ? new Date(chart.at(-1)!.timestamp).toLocaleTimeString() : ''}</span>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}

function Metric({ title, value, icon }: { title: string; value: string; icon: React.ReactNode }) {
  return (
    <div className="metric-card">
      <span>{icon}</span>
      <small>{title}</small>
      <b>{value}</b>
    </div>
  );
}
