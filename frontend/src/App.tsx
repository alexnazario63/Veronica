import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { Activity, ArrowDown, ArrowUp, Bot, BrainCircuit, Check, ChevronDown, CircleHelp, ClipboardList, Command, FileText, Gauge, GitBranch, Menu, Moon, Network, Plus, Search, Settings2, ShieldCheck, Square, Sun, Trash2, Wrench } from 'lucide-react';
import { create } from 'zustand';
import cytoscape, { type ElementDefinition } from 'cytoscape';
import type { ArchitectureNode, StructuredAssistantPayload, TrainingStepMetric } from './types/veronica';
const Admin = lazy(() => import('./Admin'));

type View = 'work' | 'observatory' | 'admin';
type Message = { role: 'user' | 'assistant'; content: string; payload?: StructuredAssistantPayload; profile?: ModelProfile; style?: ResponseStyle };
type Conversation = { conversation_id: string; title: string };
type ModelProfile = 'vrn-fast' | 'vrn-pro' | 'vrn-ultra';
type ResponseStyle = 'technical' | 'natural' | 'mixed' | 'engineer';
type Artifact = { id: string; version: number; type: string; baseline_content?: string | null; proposed_content?: string | null; blast_radius_graph?: Record<string, unknown> | null; change_stage?: string | null; sha256_hash: string };
type StoredArtifact = { artifact: Artifact; created_at?: string };
type Store = { view: View; setView: (view: View) => void; artifact: Artifact | null; setArtifact: (artifact: Artifact | null) => void };
const viewFromPath = (): View => window.location.pathname === '/admin' ? 'admin' : 'work';
const useUi = create<Store>((set) => ({ view: viewFromPath(), setView: (view) => { const path = view === 'admin' ? '/admin' : '/'; if (window.location.pathname !== path) window.history.pushState({}, '', path); set({ view }); }, artifact: null, setArtifact: (artifact) => set({ artifact }) }));

const api = async <T,>(path: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json', 'X-User-Id': 'usr-eng-01', ...init?.headers } });
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `API ${response.status}`); }
  return response.json();
};

export default function VeronicaApp() {
  const { view, setView, artifact, setArtifact } = useUi();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversationId, setConversationId] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [devices, setDevices] = useState<{ device_id: string; vendor: string; management_ip: string }[]>([]);
  const [workspaceArtifacts, setWorkspaceArtifacts] = useState<StoredArtifact[]>([]);
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
    try { return localStorage.getItem('veronica-theme') === 'dark' ? 'dark' : 'light'; }
    catch { return 'light'; }
  });
  const [modelProfile, setModelProfile] = useState<ModelProfile>(() => {
    try { const saved = localStorage.getItem('veronica-model-profile'); return saved === 'vrn-fast' || saved === 'vrn-ultra' ? saved : 'vrn-pro'; }
    catch { return 'vrn-pro'; }
  });
  const [responseStyle, setResponseStyle] = useState<ResponseStyle>(() => {
    try { const saved = localStorage.getItem('veronica-response-style'); return saved === 'natural' || saved === 'mixed' || saved === 'engineer' ? saved : 'technical'; }
    catch { return 'technical'; }
  });
  const aborter = useRef<AbortController | null>(null);
  const chatScrollRef = useRef<HTMLElement | null>(null);
  const stickToBottom = useRef(true);
  const restoredLatestConversation = useRef(false);
  const logLines = useMemo(() => draft.split('\n').length, [draft]);
  const displayDraft = logLines > 14 ? `[Log: ${logLines} linhas]` : draft;
  const activeConversation = conversations.find(item => item.conversation_id === conversationId);

  useEffect(() => {
    const syncRoute = () => setView(viewFromPath());
    window.addEventListener('popstate', syncRoute);
    return () => window.removeEventListener('popstate', syncRoute);
  }, [setView]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('veronica-theme', theme); } catch { /* storage can be unavailable */ }
  }, [theme]);
  useEffect(() => {
    try { localStorage.setItem('veronica-model-profile', modelProfile); localStorage.setItem('veronica-response-style', responseStyle); }
    catch { /* storage can be unavailable */ }
  }, [modelProfile, responseStyle]);

  const refresh = async () => {
    try {
      const [c, inventory, saved] = await Promise.all([api<{ conversations: Conversation[] }>('/api/conversations'), api<{ devices: typeof devices }>('/api/network/inventory'), api<{ artifacts: StoredArtifact[] }>('/api/workspace/artifacts').catch(() => ({ artifacts: [] }))]);
      const rows = c.conversations || [];
      setConversations(rows); setDevices(inventory.devices || []); setWorkspaceArtifacts(saved.artifacts || []);
      if (!restoredLatestConversation.current) {
        restoredLatestConversation.current = true;
        if (rows[0]) void selectConversation(rows[0].conversation_id);
      }
    } catch (e) { setError(e instanceof Error ? e.message : 'API indisponível'); }
  };
  useEffect(() => { void refresh(); }, []);
  useEffect(() => {
    if (!stickToBottom.current) return;
    const frame = requestAnimationFrame(() => {
      const scroller = chatScrollRef.current;
      if (scroller) scroller.scrollTop = scroller.scrollHeight;
    });
    return () => cancelAnimationFrame(frame);
  }, [messages]);
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
    setError(''); setSidebarOpen(false);
    stickToBottom.current = true; setShowScrollToBottom(false);
    try { const data = await api<{ messages?: { role: string; content: string; structured_payload?: StructuredAssistantPayload & { model_profile?: string; response_style?: string } }[] }>(`/api/conversations/${id}`); setConversationId(id); setMessages((data.messages || []).map(m => ({ role: m.role === 'user' ? 'user' : 'assistant', content: m.content, payload: m.structured_payload, profile: ['vrn-fast', 'vrn-pro', 'vrn-ultra'].includes(m.structured_payload?.model_profile || '') ? m.structured_payload?.model_profile as ModelProfile : undefined, style: ['technical', 'natural', 'mixed', 'engineer'].includes(m.structured_payload?.response_style || '') ? m.structured_payload?.response_style as ResponseStyle : undefined }))); }
    catch (e) { setConversationId(''); setMessages([]); setError(e instanceof Error ? e.message : 'Não foi possível abrir a conversa'); }
  }
  async function newConversation() {
    restoredLatestConversation.current = true;
    setConversationId(''); setMessages([]); setThoughts([]); setTools([]); setShowScrollToBottom(false); setSidebarOpen(false);
    try { const c = await api<Conversation>('/api/conversations', { method: 'POST', body: JSON.stringify({ title: 'Nova conversa' }) }); await refresh(); setConversationId(c.conversation_id); setMessages([]); stickToBottom.current = true; setShowScrollToBottom(false); setSidebarOpen(false); }
    catch (e) { setError(e instanceof Error ? e.message : 'Não foi possível criar conversa'); }
  }
  async function deleteConversation(event: React.MouseEvent, conversation: Conversation) {
    event.stopPropagation();
    if (!window.confirm(`Excluir a conversa “${conversation.title || 'Conversa sem título'}”? Esta ação não pode ser desfeita.`)) return;
    try {
      await api<{ deleted: boolean }>(`/api/conversations/${encodeURIComponent(conversation.conversation_id)}`, { method: 'DELETE' });
      setConversations(current => current.filter(item => item.conversation_id !== conversation.conversation_id));
      if (conversationId === conversation.conversation_id) { aborter.current?.abort(); setConversationId(''); setMessages([]); setThoughts([]); setTools([]); }
    } catch (e) { setError(e instanceof Error ? e.message : 'Não foi possível excluir a conversa'); }
  }
  async function openStoredArtifact(id: string, version: number) {
    try { const result = await api<{ versions: StoredArtifact[] }>(`/api/workspace/artifacts/${encodeURIComponent(id)}/versions`); const selected = result.versions.find(row => row.artifact.version === version)?.artifact; if (selected) setArtifact(selected); }
    catch (e) { setError(e instanceof Error ? e.message : 'Não foi possível abrir o artefato'); }
  }
  async function send() {
    const content = draft.trim(); if (!content || busy) return;
    const title = conversationTitle(content);
    let id = conversationId;
    if (!id) { try { const c = await api<Conversation>('/api/conversations', { method: 'POST', body: JSON.stringify({ title }) }); id = c.conversation_id; setConversationId(id); restoredLatestConversation.current = true; await refresh(); } catch (e) { setError(e instanceof Error ? e.message : 'Falha ao criar conversa'); return; } }
    else {
      try {
        await api<Conversation>(`/api/conversations/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify({ title }) });
        setConversations(current => [{ conversation_id: id!, title }, ...current.filter(item => item.conversation_id !== id)]);
      } catch (e) { setError(e instanceof Error ? e.message : 'Não foi possível atualizar o título da conversa'); return; }
    }
    stickToBottom.current = true; setShowScrollToBottom(false);
    setDraft(''); setError(''); setThoughts([]); setTools([]); setMessages(prev => [...prev, { role: 'user', content: displayDraft, profile: modelProfile, style: responseStyle }, { role: 'assistant', content: '', profile: modelProfile, style: responseStyle }]); setBusy(true);
    const controller = new AbortController(); aborter.current = controller;
    try {
      const endpoint = mode === 'agent' ? 'agent-stream' : 'stream';
      const response = await fetch(`/api/conversations/${id}/${endpoint}`, { method: 'POST', signal: controller.signal, headers: { 'Content-Type': 'application/json', 'X-User-Id': 'usr-eng-01' }, body: JSON.stringify({ content, mode, model_profile: modelProfile, response_style: responseStyle }) });
      if (!response.ok || !response.body) {
        const failure = await response.json().catch(() => ({}));
        throw new Error(failure.detail || `Falha no stream (${response.status})`);
      }
      const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = '';
      while (true) {
        const { value, done } = await reader.read(); if (done) break; buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split('\n\n'); buffer = chunks.pop() || '';
        for (const chunk of chunks) {
          const line = chunk.split('\n').find(s => s.startsWith('data: ')); if (!line) continue;
          const ev = JSON.parse(line.slice(6));
          if (ev.type === 'error') throw new Error(ev.message || 'Falha ao gerar a resposta');
          else if (ev.type === 'token') appendAssistant(ev.token_text || ev.token || '');
          else if (ev.type === 'complete') setMessages(prev => prev.map((m, i) => i === prev.length - 1 ? { ...m, content: ev.content || m.content, payload: ev.structured_payload } : m));
          else if (ev.type === 'message_delta') appendAssistant(ev.delta || '');
          else if (ev.type === 'thought_delta') setThoughts(prev => [...prev, ev.delta || '']);
          else if (ev.type === 'tool_start') setTools(prev => [...prev, `Iniciado: ${ev.tool_name || 'ferramenta'} · ${ev.tier || 'tier desconhecido'}`]);
          else if (ev.type === 'tool_end') setTools(prev => [...prev, `Concluído: ${ev.tool_call_id || 'ferramenta'} · ${ev.status || 'ok'}${ev.duration_ms ? ` · ${Math.round(ev.duration_ms)} ms` : ''}`]);
          else if (ev.type === 'artifact_open' && ev.artifact) { setArtifact(ev.artifact); void refresh(); }
          else if (ev.type === 'artifact_delta') { const current = useUi.getState().artifact; if (current && current.id === ev.artifact_id) setArtifact({ ...current, version: ev.version || current.version, proposed_content: `${current.proposed_content || ''}${ev.delta || ''}` }); }
          else if (ev.type === 'safety_gate_interrupt') setTools(prev => [...prev, `Aprovação humana necessária · ${ev.change_stage || ''} · ${ev.reason || ''}`]);
        }
      }
      await refresh();
    } catch (e) { if (!(e instanceof DOMException && e.name === 'AbortError')) setError(e instanceof Error ? e.message : 'Erro ao gerar resposta'); }
    finally { setBusy(false); aborter.current = null; }
  }

  function appendAssistant(delta: string) { if (!delta) return; setMessages(prev => prev.map((m, i) => i === prev.length - 1 ? { ...m, content: m.content + delta } : m)); }
  if (view === 'admin') return <Suspense fallback={<div className="admin-loading">Abrindo administração…</div>}><Admin onBack={() => setView('work')} onToggleTheme={() => setTheme(current => current === 'light' ? 'dark' : 'light')} theme={theme}/></Suspense>;
  return <div className={`shell ${sidebarOpen ? 'sidebar-open' : ''}`}>
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><BrainCircuit size={21}/></div><div><b>VERONICA</b><small>NET NEURAL AI</small></div></div>
      <button className="new-chat" onClick={newConversation}><Plus size={17}/> Nova conversa <Command size={14}/></button>
      <div className="side-label">Workspace</div>
      <button className={`side-link ${view === 'work' ? 'selected' : ''}`} onClick={() => setView('work')}><Network size={16}/> Chat &amp; NetCanvas</button>
      <button className={`side-link ${view === 'observatory' ? 'selected' : ''}`} onClick={() => setView('observatory')}><Activity size={16}/> Neural Observatory</button>
      <button className="side-link" onClick={() => setView('admin')}><Settings2 size={16}/> Administração</button>
      <button className="side-link history-toggle" onClick={() => setHistoryOpen(!historyOpen)} aria-expanded={historyOpen}><ChevronDown size={15} className={historyOpen ? '' : 'rotated'}/> Conversas <span className="count">{conversations.length}</span></button>
      {historyOpen && <div className="conversation-list">{conversations.map(c => <div key={c.conversation_id} className={`conversation-row ${c.conversation_id === conversationId ? 'selected' : ''}`}><button className="conversation" title={c.title || 'Conversa sem título'} onClick={() => void selectConversation(c.conversation_id)}>{c.title || 'Conversa sem título'}</button><button className="delete-conversation" title="Excluir conversa" aria-label={`Excluir conversa ${c.title || ''}`} onClick={event => void deleteConversation(event, c)}><Trash2 size={14}/></button></div>)}{!conversations.length && <small className="muted pad">Sem conversas recentes</small>}</div>}
      <div className="side-label">Ativos rápidos</div><div className="device-list">{devices.slice(0, 5).map(d => <button key={d.device_id} className="device" onClick={() => setDraft(v => `${v}${v ? ' ' : ''}@${d.device_id} `)}><span className="online-dot"/><span>{d.device_id}<small>{d.vendor} · {d.management_ip}</small></span></button>)}</div>
      <div className="side-label">Artefatos</div><div className="device-list">{workspaceArtifacts.map(({ artifact: item }) => <button key={`${item.id}-${item.version}`} className="device artifact-link" onClick={() => void openStoredArtifact(item.id, item.version)}><FileText size={14}/><span>{item.type}<small>v{item.version} · {item.id}</small></span></button>)}{!workspaceArtifacts.length && <small className="muted pad">Nenhum artefato salvo</small>}</div>
      <div className="sidebar-bottom"><ShieldCheck size={16}/><span>Safety Control ativo<small>Alterações exigem aprovação humana</small></span><button className="sidebar-theme" onClick={() => setTheme(current => current === 'light' ? 'dark' : 'light')} aria-label={`Ativar tema ${theme === 'light' ? 'escuro' : 'claro'}`} title={`Tema ${theme === 'light' ? 'escuro' : 'claro'}`}>{theme === 'light' ? <Moon size={15}/> : <Sun size={15}/>}</button></div>
    </aside>
    {view === 'observatory' ? <Observatory/> : <>
      <main className="main"><header className="topbar"><div><button className="mobile-menu" onClick={() => setSidebarOpen(open => !open)} aria-label={sidebarOpen ? 'Fechar menu lateral' : 'Abrir menu lateral'}><Menu size={17}/></button><span className="status-dot"/> Veronica Transformer <span className="version">LOCAL · {mode === 'agent' ? 'AGENT' : 'CHAT'}</span></div><div className="top-actions"><label className="preference-control"><span>Modelo</span><select aria-label="Modelo Veronica" title="Perfil local: rápido, equilibrado ou resposta ampliada" disabled={busy} value={modelProfile} onChange={event => setModelProfile(event.target.value as ModelProfile)}><option value="vrn-fast">VRN-fast</option><option value="vrn-pro">VRN-pro</option><option value="vrn-ultra">VRN-ultra</option></select></label><label className="preference-control response-control"><span>Resposta</span><select aria-label="Estilo da resposta" title="Escolha como a Veronica apresenta a resposta" disabled={busy} value={responseStyle} onChange={event => setResponseStyle(event.target.value as ResponseStyle)}><option value="technical">Técnica</option><option value="natural">Natural</option><option value="mixed">Mista</option><option value="engineer">Engenheira</option></select></label><button className={mode === 'chat' ? 'mode active' : 'mode'} onClick={() => setMode('chat')}><Bot size={15}/> Chat</button><button className={mode === 'agent' ? 'mode active' : 'mode'} onClick={() => setMode('agent')}><Wrench size={15}/> Agent</button><button className="theme-toggle" onClick={() => setTheme(current => current === 'light' ? 'dark' : 'light')} aria-label={`Ativar tema ${theme === 'light' ? 'escuro' : 'claro'}`} title={`Tema ${theme === 'light' ? 'escuro' : 'claro'}`}>{theme === 'light' ? <Moon size={16}/> : <Sun size={16}/>}<span>{theme === 'light' ? 'Escuro' : 'Claro'}</span></button><span className="secure"><ShieldCheck size={14}/> Ambiente seguro</span></div></header>
        <section className="chat-scroll" ref={chatScrollRef} onScroll={onChatScroll}>{messages.length === 0 ? <div className="welcome"><div className="welcome-icon"><BrainCircuit size={30}/></div><p className="eyebrow">CHAT · AGENT · WORK</p><h1>Como posso ajudar na sua rede?</h1><p>Investigue incidentes, consulte documentação ou prepare mudanças seguras com a Veronica.</p><div className="suggestions"><button onClick={() => setDraft('/diagnostico Investigue instabilidade de BGP')}> <Activity size={17}/>Diagnosticar incidente<small>/diagnostico</small></button><button onClick={() => setDraft('/simular Verifique impacto da mudança')}> <GitBranch size={17}/>Simular impacto<small>/simular</small></button><button onClick={() => setDraft('/mudanca Planeje alteração de configuração')}> <ClipboardList size={17}/>Planejar mudança<small>/mudanca</small></button></div></div> : <div className="messages"><div className="conversation-heading"><div><small>CONVERSA ATUAL</small><h2>{activeConversation?.title || 'Nova conversa'}</h2></div><span>{messages.length} mensagens</span></div>{messages.map((m, i) => <article key={i} className={`message ${m.role}`}><div className="message-avatar">{m.role === 'user' ? 'A' : <BrainCircuit size={18}/>}</div><div className="message-body"><div className="message-author">{m.role === 'user' ? 'Você' : 'Veronica'}{m.role === 'assistant' && m.payload && m.payload.intent !== 'GREETING' && <span className="epistemic fact">Resposta técnica</span>}{m.profile && <span className={`message-profile ${m.profile}`}>{m.profile.toUpperCase()}</span>}{m.style && <span className="message-style">{({ technical: 'Técnica', natural: 'Natural', mixed: 'Mista', engineer: 'Engenheira' } as Record<ResponseStyle, string>)[m.style]}</span>}</div><div className="message-text">{m.content || (busy && i === messages.length - 1 ? <span className="typing">Analisando contexto seguro…</span> : '')}</div>{m.role === 'assistant' && m.payload && <Payload payload={m.payload}/>}</div></article>)}
          {(thoughts.length > 0 || tools.length > 0) && <details className="thinking" open><summary><BrainCircuit size={16}/> Raciocínio e ferramentas <span>{tools.length} ações</span></summary>{thoughts.map((t,i)=><p key={`t${i}`} className="thought">{t}</p>)}{tools.map((t,i)=><p key={`a${i}`} className="tool-event"><Check size={13}/>{t}</p>)}</details>}
        </div>}
        {showScrollToBottom && messages.length > 0 && <button className="scroll-to-latest" onClick={scrollToLatest}><ArrowDown size={15}/> Última resposta</button>}
        </section>
        <div className="composer-wrap"><div className="quick-commands"><button onClick={() => setDraft('/diagnostico ')}> /diagnostico</button><button onClick={() => setDraft('/mudanca ')}> /mudanca</button><button onClick={() => setDraft('/simular ')}> /simular</button><span>@equipamento · @documento</span></div><div className="composer"><textarea value={draft} onChange={e => setDraft(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void send(); } }} placeholder="Pergunte sobre sua rede ou cole saídas de comandos…" rows={Math.min(5, Math.max(1, logLines > 14 ? 1 : logLines))}/><div className="composer-footer"><span>{logLines > 14 ? <><FileText size={13}/> [Log: {logLines} linhas] · </> : null}A Veronica consulta dados em modo leitura até receber aprovação.</span><button className="send-button" onClick={() => busy ? aborter.current?.abort() : void send()} disabled={!busy && !draft.trim()}>{busy ? <Square size={15} fill="currentColor"/> : <ArrowUp size={18}/>}</button></div></div><p className="disclaimer">A IA pode cometer erros. Mudanças em rede passam pelo fluxo de aprovação.</p></div>
      </main>
      {artifact && <Canvas artifact={artifact} onClose={() => setArtifact(null)} onSaved={updated => { setArtifact(updated); void refresh(); }}/>}
    </>}
    {error && <button className="error-toast" onClick={() => setError('')}><CircleHelp size={16}/>{error}<span>×</span></button>}
  </div>;
}

function conversationTitle(prompt: string) {
  const title = prompt.replace(/^\s*\/\w+\s*/, '').replace(/\s+/g, ' ').trim();
  return title.length > 52 ? `${title.slice(0, 51).trimEnd()}\u2026` : title || 'Conversa';
}

function Payload({ payload }: { payload: StructuredAssistantPayload }) {
  const sources = payload.technical_sources || [];
  const commands = payload.proposed_commands || [];
  if (!payload.diagnosis && sources.length === 0 && commands.length === 0) return null;
  return <div className="payload">
    {payload.diagnosis && <p>{payload.diagnosis}</p>}
    {sources.length > 0 && <div className="source-list"><b>Fontes</b>{sources.slice(0, 3).map(source => <a key={source.doc_id} href={source.source_uri} target="_blank" rel="noreferrer"><FileText size={13}/>{source.title}<ArrowUp size={11}/></a>)}</div>}
    {commands.length > 0 && payload.risk && <span className="risk-tag"><ShieldCheck size={14}/>{payload.risk}</span>}
  </div>;
}

function stageNumber(stage?: string | null) { const match = stage?.match(/(?:stage[_-]?)?(\d+)/i); return match ? Number(match[1]) : 1; }

function ArtifactGraph({ graph }: { graph?: Record<string, unknown> | null }) {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!host.current || !graph) return;
    const nested = graph.blast_radius_graph && typeof graph.blast_radius_graph === 'object' ? graph.blast_radius_graph as Record<string, unknown> : graph;
    const nodes = Array.isArray(nested.nodes) ? nested.nodes as Record<string, unknown>[] : [];
    const edges = Array.isArray(nested.edges) ? nested.edges as Record<string, unknown>[] : [];
    const root = typeof nested.root === 'string' ? nested.root : '';
    const elements: ElementDefinition[] = nodes.map(node => {
      const id = String(node.id ?? node.device_id ?? '');
      return { data: { id, label: id, impact: String(node.impact ?? (id === root ? 'root' : 'transitive')) } };
    }).filter(node => Boolean(node.data.id));
    for (const node of nodes) {
      const id = String(node.id ?? node.device_id ?? '');
      if (id && id === root && !elements.some(el => el.data?.id === id)) elements.push({ data: { id, label: id, impact: 'root' } });
    }
    edges.forEach((edge, i) => {
      const source = String(edge.source ?? edge.from ?? ''); const target = String(edge.target ?? edge.to ?? '');
      if (source && target) {
        [source, target].forEach(id => { if (!elements.some(el => el.data?.id === id)) elements.push({ data: { id, label: id, impact: id === root ? 'root' : 'transitive' } }); });
        elements.push({ data: { id: `edge-${i}-${source}-${target}`, source, target } });
      }
    });
    const cy = cytoscape({ container: host.current, elements, layout: { name: 'breadthfirst', directed: false, padding: 24 }, style: [
      { selector: 'node', style: { label: 'data(label)', 'background-color': '#537b78', color: '#d8e4e4', 'font-size': 10, 'text-valign': 'bottom', 'text-margin-y': 6, width: '30px', height: '30px', 'text-wrap': 'wrap', 'text-max-width': '100px' } },
      { selector: 'node[impact = "direct"]', style: { 'background-color': '#d88b53' } },
      { selector: 'node[impact = "root"]', style: { 'background-color': '#d45e53', width: '38px', height: '38px' } },
      { selector: 'edge', style: { width: '2px', 'line-color': '#536d70', 'curve-style': 'bezier' } },
    ] });
    return () => cy.destroy();
  }, [graph]);
  const nested = graph?.blast_radius_graph && typeof graph.blast_radius_graph === 'object' ? graph.blast_radius_graph as Record<string, unknown> : graph;
  const hasGraph = Boolean(nested && ((Array.isArray(nested.nodes) && nested.nodes.length) || (Array.isArray(nested.edges) && nested.edges.length)));
  return hasGraph ? <div className="cytoscape-graph" ref={host}/> : <div className="graph-empty">O artefato não inclui nós e enlaces de topologia.</div>;
}

function Canvas({ artifact, onClose, onSaved }: { artifact: Artifact; onClose: () => void; onSaved: (updated: Artifact) => void }) {
  const [tab, setTab] = useState('Config & Diff'); const [proposed, setProposed] = useState(artifact.proposed_content || ''); const [seconds, setSeconds] = useState(300); const [saving, setSaving] = useState(false);
  useEffect(() => setProposed(artifact.proposed_content || ''), [artifact.id, artifact.version, artifact.proposed_content]);
  async function saveArtifact() {
    setSaving(true);
    try { const result = await api<{ artifact: Artifact }>(`/api/workspace/artifacts/${encodeURIComponent(artifact.id)}/versions/${artifact.version}/proposed-content`, { method: 'PATCH', body: JSON.stringify({ start_line: 1, end_line: Math.max(1, (artifact.proposed_content || '').split('\n').length), replacement: proposed }) }); onSaved(result.artifact); }
    catch (e) { window.alert(e instanceof Error ? e.message : 'Falha ao salvar versão'); }
    finally { setSaving(false); }
  }
  useEffect(() => { const timer = window.setInterval(() => setSeconds(s => Math.max(0, s - 1)), 1000); return () => clearInterval(timer); }, []);
  const tabs = ['Config & Diff', 'Blast Radius', 'Change Runner', 'Diagnóstico & RCA'];
  return <aside className="canvas"><div className="canvas-head"><div><span className="eyebrow">WORKSPACE · NETCANVAS</span><h2>{artifact.type}</h2><small className="muted">Versão {artifact.version}</small></div><button className="icon-btn" onClick={onClose}>×</button></div><div className="canvas-tabs">{tabs.map(t => <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>{t}</button>)}</div>
    <div className="canvas-content">{tab === 'Config & Diff' && <><div className="editor-label"><span>PROPOSED CONFIGURATION</span><span>Salve edições como nova versão do artefato</span></div><div className="diff-grid"><div><label>BASELINE</label><pre>{artifact.baseline_content || '# Baseline indisponível\n# O artefato não contém configuração base.'}</pre></div><div><label>PROPOSTA</label><textarea value={proposed} onChange={e => setProposed(e.target.value)} placeholder="# Conteúdo proposto pelo agente"/></div></div><div className="artifact-meta"><span>SHA-256</span><code>{artifact.sha256_hash}</code></div><button className="export" disabled={saving || proposed === (artifact.proposed_content || '')} onClick={() => void saveArtifact()}>{saving ? 'Salvando…' : 'Salvar nova versão'}</button></>}
      {tab === 'Blast Radius' && <div className="topology"><div className="topology-head"><Network size={17}/> Grafo de blast radius <span>{artifact.blast_radius_graph ? 'dados do artefato' : 'sem dados'}</span></div><ArtifactGraph graph={artifact.blast_radius_graph}/></div>}
      {tab === 'Change Runner' && <><div className="safety-note"><ShieldCheck size={19}/><div><b>Execução protegida</b><small>A etapa 7 permanece bloqueada até aprovação humana assinada.</small></div></div><div className="step-list">{['Descoberta','Baseline','Análise de impacto','Planejamento','Simulação','Aprovação humana','Execução controlada','Pós-validação','Commit ou rollback','Auditoria imutável'].map((s,i)=><div className={`step ${i + 1 < stageNumber(artifact.change_stage) ? 'done' : i + 1 === stageNumber(artifact.change_stage) ? 'current' : ''}`} key={s}><span>{i + 1 < stageNumber(artifact.change_stage) ? <Check size={13}/> : i + 1}</span>{s}{i + 1 === stageNumber(artifact.change_stage) && <small>ATUAL</small>}</div>)}</div><div className="timer"><Gauge size={17}/> Commit Confirmed <b>{String(Math.floor(seconds / 60)).padStart(2,'0')}:{String(seconds % 60).padStart(2,'0')}</b></div><button className="approval" disabled><ShieldCheck size={16}/> Aprovação humana necessária</button></>}
      {tab === 'Diagnóstico & RCA' && <><h3>Trilha de investigação</h3><div className="diagnostic-row"><span className="epistemic fact">FATO VERIFICADO</span> Evidências coletadas pela telemetria</div><div className="diagnostic-row"><span className="epistemic inference">INFERÊNCIA</span> Correlação de eventos e dependências</div><div className="diagnostic-row"><span className="epistemic hypothesis">HIPÓTESE</span> Causa provável aguardando teste seguro <button>Executar teste seguro</button></div><button className="export"><FileText size={15}/> Exportar relatório</button></>}
    </div>
  </aside>;
}

function Observatory() {
  const [nodes, setNodes] = useState<ArchitectureNode[]>([]); const [metrics, setMetrics] = useState<TrainingStepMetric[]>([]); const [selected, setSelected] = useState<ArchitectureNode | null>(null); const [telemetry, setTelemetry] = useState<Record<string, unknown>>({});
  async function loadObservatory() { try { const [a,m,t] = await Promise.all([api<{nodes: ArchitectureNode[]}>('/api/neural/architecture'), api<{history: TrainingStepMetric[]}>('/api/neural/metrics'), api<Record<string, unknown>>('/api/neural/telemetry')]); setNodes(a.nodes || []); setMetrics(m.history || []); setTelemetry(t); } catch { /* Observatory displays its offline state. */ } }
  useEffect(() => { void loadObservatory(); const id = window.setInterval(() => void loadObservatory(), 5000); return () => clearInterval(id); }, []);
  const chart = metrics.slice(-36); const values = chart.map(m => Number(m.train_loss || 0)); const max = Math.max(...values, 1); const points = values.map((v,i) => `${chart.length < 2 ? 0 : i * 500 / (chart.length - 1)},${140 - (v / max) * 122}`).join(' ');
  return <main className="observatory"><header className="topbar"><div><span className="status-dot"/> Neural Observatory <span className="version">VERONICA TRANSFORMER</span></div><button className="mode" onClick={() => useUi.getState().setView('work')}>Voltar ao workspace</button></header><div className="obs-scroll"><div className="obs-title"><div><p className="eyebrow">LIVE MODEL TELEMETRY</p><h1>Observatório Neural</h1><p>Arquitetura, gradientes e atividade de treinamento em tempo real.</p></div><span className="live"><i/> LIVE · 5s</span></div><div className="metric-cards"><Metric title="LOSS DE TREINO" value={chart.at(-1)?.train_loss?.toFixed(4) || '—'} icon={<Activity/>}/><Metric title="GRAD NORM" value={chart.at(-1)?.grad_norm?.toFixed(3) || '—'} icon={<Gauge/>}/><Metric title="CAMADAS" value={String(nodes.length)} icon={<GitBranch/>}/><Metric title="HARDWARE" value={String(telemetry.cpu_percent ?? '—') + (telemetry.cpu_percent !== undefined ? '%' : '')} icon={<BrainCircuit/>}/></div><div className="obs-grid"><section className="panel architecture"><div className="panel-heading"><h2>Arquitetura VeronicaTransformer</h2><span>{nodes.length} nós · selecione para inspecionar</span></div><div className="architecture-flow">{nodes.length ? nodes.map((n,i)=><div className="flow-wrap" key={n.id}><button className={`architecture-node type-${n.type} ${selected?.id === n.id ? 'picked' : ''}`} onClick={() => setSelected(n)}><span>{n.type}</span><b>{n.label}</b><small>{n.tensor_shape?.join(' × ')}</small></button>{i < nodes.length - 1 && <div className="flow-arrow">→</div>}</div>) : <div className="empty-state">Conectando à API de arquitetura…</div>}</div>{selected && <div className="node-detail"><b>{selected.label}</b><span>Dimensões: {selected.tensor_shape?.join(' × ')}</span><span>Parâmetros: {selected.stats?.params?.toLocaleString() || 0} · Grad norm: {selected.stats?.grad_l2_norm?.toFixed(4) || '—'}</span><code>{selected.formula}</code></div>}</section><section className="panel chart-panel"><div className="panel-heading"><h2>Loss de treinamento</h2><span>últimos {chart.length} pontos</span></div><svg className="loss-chart" viewBox="0 0 500 160" preserveAspectRatio="none"><defs><linearGradient id="fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#60d8ca" stopOpacity=".24"/><stop offset="1" stopColor="#60d8ca" stopOpacity="0"/></linearGradient></defs><path d={points ? `M ${points.replaceAll(' ', ' L ')} L 500 160 L 0 160 Z` : ''} fill="url(#fill)"/><polyline points={points} fill="none" stroke="#60d8ca" strokeWidth="2.5" vectorEffect="non-scaling-stroke"/></svg><div className="chart-labels"><span>Loss</span><span>{chart.length ? new Date(chart[0].timestamp).toLocaleTimeString() : 'Aguardando métricas'}</span><span>{chart.length ? new Date(chart.at(-1)!.timestamp).toLocaleTimeString() : ''}</span></div></section></div></div></main>;
}
function Metric({ title, value, icon }: { title: string; value: string; icon: React.ReactNode }) { return <div className="metric-card"><span>{icon}</span><small>{title}</small><b>{value}</b></div>; }
