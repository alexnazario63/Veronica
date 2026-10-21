import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { Activity, ArrowLeft, BrainCircuit, Check, Database, HardDrive, Play, RefreshCw, Save, Settings2, Sparkles, Square } from 'lucide-react';

type Neuron = {
  neuron_id: string;
  trigger_concept: string;
  developmental_stage: string;
  firing_count: number;
  coords_3d: [number, number, number];
};
type Graph = { nodes: Neuron[]; edges: { source: string; target: string; strength: number }[]; total_neurons: number };
type Runtime = { is_running: boolean; workers: number; interval_seconds: number; cycles_completed: number; total_episodic_memories: number; total_neurons_in_circuit: number };
type ResearchQuality = {
  indexed_documents: number;
  queries_recorded: number;
  by_domain: Record<string, { queries: number; mean_source_count: number; mean_full_page_count: number; mean_primary_source_count: number; mean_claim_count: number; mean_coverage: number; mean_corroboration: number; possible_conflict_rate: number }>;
};
type Overview = {
  model: Record<string, unknown>;
  development: Record<string, unknown>;
  autonomous_learning: Runtime;
  binary_memory: Record<string, unknown>;
  memory: { total_semantic_concepts: number; total_episodic_memories: number; total_consolidation_cycles: number; semantic_nodes: { concept: string; category?: string }[] };
  runtime: { array_backend: string; device_writes_enabled: boolean };
};

const api = async <T,>(path: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json', 'X-User-Id': 'usr-eng-01', ...init?.headers } });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `API ${response.status}`);
  }
  return response.json();
};

export default function Admin({ onBack }: { onBack: () => void }) {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [graph, setGraph] = useState<Graph | null>(null);
  const [researchQuality, setResearchQuality] = useState<ResearchQuality | null>(null);
  const [workers, setWorkers] = useState(2);
  const [interval, setInterval] = useState(6);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const sceneHost = useRef<HTMLDivElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [o, g, q] = await Promise.all([
        api<Overview>('/api/admin/overview'),
        api<Graph>('/api/admin/memory-graph?limit=400'),
        api<ResearchQuality>('/api/admin/research-quality'),
      ]);
      setOverview(o);
      setGraph(g);
      setResearchQuality(q);
      setWorkers(o.autonomous_learning.workers);
      setInterval(o.autonomous_learning.interval_seconds);
      setNotice('');
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Não foi possível carregar os dados administrativos.');
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  useEffect(() => {
    const host = sceneHost.current;
    if (!host || !graph) return;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 400);
    camera.position.set(0, 0, 125);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    host.replaceChildren(renderer.domElement);
    const cloud = new THREE.Group();
    scene.add(cloud);

    const positions: number[] = [];
    const colors: number[] = [];
    const stageColors: Record<string, THREE.Color> = {
      SPECIALIZED: new THREE.Color('#8d78c7'),
      MATURE: new THREE.Color('#55a89c'),
      SYNAPTICALLY_INTEGRATED: new THREE.Color('#79b8ad'),
      NEONATAL: new THREE.Color('#a7c9c0'),
    };
    const byId = new Map<string, Neuron>();
    graph.nodes.forEach((node, index) => {
      byId.set(node.neuron_id, node);
      const angle = index * 2.399963;
      const radius = 36 + (index % 13) * 0.85;
      const fallback: [number, number, number] = [Math.cos(angle) * radius, Math.sin(angle) * radius, Math.sin(angle * 0.67) * radius * 0.55];
      const point = node.coords_3d?.length === 3 ? node.coords_3d : fallback;
      positions.push(...point);
      const color = stageColors[node.developmental_stage] || stageColors.NEONATAL;
      colors.push(color.r, color.g, color.b);
    });
    const pointsGeometry = new THREE.BufferGeometry();
    pointsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    pointsGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    const points = new THREE.Points(pointsGeometry, new THREE.PointsMaterial({ size: 2.4, vertexColors: true, transparent: true, opacity: 0.94, sizeAttenuation: true }));
    cloud.add(points);

    const edgePositions: number[] = [];
    graph.edges.slice(0, 1100).forEach(edge => {
      const a = byId.get(edge.source)?.coords_3d;
      const b = byId.get(edge.target)?.coords_3d;
      if (a && b) edgePositions.push(...a, ...b);
    });
    if (edgePositions.length) {
      const edgeGeometry = new THREE.BufferGeometry();
      edgeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(edgePositions, 3));
      cloud.add(new THREE.LineSegments(edgeGeometry, new THREE.LineBasicMaterial({ color: '#9bbcb5', transparent: true, opacity: 0.16 })));
    }

    const resize = () => {
      const width = Math.max(host.clientWidth, 1);
      const height = Math.max(host.clientHeight, 1);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();
    let frame = 0;
    let pointerDown = false;
    let previousX = 0;
    const down = (event: PointerEvent) => { pointerDown = true; previousX = event.clientX; };
    const up = () => { pointerDown = false; };
    const move = (event: PointerEvent) => {
      if (!pointerDown) return;
      cloud.rotation.y += (event.clientX - previousX) * 0.006;
      cloud.rotation.x += event.movementY * 0.003;
      previousX = event.clientX;
    };
    renderer.domElement.addEventListener('pointerdown', down);
    window.addEventListener('pointerup', up);
    renderer.domElement.addEventListener('pointermove', move);
    const draw = () => {
      frame = requestAnimationFrame(draw);
      if (!pointerDown) cloud.rotation.y += 0.0009;
      renderer.render(scene, camera);
    };
    draw();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.domElement.removeEventListener('pointerdown', down);
      renderer.domElement.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      pointsGeometry.dispose();
      renderer.dispose();
      host.replaceChildren();
    };
  }, [graph]);

  async function saveRuntimeConfig() {
    setBusy(true);
    try {
      await api('/api/admin/config', { method: 'PUT', body: JSON.stringify({ workers, interval_seconds: interval }) });
      setNotice('Configurações de aprendizado salvas.');
      await refresh();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Não foi possível salvar as configurações.');
    } finally { setBusy(false); }
  }

  async function action(path: string, success: string) {
    setBusy(true);
    try {
      await api(path, { method: 'POST', body: '{}' });
      setNotice(success);
      await refresh();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'A operação não foi concluída.');
    } finally { setBusy(false); }
  }

  const model = overview?.model || {};
  const runtime = overview?.autonomous_learning;
  const memory = overview?.memory;
  const binary = overview?.binary_memory || {};

  return <main className="admin-page">
    <header className="admin-topbar"><button className="back-link" onClick={onBack}><ArrowLeft size={17}/> Workspace</button><div className="admin-title"><span className="admin-mark"><Settings2 size={18}/></span><div><b>Administração</b><small>Memória e configuração do motor</small></div></div><button className="refresh-button" onClick={() => void refresh()}><RefreshCw size={15}/> Atualizar</button></header>
    {notice && <div className="admin-notice">{notice}<button onClick={() => setNotice('')}>Fechar</button></div>}
    <div className="admin-content">
      <div className="admin-heading"><div><p className="admin-eyebrow">VISÃO INTERNA</p><h1>Como a Veronica está funcionando</h1><p>Memória associativa, estado do modelo e controles de aprendizado local em um só lugar.</p></div><span className={`runtime-pill ${runtime?.is_running ? 'running' : ''}`}><i/>{runtime?.is_running ? 'Aprendizado ativo' : 'Aprendizado pausado'}</span></div>
      <section className="admin-stats">
        <Stat icon={<BrainCircuit/>} label="Neurônios" value={Number(graph?.total_neurons ?? runtime?.total_neurons_in_circuit ?? 0).toLocaleString('pt-BR')}/>
        <Stat icon={<Sparkles/>} label="Conceitos" value={Number(memory?.total_semantic_concepts ?? 0).toLocaleString('pt-BR')}/>
        <Stat icon={<Activity/>} label="Memórias episódicas" value={Number(memory?.total_episodic_memories ?? 0).toLocaleString('pt-BR')}/>
        <Stat icon={<HardDrive/>} label="Memória binária" value={binary.is_valid_binary ? 'íntegra' : 'sem snapshot válido'}/>
      </section>
      <section className="admin-card research-quality-card">
        <div className="admin-card-head"><div><span className="admin-eyebrow">PESQUISA ABERTA</span><h2>Qualidade da recuperação por assunto</h2></div><span className="node-count">{researchQuality?.queries_recorded ?? 0} consultas · {researchQuality?.indexed_documents ?? 0} fontes persistidas</span></div>
        <div className="research-domain-grid">
          {Object.entries(researchQuality?.by_domain || {}).map(([domain, metrics]) => <div className="config-item" key={domain}><span>{domain}</span><b>{metrics.mean_full_page_count.toFixed(1)} páginas · cobertura {(metrics.mean_coverage * 100).toFixed(0)}%</b><small>{metrics.queries} consultas · {metrics.mean_source_count.toFixed(1)} fontes · corroboracao {(metrics.mean_corroboration * 100).toFixed(0)}%</small>{metrics.possible_conflict_rate > 0 && <small className="quality-conflict">Possíveis divergências: {(metrics.possible_conflict_rate * 100).toFixed(0)}%</small>}</div>)}
          {!Object.keys(researchQuality?.by_domain || {}).length && <small className="concept-empty">As métricas aparecem depois das primeiras pesquisas.</small>}
        </div>
        <p className="readonly-note">Cobertura, páginas lidas e corroboracao medem a recuperação de fontes; não são uma taxa de acerto factual.</p>
      </section>
      <div className="admin-grid">
        <section className="admin-card memory-card"><div className="admin-card-head"><div><span className="admin-eyebrow">MEMÓRIA ASSOCIATIVA</span><h2>Mapa neural em 3D</h2></div><span className="node-count">{graph?.nodes.length ?? 0} pontos · arraste para girar</span></div><div className="memory-3d" ref={sceneHost}>{!graph?.nodes.length && <div className="memory-empty">Carregando mapa de memória…</div>}</div><div className="graph-legend"><span><i className="legend-young"/> Em formação</span><span><i className="legend-mature"/> Consolidada</span><span><i className="legend-special"/> Especializada</span><small>Exibindo até 400 neurônios mais recentes e suas associações.</small></div></section>
        <section className="admin-card settings-card"><div className="admin-card-head"><div><span className="admin-eyebrow">CONTROLES DE EXECUÇÃO</span><h2>Aprendizado autônomo</h2></div><span className={`status-label ${runtime?.is_running ? 'on' : ''}`}>{runtime?.is_running ? 'ATIVO' : 'PAUSADO'}</span></div><p className="settings-copy">Ajuste o ritmo e a concorrência do processo local de aprendizado. Alterações de arquitetura exigem reinicialização e aparecem como somente leitura.</p><label className="field-label">Workers em paralelo <b>{workers}</b><input type="range" min="1" max="16" value={workers} onChange={event => setWorkers(Number(event.target.value))}/><span className="range-hints"><small>1 worker</small><small>16 workers</small></span></label><label className="field-label">Intervalo entre ciclos <b>{interval.toFixed(1)} s</b><input type="range" min="0.5" max="30" step="0.5" value={interval} onChange={event => setInterval(Number(event.target.value))}/><span className="range-hints"><small>0,5 s</small><small>30 s</small></span></label><div className="settings-actions"><button className="primary-admin" disabled={busy} onClick={() => void saveRuntimeConfig()}><Save size={15}/> Salvar</button><button className="soft-admin" disabled={busy} onClick={() => void action(runtime?.is_running ? '/api/development/autonomous-learning/stop' : '/api/development/autonomous-learning/start', runtime?.is_running ? 'Aprendizado pausado.' : 'Aprendizado iniciado.')} >{runtime?.is_running ? <><Square size={14}/> Pausar</> : <><Play size={14}/> Iniciar</>}</button></div><div className="cycle-note"><Check size={15}/> {runtime?.cycles_completed ?? 0} ciclos concluídos · último intervalo {runtime?.interval_seconds?.toFixed(1) ?? '—'} s</div></section>
      </div>
      <div className="admin-grid admin-lower">
        <section className="admin-card"><div className="admin-card-head"><div><span className="admin-eyebrow">VERONICA TRANSFORMER</span><h2>Configuração do modelo</h2></div><span className="readonly-tag">SOMENTE LEITURA</span></div><div className="config-grid">{[['Dimensão do modelo','d_model'],['Camadas','n_layers'],['Cabeças de atenção','n_heads'],['Dimensão feed-forward','d_ff'],['Contexto máximo','max_seq_len'],['Vocabulário','vocab_size'],['Posição','positional_type'],['Normalização','norm_type'],['Precisão','precision_mode'],['Seed','seed']].map(([label,key])=><div className="config-item" key={key}><span>{label}</span><b>{String(model[key] ?? '—')}</b></div>)}</div><p className="readonly-note">Esses parâmetros definem os pesos e a forma do modelo; a edição ao vivo poderia invalidar checkpoints. Backend: {overview?.runtime.array_backend}. Escrita em equipamentos: {overview?.runtime.device_writes_enabled ? 'habilitada' : 'bloqueada'}.</p></section>
        <section className="admin-card"><div className="admin-card-head"><div><span className="admin-eyebrow">CONSOLIDAÇÃO</span><h2>Persistência da memória</h2></div><Database size={18} className="admin-card-icon"/></div><div className="persistence-row"><span>Arquivo binário válido</span><b>{binary.is_valid_binary ? 'Sim' : 'Ainda não'}</b></div><div className="persistence-row"><span>Último salvamento</span><b>{String(binary.last_saved_at || 'Sem registro')}</b></div><div className="persistence-row"><span>Ciclos de consolidação</span><b>{memory?.total_consolidation_cycles ?? 0}</b></div><button className="soft-admin consolidate" disabled={busy} onClick={() => void action('/api/admin/memory/consolidate', 'Consolidação da memória concluída.')}><Database size={15}/> Consolidar memória agora</button></section>
      </div>
      <section className="admin-card concepts-card"><div className="admin-card-head"><div><span className="admin-eyebrow">MEMÓRIA SEMÂNTICA</span><h2>Conceitos recentes</h2></div><span className="node-count">{memory?.total_semantic_concepts ?? 0} no total</span></div><div className="concept-chips">{(memory?.semantic_nodes || []).slice(-32).reverse().map((item, index) => <span key={`${item.concept}-${index}`}>{item.concept}<small>{item.category || 'conceito'}</small></span>)}{!memory?.semantic_nodes?.length && <small className="concept-empty">Ainda não há conceitos semânticos consolidados.</small>}</div></section>
      <p className="admin-footnote">O grafo mostra associações neurais locais; não representa telemetria nem estado de equipamentos de rede.</p>
    </div>
  </main>;
}

function Stat({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return <div className="admin-stat"><span>{icon}</span><div><small>{label}</small><b>{value}</b></div></div>;
}
