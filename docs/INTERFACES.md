# Contratos e Interfaces entre Módulos — VERONICA

Este documento especifica formalmente as interfaces de programação, contratos de tipagem, assinaturas de métodos e protocolos de comunicação entre os subsistemas da **Veronica**.

---

## 1. Contratos do Núcleo Matemático e Neural (`core/neural` & `core/transformer`)

### 1.1 `Tensor` ([core/neural/tensor.py](file:///c:/Users/alexn/Desktop/Veronica/core/neural/tensor.py))
Array multidimensional com autograd reverso e grafo computacional:
```python
class Tensor:
    data: np.ndarray
    grad: Optional[np.ndarray]
    requires_grad: bool
    creator: Optional[Function]

    def __init__(self, data: ArrayLike, requires_grad: bool = False, name: Optional[str] = None): ...
    def backward(self, grad: Optional[np.ndarray] = None) -> None: ...
    def zero_grad(self) -> None: ...
    def matmul(self, other: Tensor) -> Tensor: ...
    def relu(self) -> Tensor: ...
    def gelu(self) -> Tensor: ...
    def softmax(self, axis: int = -1) -> Tensor: ...
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> Tensor: ...
    def transpose(self, *axes: int) -> Tensor: ...
    def reshape(self, *shape: int) -> Tensor: ...
```

### 1.2 `Module` e `Parameter` ([core/neural/layers.py](file:///c:/Users/alexn/Desktop/Veronica/core/neural/layers.py))
```python
class Parameter(Tensor):
    def __init__(self, data: np.ndarray, name: Optional[str] = None): ...

class Module(ABC):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def parameters(self) -> List[Parameter]: ...
    def named_parameters(self) -> Dict[str, Parameter]: ...
    def architecture_summary(self) -> Dict[str, Any]: ...
```

### 1.3 `VeronicaTransformer` ([core/transformer/model.py](file:///c:/Users/alexn/Desktop/Veronica/core/transformer/model.py))
```python
class VeronicaTransformer(Module):
    def __init__(self, config: VeronicaConfig): ...
    def forward(
        self,
        input_ids: np.ndarray,
        kv_cache: Optional[PrefixKVCache] = None,
        capture_diagnostics: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]: ...
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_checker: Optional[Callable[[List[int]], bool]] = None,
    ) -> Iterator[int]: ...
```

### 1.4 `ConstrainedJSONDecoder` ([core/inference/constrained_json.py](file:///c:/Users/alexn/Desktop/Veronica/core/inference/constrained_json.py))
Garante que a decodificação autoregressiva respeite esquemas JSON estritos sem quebra sintática:
```python
class ConstrainedJSONDecoder:
    def __init__(self, schema: Dict[str, Any], tokenizer: TechnicalTokenizer): ...
    def filter_logits(self, prefix_tokens: List[int], logits: np.ndarray) -> np.ndarray: ...
    def is_valid_terminal(self, tokens: List[int]) -> bool: ...
```

---

## 2. Contratos de Desenvolvimento Cognitivo e Neurogênese (`core/development/`)

### 2.1 `DevelopmentEngine` ([core/development/development_engine.py](file:///c:/Users/alexn/Desktop/Veronica/core/development/development_engine.py))
```python
class DevelopmentalStage(Enum):
    INFANT = "INFANT"
    CHILD = "CHILD"
    ADOLESCENT = "ADOLESCENT"
    SPECIALIST = "SPECIALIST"
    EXPERT = "EXPERT"

class DevelopmentEngine:
    def get_status(self) -> Dict[str, Any]: ...
    def record_perception(self, modality: str, raw_data: Any, context_hint: Optional[str] = None) -> Dict[str, Any]: ...
    def consolidate_cognition(self, force: bool = False) -> Dict[str, Any]: ...
    def assess_milestones(self) -> List[DevelopmentalMilestone]: ...
```

### 2.2 `NeurogenesisEngine` ([core/development/neurogenesis.py](file:///c:/Users/alexn/Desktop/Veronica/core/development/neurogenesis.py))
```python
class DynamicNeuron:
    id: str
    circuit_name: str
    weights: Parameter
    bias: Parameter
    created_at: str
    activation_history: List[float]

class NeurogenesisEngine:
    def evaluate_signal_surprise(self, circuit: str, signal: np.ndarray) -> float: ...
    def trigger_neurogenesis(self, circuit: str, signal: np.ndarray, reason: str) -> DynamicNeuron: ...
    def prune_idle_synapses(self, inactivity_threshold_cycles: int) -> int: ...
```

### 2.3 `MemorySystem` & `BinaryMachineMemoryStore` ([core/development/memory_system.py](file:///c:/Users/alexn/Desktop/Veronica/core/development/memory_system.py))
```python
class MemorySystem:
    def record_working_memory(self, item: WorkingMemoryItem) -> None: ...
    def commit_episodic_memory(self, event: EpisodicMemoryRecord) -> str: ...
    def update_concept_graph(self, subject: str, predicate: str, target: str, weight: float) -> None: ...
    def query_concepts(self, query: str, limit: int = 10) -> List[ConceptNode]: ...

class BinaryMachineMemoryStore:
    def serialize_to_binary(self, filepath: Path) -> int: ...
    def load_from_binary(self, filepath: Path) -> None: ...
    def fast_knn_lookup(self, vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]: ...
```

### 2.4 `AutonomousCognitiveDaemon` ([core/development/autonomous_learning.py](file:///c:/Users/alexn/Desktop/Veronica/core/development/autonomous_learning.py))
```python
class AutonomousCognitiveDaemon:
    def start(self, interval_seconds: float = 60.0, num_workers: int = 2) -> None: ...
    def stop(self) -> None: ...
    def run_consolidation_cycle(self) -> Dict[str, Any]: ...
    def get_daemon_status(self) -> Dict[str, Any]: ...
```

---

## 3. Contratos de Agente ReAct (`core/agent/orchestrator.py`)

```python
class AgentTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    is_safe_read_only: bool

    @abstractmethod
    def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]: ...

class ReActOrchestrator:
    def __init__(self, tools: List[AgentTool], llm_engine: Any): ...
    def run_step(self, question: str, history: List[Dict[str, str]]) -> Dict[str, Any]: ...
    def solve(self, question: str, max_iterations: int = 8) -> AsyncIterator[Dict[str, Any]]: ...
```

---

## 4. Contratos de Adaptadores Multi-Vendor (`networking/vendors/`)

Todo adaptador de rede (seja síncrono via `BaseVendorAdapter` ou assíncrono via `AsyncVendorAdapter`) implementa os **16 métodos fundamentais padronizados**:

```python
class BaseVendorAdapter(ABC):
    @abstractmethod
    def discover_device(self, host: str, context: Dict[str, Any]) -> DeviceIdentity:
        """1. Identifica com precisão fabricante, modelo, SO e versão em execução."""
        ...

    @abstractmethod
    def get_capabilities(self, identity: DeviceIdentity) -> DeviceCapabilities:
        """2. Retorna capacidades suportadas (NETCONF, BGP-4, EVPN, Rollback Timer)."""
        ...

    @abstractmethod
    def get_version(self, identity: DeviceIdentity) -> VersionInfo:
        """3. Coleta informações detalhadas de build, patches e packages instalados."""
        ...

    @abstractmethod
    def get_running_config(self, identity: DeviceIdentity) -> ConfigSnapshot:
        """4. Extrai a configuração ativa atual e gera seu hash SHA-256 canônico."""
        ...

    @abstractmethod
    def get_operational_state(self, identity: DeviceIdentity) -> OperationalState:
        """5. Coleta estado de interfaces, CPU, memória, tabelas de rotas e alarmes."""
        ...

    @abstractmethod
    def get_topology_neighbors(self, identity: DeviceIdentity) -> List[NeighborLink]:
        """6. Descobre adjacências L2/L3 via LLDP, CDP, OSPF e sessões BGP."""
        ...

    @abstractmethod
    def collect_diagnostics(self, identity: DeviceIdentity, target: str) -> DiagnosticBundle:
        """7. Coleta telemetria e buffers de diagnóstico de forma não intrusiva (read-only)."""
        ...

    @abstractmethod
    def validate_commands(self, identity: DeviceIdentity, commands: List[str]) -> CommandValidationResult:
        """8. Valida estritamente a sintaxe dos comandos contra a gramática da versão exata."""
        ...

    @abstractmethod
    def generate_candidate(self, identity: DeviceIdentity, intent: ChangeIntent) -> CandidateConfig:
        """9. Converte a intenção operacional em bloco de comandos específico do SO."""
        ...

    @abstractmethod
    def compare_configurations(self, running: str, candidate: str) -> ConfigDiff:
        """10. Computa o diff padronizado com linhas de adição, remoção e contexto."""
        ...

    @abstractmethod
    def check_preconditions(self, identity: DeviceIdentity, plan: ChangePlan) -> PreconditionReport:
        """11. Valida espaço em disco, concorrência, interfaces administrativas e riscos."""
        ...

    @abstractmethod
    def create_backup(self, identity: DeviceIdentity) -> BackupRecord:
        """12. Gera snapshot de segurança pré-mudança com hash SHA-256 e timestamp."""
        ...

    @abstractmethod
    def prepare_rollback(self, identity: DeviceIdentity, backup: BackupRecord, plan: ChangePlan) -> RollbackPlan:
        """13. Prepara comandos ou arquivo determinístico de reversão em caso de falha."""
        ...

    @abstractmethod
    def apply_change(self, identity: DeviceIdentity, plan: ChangePlan, approval_token: str) -> ExecutionResult:
        """14. Aplica os comandos sob commit confirmed e monitoramento contínuo de sessão."""
        ...

    @abstractmethod
    def verify_change(self, identity: DeviceIdentity, baseline: OperationalState, post_state: OperationalState) -> VerificationReport:
        """15. Compara telemetria pós-mudança contra baseline e confirma integridade de tráfego."""
        ...

    @abstractmethod
    def rollback_change(self, identity: DeviceIdentity, rollback_plan: RollbackPlan) -> RollbackResult:
        """16. Executa a reversão completa e imediata caso a pós-validação seja rejeitada."""
        ...
```

---

## 5. Contratos de Segurança e Mudança Determinística (`safety/`)

### 5.1 `DeterministicChangeController` ([safety/change_control/controller.py](file:///c:/Users/alexn/Desktop/Veronica/safety/change_control/controller.py))
```python
class DeterministicChangeController:
    def stage_change(self, target: DeviceIdentity, intent: ChangeIntent) -> ChangePlan: ...
    def authorize_change(self, plan_id: str, approver_id: str, token: str, window_id: str) -> AuthorizationReceipt: ...
    def execute_change(self, plan_id: str, authorization_receipt: AuthorizationReceipt) -> ExecutionResult: ...
    def trigger_abort_and_rollback(self, plan_id: str, reason: str) -> RollbackResult: ...
```

### 5.2 `TamperEvidentAuditLog` ([safety/policies/rbac_vault.py](file:///c:/Users/alexn/Desktop/Veronica/safety/policies/rbac_vault.py))
```python
class AuditRecord:
    index: int
    timestamp: str
    operator_id: str
    action: str
    payload_hash: str
    previous_hash: str
    record_hash: str

class TamperEvidentAuditLog:
    def append_record(self, operator_id: str, action: str, payload: Dict[str, Any]) -> AuditRecord: ...
    def verify_integrity(self) -> Tuple[bool, Optional[int]]: ...
```

---

## 6. Contratos de Workspace e Persistência de Artefatos (`services/api/workspace/`)

Interface de versionamento de documentos ([services/api/workspace/store.py](file:///c:/Users/alexn/Desktop/Veronica/services/api/workspace/store.py)):
```python
class ArtifactStore:
    def __init__(self, db_path: Path): ...
    def save_artifact(self, artifact_id: str, title: str, content: str, kind: str, author: str) -> int:
        """Salva nova versão do artefato e retorna o número incremental da versão."""
        ...
    def get_artifact(self, artifact_id: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]: ...
    def list_artifacts(self) -> List[Dict[str, Any]]: ...
    def get_version_history(self, artifact_id: str) -> List[Dict[str, Any]]: ...
    def propose_patch(self, artifact_id: str, version: int, patch_diff: str) -> Dict[str, Any]: ...
```

---

## 7. Serviços Integrados do `ServicesHub` (`services/api/services_hub.py`)

A camada de aplicação expõe 10 serviços modulares instanciados e integrados através do `ServicesHub`:

1. **`ConversationService`**: Gestão do ciclo de vida das conversas, histórico de turnos e persistência estruturada.
2. **`SpeechRecognitionService`**: Transcrição fonética adaptada para termos técnicos (`10GE`, `GigabitEthernet`, `OSPF`, `CIDR`) com checagem de consentimento.
3. **`SpeechSynthesisService`**: Síntese de áudio de alta inteligibilidade e interrupção instantânea via *barge-in*.
4. **`ChatService`**: Coordenação da resposta textual, fusão do RAG, modelo neural e rotulação epistêmica (`VERIFIED_FACT`, `TECHNICAL_INFERENCE`, `HYPOTHESIS`).
5. **`ArchitectureVisualizationService`**: Exportação serializada do grafo neural e das conexões de camadas para o Neural Observatory.
6. **`NeuralMetricsService`**: Agregação de métricas de loss, perplexidade, dispersão de pesos, normas de gradientes e telemetria de hardware (CPU, RAM, GPU, VRAM).
7. **`ModelRegistryService`**: Catálogo de checkpoints treinados, validação de hash SHA-256 e promoção entre `TRAINING`, `STAGING` e `PRODUCTION`.
8. **`BenchmarkService`**: Execução de suites técnicas de avaliação por protocolo, fabricante e cenário de falha.
9. **`NeuralTrainingService`**: Orquestração do loop de treinamento em minibatches, acumulação de gradientes e gradient clipping.
10. **`NetworkOperationsService`**: Barramento seguro que conecta a interface do usuário aos motores de diagnóstico e mudança segura, sem acesso direto do frontend aos equipamentos.
