# Contratos e Interfaces entre Módulos — VERONICA

## 1. Interfaces do Núcleo Neural (`core/`)

### 1.1 `Tensor` (`core/neural/tensor.py`)
Representa um array multidimensional com rastreamento de grafo computacional e diferenciação automática reversa (Backpropagation).
- `Tensor(data, requires_grad=False, name=None)`
- Operações suportadas com gradiente analítico exato: `+`, `-`, `*`, `/`, `@` (`matmul`), `transpose`, `reshape`, `sum`, `mean`, `exp`, `log`, `pow`, `relu`, `gelu`, `sigmoid`, `tanh`, `softmax`, `log_softmax`.
- `backward(grad=None)`: Propaga gradientes topologicamente da saída até as folhas (`requires_grad=True`).
- `zero_grad()`: Zera o acumulador `.grad`.

### 1.2 `Module` & `Parameter` (`core/neural/layers.py`)
- `forward(*args, **kwargs) -> Tensor`
- `parameters() -> List[Parameter]`
- `named_parameters() -> Dict[str, Parameter]`
- `architecture_summary() -> Dict[str, Any]` (retorna grafo de camadas, dimensões, parâmetros treináveis e congelados para o Neural Observatory).

### 1.3 `VeronicaTransformer` (`core/transformer/model.py`)
- `forward(input_ids: np.ndarray, capture_diagnostics: bool = False) -> Tuple[Tensor, Dict[str, Any]]`
- `generate(prompt_ids: List[int], max_new_tokens: int, temperature: float, top_k: int, top_p: float, stop_checker: Callable) -> Iterator[int]`

---

## 2. Interface Padronizada de Adaptadores de Equipamentos (`networking/vendors/base.py`)

Todo adaptador de fabricante deve implementar obrigatoriamente os **16 métodos padronizados** definidos na Seção 9:

```python
class BaseVendorAdapter(ABC):
    @abstractmethod
    def discover_device(self, host: str, context: Dict[str, Any]) -> DeviceIdentity: ...
    @abstractmethod
    def get_capabilities(self, identity: DeviceIdentity) -> DeviceCapabilities: ...
    @abstractmethod
    def get_version(self, identity: DeviceIdentity) -> VersionInfo: ...
    @abstractmethod
    def get_running_config(self, identity: DeviceIdentity) -> ConfigSnapshot: ...
    @abstractmethod
    def get_operational_state(self, identity: DeviceIdentity) -> OperationalState: ...
    @abstractmethod
    def get_topology_neighbors(self, identity: DeviceIdentity) -> List[NeighborLink]: ...
    @abstractmethod
    def collect_diagnostics(self, identity: DeviceIdentity, target: str) -> DiagnosticBundle: ...
    @abstractmethod
    def validate_commands(self, identity: DeviceIdentity, commands: List[str]) -> CommandValidationResult: ...
    @abstractmethod
    def generate_candidate(self, identity: DeviceIdentity, intent: ChangeIntent) -> CandidateConfig: ...
    @abstractmethod
    def compare_configurations(self, running: str, candidate: str) -> ConfigDiff: ...
    @abstractmethod
    def check_preconditions(self, identity: DeviceIdentity, plan: ChangePlan) -> PreconditionReport: ...
    @abstractmethod
    def create_backup(self, identity: DeviceIdentity) -> BackupRecord: ...
    @abstractmethod
    def prepare_rollback(self, identity: DeviceIdentity, backup: BackupRecord, plan: ChangePlan) -> RollbackPlan: ...
    @abstractmethod
    def apply_change(self, identity: DeviceIdentity, plan: ChangePlan, approval_token: str) -> ExecutionResult: ...
    @abstractmethod
    def verify_change(self, identity: DeviceIdentity, baseline: OperationalState, post_state: OperationalState) -> VerificationReport: ...
    @abstractmethod
    def rollback_change(self, identity: DeviceIdentity, rollback_plan: RollbackPlan) -> RollbackResult: ...
```

---

## 3. Serviços da Camada de Aplicação e GUI (Seção 16.6)

1. **`ChatService`**: Coordena a geração de respostas combinando RAG documental, diagnóstico determinístico e inferência do modelo neural próprio Veronica, separando fatos comprovados (`VERIFIED_FACT`), inferências (`TECHNICAL_INFERENCE`) e hipóteses (`HYPOTHESIS`).
2. **`ConversationService`**: Gerencia sessões por usuário, persistência de histórico, sumarização de contexto, anexos e referências documentais.
3. **`SpeechRecognitionService`**: Gerencia transcrição de voz (ASR), detecção de atividade vocal (VAD) e normalização de termos de engenharia de redes.
4. **`SpeechSynthesisService`**: Gerencia síntese de voz (TTS), controle de reprodução e interrupção imediata (barge-in).
5. **`NeuralTrainingService`**: Controla o ciclo de vida de experimentos (`STOPPED`, `INITIALIZING`, `TRAINING`, `EVALUATING`, `PAUSED`, `COMPLETED`), minibatches, gradient clipping e publicação de eventos via streaming.
6. **`NeuralMetricsService`**: Agrega estatísticas reais de loss, perplexidade, normas de gradientes, distribuições de pesos, saturação de ativações e telemetria real de CPU/RAM/GPU/VRAM.
7. **`ArchitectureVisualizationService`**: Exporta a topologia completa do modelo neural (neurônios individuais para modelos compactos e blocos/cabeças agrupados para modelos maiores).
8. **`BenchmarkService`**: Executa baterias de avaliação técnica repetíveis por protocolo, fabricante e versão, medindo precisão de citações, detecção de comandos inválidos e acurácia diagnóstica.
9. **`ModelRegistryService`**: Gerencia checkpoints serializados, validação de integridade SHA-256 e promoção controlada entre `TRAINING`, `STAGING` e `PRODUCTION`.
10. **`NetworkOperationsService`**: Orquestra o motor de diagnóstico de 11 etapas e o fluxo de configuração segura de 10 etapas, garantindo que a GUI nunca acesse equipamentos diretamente.
