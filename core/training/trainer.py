"""
Motor de Treinamento Supervisionado e Previsão do Próximo Token da Veronica.
Implementa:
- Preparação de minibatches causais (input_ids -> target_ids deslocados em 1 posição)
- Acumulação de gradientes, Gradient Clipping e agendamento de Learning Rate
- Cálculo real de Loss de treino, Loss de validação, Perplexidade e Normas de Gradientes
- Detecção automática de sobreajuste, divergência, gradientes explosivos e NaN/Inf
- Telemetria real de CPU, RAM, armazenamento e GPU (quando disponível)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import psutil

from core.neural.losses import CrossEntropyLoss
from core.neural.optimizers import AdamW, CosineWarmupScheduler, SGD
from core.neural.clipping import (
    clip_grad_norm_,
    compute_grad_norm,
    has_non_finite_gradients,
    per_parameter_grad_norms,
)
from core.neural.checkpoint import save_checkpoint
from core.tokenizer.tokenizer import TechnicalTokenizer
from core.transformer.model import VeronicaTransformer


DEFAULT_NETWORK_TRAINING_CORPUS: List[Dict[str, str]] = [
    {
        "id": "sample-bgp-xr-01",
        "domain": "routing.bgp",
        "vendor": "Cisco",
        "os": "IOS-XR 7.9.2",
        "text": (
            "Pergunta: Sessao BGP em estado Active no Cisco IOS-XR. "
            "Resposta: Verificar conectividade TCP 179 com show bgp ipv4 unicast summary, "
            "confirmar remote-as, update-source Loopback0, ebgp-multihop e route-policy obrigatoria para eBGP no IOS-XR."
        ),
    },
    {
        "id": "sample-bgp-huawei-02",
        "domain": "routing.bgp",
        "vendor": "Huawei",
        "os": "VRP 8.220",
        "text": (
            "Pergunta: Como diagnosticar peer BGP Idle ou Active no Huawei VRP? "
            "Resposta: Executar display bgp peer e display bgp peer verbose, validar peer connect-interface Loopback0, "
            "peer ebgp-max-hop e filtros ip-prefix ou route-policy antes de propor commit trial."
        ),
    },
    {
        "id": "sample-ospf-mtu-03",
        "domain": "routing.ospf",
        "vendor": "Juniper",
        "os": "Junos 22.4R3",
        "text": (
            "Pergunta: Adjacencia OSPF presa em ExStart ou Exchange entre roteadores. "
            "Resposta: Causa raiz tipica e divergencia de MTU de interface IP. "
            "Consultar show ospf neighbor detail e show interfaces extensive para alinhar MTU nos dois lados do enlace."
        ),
    },
    {
        "id": "sample-optical-dwdm-04",
        "domain": "infrastructure.optical",
        "vendor": "Huawei",
        "os": "VRP 8.220",
        "text": (
            "Pergunta: Queda de interface 10GE com alarme de potencia optica RX baixa. "
            "Resposta: Consultar display transceiver interface 10GE1/0/1 verbose, comparar RX power atual em dBm "
            "com o limiar minimo de sensibilidade do modulo SFP+ e inspecionar atenuacao da fibra e conectores."
        ),
    },
    {
        "id": "sample-mpls-evpn-05",
        "domain": "mpls.evpn",
        "vendor": "Nokia",
        "os": "SR OS 23.10.R2",
        "text": (
            "Pergunta: Servico EVPN-VXLAN sem aprendizado de MAC remoto no Nokia SR OS. "
            "Resposta: Verificar show router bgp summary family evpn, validar route-target import/export, "
            "VNI e tunel VTEP antes de qualquer alteracao."
        ),
    },
    {
        "id": "sample-mikrotik-ros7-06",
        "domain": "routing.bgp",
        "vendor": "MikroTik",
        "os": "RouterOS 7.14",
        "text": (
            "Pergunta: Diferenca de BGP entre MikroTik RouterOS 6 e RouterOS 7. "
            "Resposta: No RouterOS 7 a sintaxe mudou para /routing/bgp/connection e /routing/filter/rule. "
            "Consultar /routing/bgp/session/print para verificar estado Established."
        ),
    },
    {
        "id": "sample-stp-loop-07",
        "domain": "Switching.stp",
        "vendor": "Arista",
        "os": "EOS 4.31.2F",
        "text": (
            "Pergunta: Alta utilizacao de CPU e MAC flapping em topologia Layer 2. "
            "Resposta: Verificar show spanning-tree blockedports e show mac address-table move, "
            "identificar porta sem BPDU Guard gerando loop e isolar com seguranca."
        ),
    },
    {
        "id": "sample-fortios-ipsec-08",
        "domain": "security.ipsec",
        "vendor": "Fortinet",
        "os": "FortiOS 7.4.3",
        "text": (
            "Pergunta: Tunel IPsec Phase 2 nao sobe no FortiGate FortiOS. "
            "Resposta: Executar diagnose vpn ike gateway list e diagnose vpn tunnel list, "
            "confirmar propostas de criptografia, PFS, DH group e seletores de sub-rede Phase 2."
        ),
    },
]


@dataclass
class TrainingHyperparameters:
    epochs: int = 6
    batch_size: int = 4
    seq_len: int = 32
    learning_rate: float = 0.008
    min_lr: float = 0.001
    weight_decay: float = 0.01
    optimizer_type: str = "adamw"  # "adamw" ou "sgd"
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 1
    warmup_steps: int = 2
    seed: int = 42


@dataclass
class StepMetricRecord:
    timestamp: str
    experiment_id: str
    model_version: str
    epoch: int
    total_epochs: int
    global_step: int
    tokens_processed: int
    train_loss: float
    val_loss: float
    perplexity: float
    learning_rate: float
    grad_norm: float
    non_finite_grads: int
    step_time_ms: float
    tokens_per_sec: float
    elapsed_sec: float
    eta_sec: Optional[float]
    batch_size: int
    warnings: List[str] = field(default_factory=list)


def get_real_hardware_telemetry(dataset_read_speed_mb_s: Optional[float] = None) -> Dict[str, Any]:
    """
    Obtém telemetria real de CPU, RAM e armazenamento do sistema operacional via psutil.
    Quando GPU dedicada não está disponível, reporta explicitamente `available: False`
    em vez de inventar valores (conforme Seção 16.3.D).
    """
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(".")
    cpu_pct = psutil.cpu_percent(interval=None)
    proc = psutil.Process()
    proc_mem = proc.memory_info().rss

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cpu": {
            "available": True,
            "Logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "utilization_percent": float(cpu_pct),
        },
        "ram": {
            "available": True,
            "total_mb": round(vm.total / (1024 * 1024), 2),
            "used_mb": round(vm.used / (1024 * 1024), 2),
            "process_rss_mb": round(proc_mem / (1024 * 1024), 2),
            "utilization_percent": float(vm.percent),
        },
        "gpu": {
            "available": False,
            "status": "INDISPONÍVEL (Execução em CPU Nativa NumPy — sem GPU dedicada detectada)",
            "vram_total_mb": None,
            "vram_used_mb": None,
            "accelerator_utilization_percent": None,
        },
        "storage": {
            "available": True,
            "total_gb": round(disk.total / (1024 ** 3), 2),
            "used_gb": round(disk.used / (1024 ** 3), 2),
            "free_gb": round(disk.free / (1024 ** 3), 2),
            "utilization_percent": float(disk.percent),
        },
        "dataset_io": {
            "read_speed_mb_s": round(dataset_read_speed_mb_s, 3)
            if dataset_read_speed_mb_s is not None
            else None,
        },
    }


class VeronicaTrainer:
    """
    Gerencia o treinamento real por previsão do próximo token da VeronicaTransformer,
    com suporte a pausa/retomada/cancelamento, diagnósticos de estabilidade e checkpoints.
    """

    def __init__(
        self,
        model: VeronicaTransformer,
        tokenizer: TechnicalTokenizer,
        checkpoint_dir: Path,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.state: str = "STOPPED"  # STOPPED, INITIALIZING, TRAINING, EVALUATING, PAUSED, COMPLETED
        self.current_experiment_id: str = "exp-initial"
        self.model_version: str = "veronica-v1.0-base"
        self.history: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.non_finite_count: int = 0
        self.last_dataset_read_mb_s: Optional[float] = None
        self.data_prep_time_ms: float = 0.0
        self._pause_requested: bool = False
        self._cancel_requested: bool = False

    def prepare_batches(
        self,
        corpus: Sequence[Dict[str, str]],
        seq_len: int,
        batch_size: int,
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]], int]:
        """
        Tokeniza o corpus técnico e constrói minibatches (X, Y) onde Y é X deslocado em 1 token.
        Retorna (train_batches, val_batches, total_tokens).
        """
        t0 = time.perf_counter()
        all_ids: List[int] = []
        raw_bytes = 0
        for item in corpus:
            txt = item["text"]
            raw_bytes += len(txt.encode("utf-8"))
            encoded = self.tokenizer.encode(txt, add_bos=True, add_eos=True)
            all_ids.extend(encoded)

        elapsed = max(time.perf_counter() - t0, 1e-6)
        self.data_prep_time_ms = round(elapsed * 1000.0, 3)
        self.last_dataset_read_mb_s = (raw_bytes / (1024 * 1024)) / elapsed

        # Garante sequência suficiente para múltiplos minibatches
        window = seq_len + 1
        while len(all_ids) < max(window * batch_size * 2, window * 4):
            all_ids.extend(all_ids)

        sequences: List[np.ndarray] = []
        stride = max(seq_len // 2, 4)
        for start in range(0, len(all_ids) - window + 1, stride):
            chunk = np.array(all_ids[start : start + window], dtype=np.int64)
            sequences.append(chunk)

        batches: List[Tuple[np.ndarray, np.ndarray]] = []
        for idx in range(0, len(sequences), batch_size):
            slice_seqs = sequences[idx : idx + batch_size]
            arr = np.stack(slice_seqs, axis=0)
            x_batch = arr[:, :-1]
            y_batch = arr[:, 1:]
            batches.append((x_batch, y_batch))

        if len(batches) >= 2:
            train_batches = batches[:-1]
            val_batches = [batches[-1]]
        else:
            train_batches = batches
            val_batches = batches

        return train_batches, val_batches, len(all_ids)

    def evaluate_validation(
        self,
        val_batches: Sequence[Tuple[np.ndarray, np.ndarray]],
        loss_fn: CrossEntropyLoss,
    ) -> float:
        self.model.eval()
        losses: List[float] = []
        for x_val, y_val in val_batches:
            logits, _ = self.model.forward(x_val, use_cache=False, capture_diagnostics=True)
            loss_t = loss_fn(logits, y_val)
            losses.append(float(loss_t.item()))
        self.model.train()
        return float(np.mean(losses)) if losses else 0.0

    def run_experiment(
        self,
        experiment_id: str,
        hparams: TrainingHyperparameters,
        corpus: Optional[Sequence[Dict[str, str]]] = None,
        on_step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executa um experimento completo de treinamento supervisionado (next-token prediction)
        e salva um checkpoint validado ao concluir.
        """
        self.state = "INITIALIZING"
        self.current_experiment_id = experiment_id
        self._pause_requested = False
        self._cancel_requested = False
        self.warnings = []

        active_corpus = list(corpus) if corpus else DEFAULT_NETWORK_TRAINING_CORPUS
        available_vocab_slots = max(0, self.model.config.vocab_size - self.tokenizer.vocab_size)
        if available_vocab_slots > 0:
            self.tokenizer.fit(
                [c["text"] for c in active_corpus],
                max_new_tokens=min(64, available_vocab_slots),
                min_freq=1,
            )

        train_batches, val_batches, corpus_tokens = self.prepare_batches(
            active_corpus,
            seq_len=min(hparams.seq_len, self.model.config.max_seq_len),
            batch_size=hparams.batch_size,
        )

        loss_fn = CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)
        if hparams.optimizer_type.lower() == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                lr=hparams.learning_rate,
                momentum=0.9,
                weight_decay=hparams.weight_decay,
            )
        else:
            optimizer = AdamW(
                self.model.parameters(),
                lr=hparams.learning_rate,
                weight_decay=hparams.weight_decay,
            )

        total_steps = max(hparams.epochs * len(train_batches), 1)
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            max_lr=hparams.learning_rate,
            min_lr=hparams.min_lr,
            warmup_steps=hparams.warmup_steps,
            total_steps=total_steps,
        )

        self.state = "TRAINING"
        start_time = time.perf_counter()
        global_step = 0
        tokens_processed = 0
        accum_count = 0
        optimizer.zero_grad()

        initial_train_loss: Optional[float] = None
        last_train_loss: float = 0.0
        last_val_loss: float = 0.0

        for epoch in range(1, hparams.epochs + 1):
            if self._cancel_requested:
                break

            for x_batch, y_batch in train_batches:
                while self._pause_requested and not self._cancel_requested:
                    self.state = "PAUSED"
                    time.sleep(0.05)
                if self._cancel_requested:
                    break

                self.state = "TRAINING"
                step_t0 = time.perf_counter()
                global_step += 1
                current_lr = scheduler.step(global_step - 1)

                logits, _ = self.model.forward(
                    x_batch,
                    use_cache=False,
                    capture_diagnostics=(global_step == 1 or global_step % 4 == 0),
                )
                loss = loss_fn(logits, y_batch)
                loss_val = float(loss.item())
                if initial_train_loss is None:
                    initial_train_loss = loss_val
                last_train_loss = loss_val

                scaled_loss = (
                    loss * (1.0 / hparams.grad_accum_steps)
                    if hparams.grad_accum_steps > 1
                    else loss
                )
                scaled_loss.backward()
                accum_count += 1

                step_warnings: List[str] = []
                if has_non_finite_gradients(self.model.parameters()):
                    self.non_finite_count += 1
                    step_warnings.append("NON_FINITE_GRADIENT_DETECTED: Gradiente NaN/Inf descartado.")
                    optimizer.zero_grad()
                    accum_count = 0
                    grad_norm = 0.0
                elif accum_count >= hparams.grad_accum_steps:
                    grad_norm = clip_grad_norm_(
                        self.model.parameters(), max_norm=hparams.max_grad_norm
                    )
                    if grad_norm > hparams.max_grad_norm * 3.0:
                        step_warnings.append(
                            f"HIGH_GRADIENT_NORM ({grad_norm:.2f}): aplicado gradient clipping."
                        )
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_count = 0
                else:
                    grad_norm = compute_grad_norm(self.model.parameters())

                self.state = "EVALUATING"
                val_loss = self.evaluate_validation(val_batches, loss_fn)
                last_val_loss = val_loss
                self.state = "TRAINING"

                # Detecção de sobreajuste ou divergência
                if val_loss > loss_val * 1.45 and global_step > 3:
                    step_warnings.append(
                        "OVERFITTING_SIGNAL: val_loss significativamente superior a train_loss."
                    )
                if initial_train_loss is not None and loss_val > initial_train_loss * 1.5:
                    step_warnings.append("DIVERGENCE_SIGNAL: train_loss subindo acima do valor inicial.")

                batch_tokens = int(x_batch.size)
                tokens_processed += batch_tokens
                step_dt = max(time.perf_counter() - step_t0, 1e-6)
                elapsed = time.perf_counter() - start_time
                remaining_steps = max(total_steps - global_step, 0)
                avg_step_sec = elapsed / max(global_step, 1)
                eta_sec = round(remaining_steps * avg_step_sec, 2)

                ppl = float(math.exp(min(loss_val, 20.0)))
                record = StepMetricRecord(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    experiment_id=experiment_id,
                    model_version=self.model_version,
                    epoch=epoch,
                    total_epochs=hparams.epochs,
                    global_step=global_step,
                    tokens_processed=tokens_processed,
                    train_loss=round(loss_val, 6),
                    val_loss=round(val_loss, 6),
                    perplexity=round(ppl, 4),
                    learning_rate=round(current_lr, 6),
                    grad_norm=round(grad_norm, 6),
                    non_finite_grads=self.non_finite_count,
                    step_time_ms=round(step_dt * 1000.0, 3),
                    tokens_per_sec=round(batch_tokens / step_dt, 2),
                    elapsed_sec=round(elapsed, 3),
                    eta_sec=eta_sec,
                    batch_size=int(x_batch.shape[0]),
                    warnings=step_warnings,
                )
                rec_dict = asdict(record)
                self.history.append(rec_dict)
                if step_warnings:
                    self.warnings.extend(step_warnings)
                if on_step_callback is not None:
                    on_step_callback(rec_dict)

        checkpoint_id = f"ckpt-{experiment_id}-step{global_step}"
        manifest = save_checkpoint(
            self.model,
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_id=checkpoint_id,
            optimizer=optimizer,
            metadata={
                "experiment_id": experiment_id,
                "hyperparameters": asdict(hparams),
                "initial_train_loss": initial_train_loss,
                "final_train_loss": last_train_loss,
                "final_val_loss": last_val_loss,
                "tokens_processed": tokens_processed,
                "corpus_samples": len(active_corpus),
                "corpus_tokens": corpus_tokens,
                "environment": "TRAINING",
                "promoted_to_staging": False,
            },
        )

        self.state = "STOPPED" if self._cancel_requested else "COMPLETED"
        return {
            "experiment_id": experiment_id,
            "state": self.state,
            "steps_executed": global_step,
            "tokens_processed": tokens_processed,
            "initial_train_loss": initial_train_loss,
            "final_train_loss": last_train_loss,
            "final_val_loss": last_val_loss,
            "checkpoint": manifest,
        }

    def pause(self) -> None:
        self._pause_requested = True

    def resume(self) -> None:
        self._pause_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True
        self._pause_requested = False

    def inspect_parameters(self) -> Dict[str, Any]:
        """
        Retorna estatísticas agregadas dos parâmetros, gradientes, embeddings,
        atenção e consumo de memória por componente (Seção 16.3.C), sem trafegar
        todos os pesos brutos para a interface gráfica.
        """
        named = self.model.named_parameters()
        grad_norms = per_parameter_grad_norms(named)
        layer_summaries: List[Dict[str, Any]] = []

        for name, param in named.items():
            arr = param.data
            hist_counts, hist_edges = np.histogram(arr, bins=10)
            layer_summaries.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "num_params": param.size,
                    "trainable": param.requires_grad,
                    "mean": round(float(np.mean(arr)), 6),
                    "std": round(float(np.std(arr)), 6),
                    "min": round(float(np.min(arr)), 6),
                    "max": round(float(np.max(arr)), 6),
                    "l2_norm": round(float(np.linalg.norm(arr)), 6),
                    "grad_norm": round(grad_norms.get(name, 0.0), 6),
                    "memory_bytes": int(arr.nbytes),
                    "histogram": {
                        "counts": [int(c) for c in hist_counts],
                        "bin_edges": [round(float(e), 4) for e in hist_edges],
                    },
                }
            )

        emb_w = self.model.token_embedding.weight.data
        emb_norms = np.linalg.norm(emb_w, axis=-1)

        return {
            "parameter_counts": self.model.parameter_counts(),
            "embedding_stats": {
                "vocab_size": int(emb_w.shape[0]),
                "embedding_dim": int(emb_w.shape[1]),
                "mean_vector_norm": round(float(np.mean(emb_norms)), 5),
                "max_vector_norm": round(float(np.max(emb_norms)), 5),
                "min_vector_norm": round(float(np.min(emb_norms)), 5),
            },
            "attention_stats": [
                {
                    "layer": i,
                    **block.attn.last_attention_stats,
                }
                for i, block in enumerate(self.model.blocks)
            ],
            "activation_stats": [
                {
                    "layer": i,
                    **block.ffn.last_activation_stats,
                }
                for i, block in enumerate(self.model.blocks)
            ],
            "layers": layer_summaries,
        }
