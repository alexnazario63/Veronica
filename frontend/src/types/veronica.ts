/**
 * Contratos TypeScript da Interface Gráfica e Observabilidade Neural da Veronica (Seção 16).
 */

export interface CitationMetadata {
  doc_id: string;
  title: string;
  vendor: string;
  family_model: string;
  os_name: string;
  version_build: string;
  feature: string;
  original_source: string;
  source_uri: string;
  page_section: string;
  published_date: string;
  consulted_date: string;
  reliability_score: number;
  applicability_restrictions: string;
  is_official: boolean;
  is_obsolete: boolean;
}

export interface EpistemicBreakdown {
  verified_facts: string[];
  technical_inferences: string[];
  hypotheses: string[];
}

export interface StructuredAssistantPayload {
  agent_name: string;
  model_version: string;
  diagnosis: string;
  epistemic_breakdown: EpistemicBreakdown;
  evidences: string[];
  technical_sources: CitationMetadata[];
  compatibility: string[];
  risk: string;
  safe_diagnostic_commands: string[];
  proposed_commands: string[];
  validation_plan: string[];
  rollback_plan: string[];
  limitations_and_unknowns: string[];
  confidence: number;
}

export interface ArchitectureNode {
  id: string;
  label: string;
  type: "input" | "embedding" | "positional" | "attention" | "ffn" | "norm" | "output";
  tensor_shape: (string | number)[];
  formula: string;
  stats: {
    params: number;
    trainable: boolean;
    status: "trainable" | "frozen" | "fixed";
    weight_mean?: number;
    weight_std?: number;
    weight_l2_norm?: number;
    grad_l2_norm?: number;
    memory_bytes?: number;
  };
}

export interface TrainingStepMetric {
  timestamp: string;
  experiment_id: string;
  model_version: string;
  epoch: number;
  total_epochs: number;
  global_step: number;
  tokens_processed: number;
  train_loss: number;
  val_loss: number;
  perplexity: number;
  learning_rate: number;
  grad_norm: number;
  non_finite_grads: number;
  step_time_ms: number;
  tokens_per_sec: number;
  elapsed_sec: number;
  eta_sec: number | null;
  batch_size: number;
  warnings: string[];
}
