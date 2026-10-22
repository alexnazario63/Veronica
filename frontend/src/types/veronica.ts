/**
 * Contratos TypeScript da Interface Gráfica, Observabilidade Neural e Workspace Poliglota da Veronica.
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

export type EpistemicBadgeType = 'VERIFIED_FACT' | 'TECHNICAL_INFERENCE' | 'HYPOTHESIS';

export interface EpistemicBreakdown {
  verified_facts: string[];
  technical_inferences: string[];
  hypotheses: string[];
}

export interface StructuredAssistantPayload {
  agent_name?: string;
  model_version?: string;
  intent?: string;
  diagnosis?: string;
  epistemic_breakdown?: EpistemicBreakdown;
  evidences?: string[];
  technical_sources?: CitationMetadata[];
  compatibility?: string[];
  risk?: string;
  safe_diagnostic_commands?: string[];
  proposed_commands?: string[];
  validation_plan?: string[];
  rollback_plan?: string[];
  limitations_and_unknowns?: string[];
  confidence?: number;
}

export type ReasoningCategory =
  | 'sandbox_execution'
  | 'ast_inspection'
  | 'web_search'
  | 'network_diagnostics'
  | 'general_reasoning';

export interface SandboxExecutionResult {
  status: 'success' | 'error';
  language: string;
  code: string;
  stdout: string;
  stderr: string;
  exit_code: number;
  duration_ms: number;
  memory_kb?: number;
  ast_summary?: string;
  timestamp: string;
}

export interface ThoughtTraceItem {
  id: string;
  agent: string;
  thought: string;
  timestamp: string;
  category?: ReasoningCategory;
  epistemic_badge?: EpistemicBadgeType;
  code_snippet?: string;
  ast_detail?: string;
}

export interface ActionTraceItem {
  id: string;
  tool_call_id: string;
  tool_name: string;
  tier: string;
  status: 'running' | 'completed' | 'error';
  duration_ms?: number;
  observation?: string;
  parameters?: Record<string, unknown>;
  category?: ReasoningCategory;
  epistemic_badge?: EpistemicBadgeType;
  sandbox_result?: {
    code?: string;
    stdout?: string;
    stderr?: string;
    exit_code?: number;
    execution_time_ms?: number;
  };
  ast_nodes?: string[];
  search_query?: string;
  vendor?: string;
}

export interface ReasoningTrace {
  thoughts: ThoughtTraceItem[];
  actions: ActionTraceItem[];
}

export type AttachedChipType = 'code' | 'doc' | 'config' | 'log' | 'telemetry' | 'data';

export interface AttachedChip {
  id: string;
  title: string;
  type: AttachedChipType;
  language?: string;
  extension?: string;
  content: string;
  lineCount: number;
  sizeBytes: number;
}

export interface Tenant {
  id: string;
  name: string;
  environment: 'production' | 'staging' | 'edge' | 'lab';
  region?: string;
  devices_count?: number;
}

export interface DeviceItem {
  device_id: string;
  vendor: string;
  management_ip: string;
  model?: string;
  status?: 'online' | 'degraded' | 'offline';
}

export interface KnowledgeDocumentItem {
  doc_id: string;
  title: string;
  vendor?: string;
  family_model?: string;
  os_name?: string;
  version_build?: string;
  feature?: string;
  reliability_score?: number;
  source_uri?: string;
}

export interface Artifact {
  id: string;
  version: number;
  type: 'code_module' | 'config_diff' | 'blast_radius' | 'rca_report' | 'diagnostic_11_steps' | 'change_plan_10_steps' | string;
  language?: string;
  filename?: string;
  baseline_content?: string | null;
  proposed_content?: string | null;
  blast_radius_graph?: Record<string, unknown> | null;
  change_stage?: string | null;
  sha256_hash: string;
  created_at?: string;
}

export interface StoredArtifact {
  artifact: Artifact;
  created_at?: string;
}

export interface ChangePlanRecord {
  change_id: string;
  device_id: string;
  title: string;
  current_stage: number;
  stage_name: string;
  status: string;
  requester_user_id: string;
  maintenance_window: string;
  plan_hash_sha256: string;
  approval?: {
    approved: boolean;
    approver_user_id?: string;
    approval_token?: string;
    approved_at?: string;
    channel?: string;
  };
  stages?: Record<string, unknown>;
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
