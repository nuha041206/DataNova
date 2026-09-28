/**
 * TypeScript mirrors of the DataNova API payloads.
 * Kept hand-written (not generated) so students can read and map them easily.
 */

export interface ColumnProfile {
  name: string;
  dtype: string;
  inferred_type: "numerical" | "categorical" | "datetime" | "boolean";
  missing: number;
  missing_pct: number;
  unique: number;
  unique_pct: number;
  mean?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
  q25?: number | null;
  median?: number | null;
  q75?: number | null;
  top_values?: { value: string | number | null; count: number }[] | null;
  is_constant: boolean;
  is_high_cardinality: boolean;
  looks_like_id: boolean;
  target_name_hint?: boolean;
}

export interface Analysis {
  rows: number;
  columns: number;
  duplicate_rows: number;
  total_missing_cells: number;
  memory_mb: number;
  dtypes_summary: Record<string, number>;
  column_profiles: ColumnProfile[];
  constant_columns: string[];
  id_like_columns: string[];
  imbalance: Record<string, { classes: number; min_share: number; max_share: number; is_imbalanced: boolean }>;
  suspicious_values: { column: string; issue: string; detail: string }[];
  warnings: string[];
}

export interface DetectionCandidate {
  column: string;
  score: number;
  task: string;
  reason: string;
}

export interface Detection {
  suggested_target: string | null;
  task: "classification" | "regression" | "unknown";
  confidence: "high" | "medium" | "low" | "user";
  score: number;
  reason: string;
  candidates: DetectionCandidate[];
}

export interface SchemaInfo {
  technical_type: string;
  semantic_type: string;
  semantic_confidence: number;
  detection_reason: string;
  value_sample: string[];
}

export interface UploadResponse {
  upload_id: string;
  filename: string;
  preview: Record<string, unknown>[];
  analysis: Analysis;
  detection: Detection;
  schema?: Record<string, SchemaInfo>;
}

export const TECHNICAL_TYPES = [
  "string", "integer", "float", "boolean", "date", "datetime", "categorical",
] as const;

export const SEMANTIC_TYPES = [
  "generic_text", "person_name", "gender", "address", "city", "state",
  "country", "indian_phone", "email", "indian_pincode", "date_of_birth",
  "date", "datetime", "age", "customer_id", "generic_id", "product_name",
  "company_name", "salary", "currency", "percentage", "category", "integer",
  "float", "boolean", "custom",
] as const;

export const SEMANTIC_LABELS: Record<string, string> = {
  generic_text: "Generic Text",
  person_name: "Person Name",
  gender: "Gender",
  address: "Address",
  city: "City",
  state: "State",
  country: "Country",
  indian_phone: "Indian Phone Number",
  email: "Email",
  indian_pincode: "Indian PIN Code",
  date_of_birth: "Date of Birth",
  date: "Date",
  datetime: "Datetime",
  age: "Age",
  customer_id: "Customer ID",
  generic_id: "Generic ID",
  product_name: "Product Name",
  company_name: "Company Name",
  salary: "Salary",
  currency: "Currency",
  percentage: "Percentage",
  category: "Category",
  integer: "Integer",
  float: "Float / Decimal",
  boolean: "Boolean",
  custom: "Custom",
};

export interface ColumnSpec {
  name: string;
  technical_type: string;
  semantic_type: string;
  min: number | null;
  max: number | null;
  categories: string[] | null;
  distribution: Record<string, unknown> | null;
  unique: boolean;
  detected: boolean;
  /** Parser's explanation for the detected semantic type (optional). */
  detection_reason?: string;
}

export interface QualityCheck {
  check: string;
  passed: boolean;
  detail: string;
}

export interface GenerationQuality {
  passed: boolean;
  checks: QualityCheck[];
  summary: {
    rows: number;
    columns: number;
    unique_rows: number;
    duplicate_rows: number;
    missing_cells: number;
    format_violations: number;
    range_violations: number;
    category_violations: number;
    dtype_violations: number;
  };
  uniqueness: {
    rows: number;
    unique_rows: number;
    duplicate_rows: number;
    duplicate_pct: number;
    unique_columns: Record<string, { unique_values: number; rows: number; is_unique: boolean }>;
  };
}

export interface GenerateRequest {
  n_rows: number;
  target?: string | null;
  task?: string | null;
  exclude_columns?: string[] | null;
  synthesizer?: string | null;
  advanced?: Record<string, unknown> | null;
}

export interface GenerationPreview {
  upload_id: string;
  rows_requested: number;
  target: string | null;
  task: string | null;
  approach: { synthesizer: string; reason: string };
  n_features: number;
  excluded_columns: string[];
  health_warnings: string[];
  sample_preview: Record<string, unknown>[];
  notes: string[];
}

export interface GenerateResponse {
  upload_id: string;
  approach: string;
  reason: string;
  fit_seconds: number;
  sample_seconds: number;
  n_rows_generated: number;
  n_rows_requested: number;
  validation_warnings: string[];
  quality_check?: GenerationQuality;
  uniqueness?: { requested: number; generated: number; unique_rows: number; duplicate_rows: number; uniqueness_fully_satisfied: boolean; notes: string[] };
  schema?: Record<string, SchemaInfo>;
  preview: Record<string, unknown>[];
  columns: string[];
}

export interface ScoreInfo {
  value: number | null;
  definition: string;
  components: Record<string, number | null>;
  weights: Record<string, number>;
}

export interface ColumnEval {
  column: string;
  type: string;
  similarity: number | null;
  ks?: { statistic: number | null; p_value: number | null; similarity: number | null };
  tvd?: { tvd: number | null; similarity: number | null; missing_categories?: string[] };
  stats?: {
    mean: { real: number | null; synthetic: number | null };
    std: { real: number | null; synthetic: number | null };
    min: { real: number | null; synthetic: number | null };
    max: { real: number | null; synthetic: number | null };
    quantiles: { real: (number | null)[]; synthetic: (number | null)[] };
  };
  proportions?: { category: string | number | null; real_share: number; synthetic_share: number }[];
}

export interface Evaluation {
  data_nova_score: ScoreInfo;
  column_overlap: { shared: string[]; only_in_real: string[]; only_in_synthetic: string[] };
  column_results: ColumnEval[];
  correlation_comparison: {
    compared_columns: string[];
    real_matrix?: number[][];
    synthetic_matrix?: number[][];
    columns?: string[];
    mean_abs_difference?: number;
    max_abs_difference?: number;
    largest_diffs?: { pair: [string, string]; real: number; synthetic: number; difference: number }[];
  };
  target_comparison: {
    target: string | null;
    similarity: number | null;
    kind?: string;
    tvd?: { tvd: number | null; similarity: number | null };
    class_proportions?: { category: string | number | null; real_share: number; synthetic_share: number }[];
    stats?: { mean: { real: number; synthetic: number }; std: { real: number; synthetic: number } };
    reason?: string;
  };
  row_counts: { real: number; synthetic: number };
}

export interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusion_matrix: { labels: string[]; matrix: number[][] };
}

export interface RegressionMetrics {
  mae: number;
  mse: number;
  rmse: number;
  r2: number;
}

export interface MLUtility {
  task: "classification" | "regression";
  target: string;
  model: { key: string; label: string };
  skipped_columns?: string[];
  dropped_columns?: string[];
  features_used?: string[];
  imbalance?: {
    class_distribution: Record<string, number>;
    minority_share: number;
    is_imbalanced: boolean;
    note: string | null;
  } | null;
  split: { real_train_rows: number; real_test_rows: number; synthetic_train_rows: number; test_size: number; random_state: number };
  experiment_a: { description: string; metrics: ClassificationMetrics | RegressionMetrics };
  experiment_b: { description: string; metrics: ClassificationMetrics | RegressionMetrics };
  retention: Record<string, number | null>;
  interpretation_note: string;
}

export interface Finding {
  title: string;
  text: string;
  kind: "positive" | "neutral" | "warning";
}

export interface Explanation {
  evaluation: Finding[];
  ml_utility: Finding[];
  generation: Finding[];
}

export interface ModelRecommendation {
  model_key: string | null;
  label: string | null;
  reason: string;
  alternatives: { key: string; label: string; strength: string }[];
  available_models?: Record<string, string>;
}

export interface Clarification {
  field: string;
  question: string;
  suggestion: unknown;
  options: (string | number)[];
}

export interface FeatureSpec {
  name: string;
  type: "integer" | "float" | "category" | "boolean";
}

export interface RequirementSpec {
  rows: number;
  task: string;
  target: string | null;
  /** Legacy single-type feature list (kept for backward compatibility). */
  features: FeatureSpec[];
  /** v2 semantic schema: editable columns with technical + semantic types. */
  columns: ColumnSpec[];
  unique: boolean;
  country: string | null;
  constraints: string[];
  class_distribution: { positive_pct?: number; kind?: string } | null;
  domain: string | null;
  clarifications: Clarification[];
  warnings: string[];
  raw_text: string;
}

export interface RequirementGenerateResponse {
  spec: RequirementSpec;
  summary: {
    rows: number;
    columns: string[];
    target: string;
    task: string;
    dtypes: Record<string, string>;
    numeric_summary: Record<string, { mean: number; std: number; min: number; max: number }>;
    target_summary: Record<string, unknown>;
  };
  description: string;
  warnings: string[];
  uniqueness: { requested: number; generated: number; unique_rows: number; duplicate_rows: number; uniqueness_fully_satisfied: boolean; notes: string[] };
  quality_check: GenerationQuality;
  session_id: string;
  preview: Record<string, unknown>[];
}

export interface ProjectFiles {
  synthetic_csv: { exists: boolean; file: string | null };
  report_pdf: { exists: boolean; generated_at: string | null };
  messy_csv: { exists: boolean; file: string | null };
}

export interface MessyConfig {
  missing_values: number;
  duplicate_rows: number;
  whitespace: number;
  capitalization: number;
  format_inconsistency: number;
  invalid_values: number;
  outliers: number;
}

export interface MessyPreviewResponse {
  dataset: { rows: number; columns: number; total_cells: number; filename: string | null };
  requested_config: MessyConfig;
  estimates: {
    missing_cells: number;
    duplicate_rows: number;
    whitespace_cells: number;
    capitalization_cells: number;
    format_cells: number;
    invalid_cells: number;
    outlier_cells: number;
    text_columns: string[];
    numeric_columns: string[];
    formatable_columns: string[];
  };
}

export interface MessySummary {
  original_rows: number;
  original_columns: number;
  total_cells: number;
  requested: Record<string, number>;
  rows_after_duplicates: number;
  duplicate_rows_added: number;
  duplicate_rows_total_present: number;
  missing_cells_actual: number;
  missing_cells_target: number;
  counts: {
    missing_values: number;
    duplicate_rows: number;
    whitespace: number;
    capitalization: number;
    format_inconsistencies: number;
    invalid_values: number;
    outliers: number;
  };
  file?: string;
}

export interface MessyResult {
  file: string;
  summary: MessySummary;
  preview: Record<string, unknown>[];
}

export interface SessionMessyResult {
  summary: MessySummary;
  actual_percentages: { missing_values: number; duplicate_rows: number };
  preview: Record<string, unknown>[];
}

export interface ProjectSummary {
  id: number;
  name: string;
  mode: "upload" | "requirements";
  created_at: string;
  dataset_filename: string | null;
  rows_real: number | null;
  columns: number | null;
  target: string | null;
  task: string | null;
  target_overridden: boolean | null;
  synthesizer: string | null;
  generation_reason: string | null;
  rows_requested: number | null;
  rows_synthetic: number | null;
  fit_seconds: number | null;
  sample_seconds: number | null;
  report_generated_at: string | null;
  has_report: boolean;
  has_messy?: boolean;
  messy?: MessySummary | null;
}

export interface ProjectDetail extends ProjectSummary {
  analysis: Analysis | null;
  detection: Detection | null;
  evaluation: Evaluation | null;
  ml_utility: MLUtility | null;
  explanation: Explanation | null;
  requirement_spec: RequirementSpec | null;
  warnings: string[];
  synth_preview: { file?: string; rows?: Record<string, unknown>[] } | null;
}

export interface SaveProjectRequest {
  name: string;
  mode: "upload" | "requirements";
  /** Session whose generated synthetic CSV should be persisted (Mode A). */
  upload_id?: string | null;
  dataset_filename?: string | null;
  rows_real?: number | null;
  columns?: number | null;
  target?: string | null;
  task?: string | null;
  target_overridden?: boolean;
  synthesizer?: string | null;
  generation_reason?: string | null;
  rows_requested?: number | null;
  rows_synthetic?: number | null;
  fit_seconds?: number | null;
  sample_seconds?: number | null;
  analysis?: Analysis | null;
  detection?: Detection | null;
  evaluation?: Evaluation | null;
  ml_utility?: MLUtility | null;
  explanation?: Explanation | null;
  requirement_spec?: RequirementSpec | null;
  warnings?: string[];
}
