/**
 * DataNova API client.
 *
 * Single place where the frontend talks to the FastAPI backend. Centralising
 * fetch logic keeps pages clean and makes the backend base URL configurable.
 */
import type {
  Detection,
  Evaluation,
  Explanation,
  MLUtility,
  ModelRecommendation,
  ProjectDetail,
  ProjectSummary,
  RequirementGenerateResponse,
  RequirementSpec,
  SaveProjectRequest,
  UploadResponse,
  GenerationPreview,
  GenerateResponse,
  ProjectFiles,
  MessyConfig,
  MessyPreviewResponse,
  MessyResult,
  SessionMessyResult,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* keep statusText */
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handle<T>(res);
}

export const api = {
  health: () => fetch(`${API_BASE}/api/health`).then((r) => r.json()),

  // ---- Mode A -------------------------------------------------------------
  uploadDataset(file: File): Promise<UploadResponse> {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${API_BASE}/api/upload`, { method: "POST", body: form }).then((r) =>
      handle<UploadResponse>(r)
    );
  },

  redetect(uploadId: string, target: string | null): Promise<Detection> {
    return post(`/api/upload/${uploadId}/detect`, { target });
  },

  preview(uploadId: string, body: GeneratePreviewBody): Promise<GenerationPreview> {
    return post(`/api/upload/${uploadId}/preview`, body);
  },

  generate(uploadId: string, body: GeneratePreviewBody): Promise<GenerateResponse> {
    return post(`/api/upload/${uploadId}/generate`, body);
  },

  downloadSessionCsv(uploadId: string) {
    window.open(`${API_BASE}/api/upload/${uploadId}/synthetic/download`, "_blank");
  },

  sessionMessy(uploadId: string, cfg: MessyConfig): Promise<SessionMessyResult> {
    return post(`/api/upload/${uploadId}/messy`, cfg);
  },

  downloadSessionMessyCsv(uploadId: string) {
    window.open(`${API_BASE}/api/upload/${uploadId}/messy/download`, "_blank");
  },

  // ---- Evaluation & ML ------------------------------------------------------
  evaluate(body: { upload_id?: string; project_id?: number; target?: string | null }): Promise<Evaluation> {
    return post("/api/evaluate", body);
  },

  recommendModel(body: { upload_id: string; task: string; target: string }): Promise<ModelRecommendation> {
    return post("/api/ml-utility/recommendation", body);
  },

  runMlUtility(body: {
    upload_id?: string;
    project_id?: number;
    target: string;
    task: string;
    model_key?: string | null;
  }): Promise<MLUtility> {
    return post("/api/ml-utility/run", body);
  },

  explain(body: {
    evaluation?: Evaluation | null;
    ml_utility?: MLUtility | null;
    generation_approach?: string | null;
    generation_reason?: string | null;
    warnings?: string[] | null;
  }): Promise<Explanation> {
    return post("/api/explain", body);
  },

  // ---- Mode B ---------------------------------------------------------------
  parseRequirements(text: string): Promise<{ spec: RequirementSpec }> {
    return post("/api/requirements/parse", { text });
  },

  generateFromRequirements(spec: RequirementSpec, seed = 42): Promise<RequirementGenerateResponse> {
    return post("/api/requirements/generate", { spec, seed });
  },

  /** Download the generated CSV for a Mode B session. */
  downloadReqSessionCsv(sessionId: string) {
    window.open(`${API_BASE}/api/upload/${sessionId}/synthetic/download`, "_blank");
  },

  // ---- Projects ---------------------------------------------------------------
  listProjects(): Promise<{ projects: ProjectSummary[] }> {
    return fetch(`${API_BASE}/api/projects`).then((r) => handle<{ projects: ProjectSummary[] }>(r));
  },

  getProject(id: number): Promise<ProjectDetail> {
    return fetch(`${API_BASE}/api/projects/${id}`).then((r) => handle<ProjectDetail>(r));
  },

  deleteProject(id: number): Promise<{ deleted: boolean }> {
    return fetch(`${API_BASE}/api/projects/${id}`, { method: "DELETE" }).then((r) =>
      handle<{ deleted: boolean }>(r)
    );
  },

  renameProject(id: number, name: string): Promise<{ id: number; name: string }> {
    return fetch(`${API_BASE}/api/projects/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }).then((r) => handle<{ id: number; name: string }>(r));
  },

  projectFiles(id: number): Promise<ProjectFiles> {
    return fetch(`${API_BASE}/api/projects/${id}/files`).then((r) => handle<ProjectFiles>(r));
  },

  saveProject(body: SaveProjectRequest): Promise<{ project_id: number; synthetic_file: string | null }> {
    return post("/api/projects/save", body);
  },

  generateReport(projectId: number): Promise<{ report_path: string; filename: string }> {
    return post(`/api/projects/${projectId}/report`, {});
  },

  downloadReport(projectId: number) {
    window.open(`${API_BASE}/api/projects/${projectId}/report/download`, "_blank");
  },

  downloadProjectCsv(projectId: number) {
    window.open(`${API_BASE}/api/projects/${projectId}/synthetic/download`, "_blank");
  },

  downloadMessyCsv(projectId: number) {
    window.open(`${API_BASE}/api/projects/${projectId}/messy/download`, "_blank");
  },

  messyPreview(projectId: number, cfg: MessyConfig): Promise<MessyPreviewResponse> {
    return post(`/api/projects/${projectId}/messy/preview`, cfg);
  },

  messyGenerate(projectId: number, cfg: MessyConfig): Promise<MessyResult> {
    return post(`/api/projects/${projectId}/messy`, cfg);
  },
};

export interface GeneratePreviewBody {
  n_rows: number;
  target?: string | null;
  task?: string | null;
  exclude_columns?: string[] | null;
  synthesizer?: string | null;
  advanced?: Record<string, unknown> | null;
}
