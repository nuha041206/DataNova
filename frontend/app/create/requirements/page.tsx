"use client";

/**
 * Mode B — "Tell DataNova What You Want".
 * Free text -> parsed schema -> "Here's what DataNova understood" editor
 * (per-column technical + semantic type, ranges, categories, uniqueness)
 * -> staged generation with quality check.
 */
import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  CircleAlert,
  CircleCheck,
  CircleX,
  Loader2,
  Plus,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Table2,
  Trash2,
  Wand2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { StepHeader } from "@/components/workflow/step-header";
import { ActionBar } from "@/components/workflow/page-shell";
import { api } from "@/lib/api";
import {
  SEMANTIC_LABELS,
  SEMANTIC_TYPES,
  TECHNICAL_TYPES,
  type ColumnSpec,
  type GenerationQuality,
  type RequirementSpec,
} from "@/lib/types";

const EXAMPLES = [
  "Create 2000 unique Indian customer records with name, gender, age, address, city, state, Indian phone number, email and salary. Age should be between 18 and 65.",
  "Generate 2000 employee records with employee name, department with values such as HR, Finance, IT, Sales, age and salary.",
  "Create a student dataset with name, age, gender and marks. Marks should be between 0 and 100.",
];

const GEN_STAGES = [
  "Understanding schema",
  "Generating records",
  "Checking uniqueness",
  "Validating generated data",
];

const TECH_FOR_SEMANTIC: Record<string, string> = {
  person_name: "string", gender: "categorical", address: "string", city: "string",
  state: "categorical", country: "categorical", indian_phone: "string", email: "string",
  indian_pincode: "string", date_of_birth: "date", date: "date", datetime: "datetime",
  age: "integer", salary: "float", currency: "float", percentage: "float",
  customer_id: "string", generic_id: "string", product_name: "string",
  company_name: "string", category: "categorical", boolean: "boolean",
  integer: "integer", float: "float", generic_text: "string",
};

interface ReqResult {
  spec: RequirementSpec;
  summary: { rows: number; columns: string[]; target: string; task: string };
  description: string;
  warnings: string[];
  uniqueness: { requested: number; generated: number; unique_rows: number; duplicate_rows: number; uniqueness_fully_satisfied: boolean; notes: string[] };
  quality_check: GenerationQuality;
  session_id: string;
  preview: Record<string, unknown>[];
}

export default function RequirementsPage() {
  const router = useRouter();

  const [text, setText] = useState("");
  const [spec, setSpec] = useState<RequirementSpec | null>(null);
  const [columns, setColumns] = useState<ColumnSpec[]>([]);
  const [rows, setRows] = useState("");
  const [unique, setUnique] = useState(true);
  const [task, setTask] = useState("none");
  const [target, setTarget] = useState("");
  const [positivePct, setPositivePct] = useState("");
  const [parsing, setParsing] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [stage, setStage] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ReqResult | null>(null);

  function applySpec(s: RequirementSpec) {
    setSpec(s);
    setColumns(s.columns?.length ? s.columns.map((c) => ({ ...c })) : []);
    setRows(s.rows ? String(s.rows) : "2000");
    setUnique(s.unique);
    setTask(s.task === "unknown" ? "none" : s.task);
    setTarget(s.target ?? "");
    setPositivePct(s.class_distribution?.positive_pct ? String(s.class_distribution.positive_pct) : "");
  }

  async function parse() {
    if (!text.trim()) return;
    setParsing(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.parseRequirements(text);
      applySpec(res.spec);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Parsing failed");
    } finally {
      setParsing(false);
    }
  }

  function updateColumn(idx: number, patch: Partial<ColumnSpec>) {
    setColumns((prev) => prev.map((c, i) => (i === idx ? { ...c, ...patch } : c)));
  }

  function addColumn() {
    setColumns((prev) => [
      ...prev,
      { name: `Column ${prev.length + 1}`, technical_type: "string", semantic_type: "generic_text",
        min: null, max: null, categories: null, distribution: null, unique: false, detected: false },
    ]);
  }

  function removeColumn(idx: number) {
    setColumns((prev) => prev.filter((_, i) => i !== idx));
  }

  async function generate() {
    setGenerating(true);
    setError(null);
    setResult(null);
    setStage(0);
    const stageTimer = setInterval(() => setStage((s) => Math.min(s + 1, GEN_STAGES.length - 1)), 1200);
    try {
      const specOut: RequirementSpec = {
        rows: parseInt(rows, 10) || 0,
        task: task === "none" ? "unknown" : task,
        target: target || null,
        features: [],
        columns: columns.map((c) => ({ ...c, categories: c.categories && c.categories.length ? c.categories : null })),
        unique,
        country: spec?.country ?? null,
        constraints: spec?.constraints ?? [],
        class_distribution: positivePct && task === "classification"
          ? { positive_pct: parseInt(positivePct, 10), kind: "positive_share" } : null,
        domain: spec?.domain ?? null,
        clarifications: [],
        warnings: [],
        raw_text: text,
      };
      const res = await api.generateFromRequirements(specOut);
      setResult(res as ReqResult);
      setStage(GEN_STAGES.length);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed");
      setStage(0);
    } finally {
      clearInterval(stageTimer);
      setGenerating(false);
    }
  }

  return (
    <div className="flex min-h-screen flex-col">
      <StepHeader
        step="Configure"
        title="Tell DataNova What You Want"
        subtitle="Describe the dataset in plain English — DataNova builds an editable schema before generating anything."
      />

      <div className="mx-auto w-full max-w-6xl flex-1 space-y-6 p-6">
        {/* Step 1: describe */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-base">
              <Wand2 className="h-4 w-4 text-brand-red" /> 1 · Describe Your Dataset
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Textarea
              rows={3}
              placeholder={EXAMPLES[0]}
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-muted-foreground">Try:</span>
              {EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => setText(ex)}
                  className="max-w-[280px] truncate rounded-full border px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:border-brand-red/50 hover:text-foreground"
                  title={ex}
                >
                  {ex.slice(0, 48)}…
                </button>
              ))}
            </div>
            <div className="flex justify-end">
              <Button onClick={parse} disabled={parsing || !text.trim()}>
                {parsing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                Interpret Requirements
              </Button>
            </div>
          </CardContent>
        </Card>

        {error && (
          <Alert variant="destructive">
            <CircleAlert className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Step 2: confirm schema */}
        {spec && (
          <>
            {spec.clarifications.length > 0 && (
              <Alert variant="warning">
                <CircleAlert className="h-4 w-4" />
                <AlertTitle>DataNova needs a few answers</AlertTitle>
                <AlertDescription>
                  <ul className="list-inside list-disc space-y-1">
                    {spec.clarifications.map((c) => (
                      <li key={c.field}>{c.question}</li>
                    ))}
                  </ul>
                  <p className="mt-1 text-xs">Answer by editing below — nothing is generated until you confirm.</p>
                </AlertDescription>
              </Alert>
            )}

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-base">
                  <Table2 className="h-4 w-4 text-brand-red" /> 2 · Here&apos;s What DataNova Understood
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* dataset-level settings */}
                <div className="grid gap-4 sm:grid-cols-4">
                  <div className="space-y-1.5">
                    <Label htmlFor="rows">Rows</Label>
                    <Input id="rows" type="number" min={10} max={200000} value={rows} onChange={(e) => setRows(e.target.value)} />
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="unique-toggle">Unique records</Label>
                    <label className="flex h-9 items-center gap-2 rounded-md border px-3 text-sm">
                      <input type="checkbox" checked={unique} onChange={(e) => setUnique(e.target.checked)} />
                      {unique ? "Enforce uniqueness" : "Allow duplicates"}
                    </label>
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="task">ML target task (optional)</Label>
                    <select id="task" className="flex h-9 w-full rounded-md border border-input bg-transparent px-2 py-1 text-sm"
                      value={task} onChange={(e) => setTask(e.target.value)}>
                      <option value="none">No ML target</option>
                      <option value="classification">Classification</option>
                      <option value="regression">Regression</option>
                    </select>
                  </div>
                  {task !== "none" && (
                    <div className="space-y-1.5">
                      <Label htmlFor="target">Target column name</Label>
                      <Input id="target" value={target} onChange={(e) => setTarget(e.target.value)} placeholder="target" />
                    </div>
                  )}
                </div>
                {task === "classification" && (
                  <div className="max-w-xs space-y-1.5">
                    <Label htmlFor="pospct">Positive class share (%) — optional</Label>
                    <Input id="pospct" type="number" min={1} max={99} value={positivePct} onChange={(e) => setPositivePct(e.target.value)} placeholder="e.g. 20" />
                  </div>
                )}

                {/* schema editor table */}
                <div className="overflow-x-auto rounded-lg border">
                  <table className="w-full min-w-[760px] text-sm">
                    <thead className="bg-muted text-left text-xs uppercase tracking-wide text-muted-foreground">
                      <tr>
                        <th className="p-2.5 font-medium">Column</th>
                        <th className="p-2.5 font-medium">Technical type</th>
                        <th className="p-2.5 font-medium">Semantic type</th>
                        <th className="p-2.5 font-medium">Rules</th>
                        <th className="w-10 p-2.5"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {columns.map((col, idx) => (
                        <tr key={idx} className="border-t align-top">
                          <td className="p-2">
                            <Input
                              value={col.name}
                              onChange={(e) => updateColumn(idx, { name: e.target.value })}
                              className="h-8 w-36"
                            />
                            {col.detected && (
                              <span className="mt-1 block text-[10px] text-muted-foreground">Detected automatically</span>
                            )}
                          </td>
                          <td className="p-2">
                            <select
                              className="flex h-8 w-32 rounded-md border border-input bg-transparent px-2 text-xs"
                              value={col.technical_type}
                              onChange={(e) => updateColumn(idx, { technical_type: e.target.value })}
                            >
                              {TECHNICAL_TYPES.map((t) => (
                                <option key={t} value={t}>{t}</option>
                              ))}
                            </select>
                          </td>
                          <td className="p-2">
                            <select
                              className="flex h-8 w-44 rounded-md border border-input bg-transparent px-2 text-xs"
                              value={col.semantic_type}
                              onChange={(e) => {
                                const sem = e.target.value;
                                updateColumn(idx, {
                                  semantic_type: sem,
                                  technical_type: TECH_FOR_SEMANTIC[sem] ?? col.technical_type,
                                });
                              }}
                            >
                              {SEMANTIC_TYPES.map((s) => (
                                <option key={s} value={s}>{SEMANTIC_LABELS[s] ?? s}</option>
                              ))}
                            </select>
                            <span className="mt-1 block max-w-44 truncate text-[10px] text-muted-foreground" title={col.detection_reason ?? ""}>
                              {SEMANTIC_LABELS[col.semantic_type] ?? col.semantic_type}
                            </span>
                          </td>
                          <td className="p-2">
                            <div className="flex flex-wrap items-center gap-2 text-xs">
                              {(col.semantic_type === "age" || col.semantic_type === "integer" ||
                                col.semantic_type === "float" || col.semantic_type === "salary" ||
                                col.semantic_type === "currency" || col.semantic_type === "percentage") && (
                                <span className="flex items-center gap-1">
                                  <Input
                                    type="number" placeholder="min" className="h-7 w-20"
                                    value={col.min ?? ""}
                                    onChange={(e) => updateColumn(idx, { min: e.target.value === "" ? null : Number(e.target.value) })}
                                  />
                                  <span>–</span>
                                  <Input
                                    type="number" placeholder="max" className="h-7 w-20"
                                    value={col.max ?? ""}
                                    onChange={(e) => updateColumn(idx, { max: e.target.value === "" ? null : Number(e.target.value) })}
                                  />
                                </span>
                              )}
                              <label className="flex items-center gap-1.5">
                                <input
                                  type="checkbox"
                                  checked={col.unique}
                                  onChange={(e) => updateColumn(idx, { unique: e.target.checked })}
                                />
                                unique values
                              </label>
                            </div>
                          </td>
                          <td className="p-2">
                            <Button variant="ghost" size="icon" className="h-8 w-8" title="Remove column" onClick={() => removeColumn(idx)}>
                              <Trash2 className="h-3.5 w-3.5 text-muted-foreground" />
                            </Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="flex flex-wrap items-center justify-between gap-3">
                  <Button variant="outline" size="sm" onClick={addColumn}>
                    <Plus className="h-4 w-4" /> Add Column
                  </Button>
                  <Button variant="ghost" size="sm" onClick={() => applySpec(spec)}>
                    <RotateCcw className="h-4 w-4" /> Reset to parsed schema
                  </Button>
                </div>
              </CardContent>
            </Card>
          </>
        )}

        {/* Staged generation progress */}
        {generating && (
          <Card>
            <CardContent className="space-y-3 py-6">
              <div className="flex items-center gap-3">
                <Loader2 className="h-5 w-5 animate-spin text-brand-red" />
                <span className="font-medium">Generating your dataset…</span>
              </div>
              <ol className="space-y-1.5 pl-1 text-sm">
                {GEN_STAGES.map((s, i) => (
                  <li key={s} className="flex items-center gap-2">
                    {i < stage ? (
                      <CircleCheck className="h-4 w-4 text-brand-red" />
                    ) : i === stage ? (
                      <Loader2 className="h-4 w-4 animate-spin text-brand-red" />
                    ) : (
                      <span className="h-4 w-4 rounded-full border border-muted-foreground/30" />
                    )}
                    <span className={i <= stage ? "text-foreground" : "text-muted-foreground"}>Step {i + 1}/4 — {s}</span>
                  </li>
                ))}
              </ol>
            </CardContent>
          </Card>
        )}

        {/* Step 3: results + quality check */}
        {result && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <ShieldCheck className="h-4 w-4 text-brand-red" /> 3 · Generation Quality Check
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-3 sm:grid-cols-4">
                <div className="rounded-lg border p-3">
                  <div className="text-xs text-muted-foreground">Rows generated</div>
                  <div className="text-xl font-bold">{result.summary.rows.toLocaleString()}</div>
                </div>
                <div className="rounded-lg border p-3">
                  <div className="text-xs text-muted-foreground">Unique rows</div>
                  <div className="text-xl font-bold text-brand-red">
                    {result.quality_check.summary.unique_rows.toLocaleString()}
                  </div>
                </div>
                <div className="rounded-lg border p-3">
                  <div className="text-xs text-muted-foreground">Duplicate rows</div>
                  <div className="text-xl font-bold">{result.quality_check.summary.duplicate_rows}</div>
                </div>
                <div className="rounded-lg border p-3">
                  <div className="text-xs text-muted-foreground">Quality check</div>
                  <div className={`text-xl font-bold ${result.quality_check.passed ? "text-brand-red" : "text-red-600"}`}>
                    {result.quality_check.passed ? "Passed" : "Failed"}
                  </div>
                </div>
              </div>

              <ul className="space-y-1.5">
                {result.quality_check.checks.map((c, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    {c.passed ? (
                      <CircleCheck className="mt-0.5 h-4 w-4 shrink-0 text-brand-red" />
                    ) : (
                      <CircleX className="mt-0.5 h-4 w-4 shrink-0 text-red-600" />
                    )}
                    <span>
                      <b>{c.check}:</b> <span className="text-muted-foreground">{c.detail}</span>
                    </span>
                  </li>
                ))}
              </ul>

              {result.warnings.length > 0 && (
                <Alert variant="warning">
                  <CircleAlert className="h-4 w-4" />
                  <AlertTitle>Generation notes</AlertTitle>
                  <AlertDescription>
                    <ul className="list-inside list-disc">
                      {result.warnings.map((w, i) => (
                        <li key={i}>{w}</li>
                      ))}
                    </ul>
                  </AlertDescription>
                </Alert>
              )}

              {/* example rows */}
              <div className="overflow-x-auto rounded-lg border">
                <table className="w-full text-xs">
                  <thead className="bg-muted text-left">
                    <tr>
                      {result.summary.columns.map((c) => (
                        <th key={c} className="whitespace-nowrap p-2 font-medium">{c}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.preview.slice(0, 8).map((r, i) => (
                      <tr key={i} className="border-t">
                        {result.summary.columns.map((c) => (
                          <td key={c} className="whitespace-nowrap p-2">{String(r[c] ?? "—")}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="flex flex-wrap gap-2">
                <Button onClick={() => {
                  sessionStorage.setItem("datanova-req-result", JSON.stringify(result));
                  router.push("/create/requirements/evaluate");
                }}>
                  Continue to Evaluate <Badge variant="secondary" className="ml-2">Step 4</Badge>
                </Button>
                <Button variant="outline" onClick={() => api.downloadReqSessionCsv(result.session_id)}>
                  Download CSV
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      <ActionBar>
        <Button variant="outline" onClick={() => router.push("/create")}>Back</Button>
        <Button onClick={generate} disabled={generating || !spec || columns.length === 0 || !(parseInt(rows, 10) > 0)}>
          {generating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Wand2 className="h-4 w-4" />}
          Generate Dataset
        </Button>
      </ActionBar>
    </div>
  );
}
