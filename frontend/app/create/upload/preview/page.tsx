"use client";

/**
 * Mode A — Generate + Preview step.
 * Shows what WILL happen (preview card) before the expensive generation,
 * then runs SDV generation and shows example synthetic rows.
 */
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { CircleAlert, CircleCheck, CircleX, Loader2, Play, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { StepHeader } from "@/components/workflow/step-header";
import { ActionBar } from "@/components/workflow/page-shell";
import { MessyOptionsSession } from "@/components/create/messy-options";
import { api, API_BASE, type GeneratePreviewBody } from "@/lib/api";
import type { GenerationPreview, GenerateResponse, Analysis, Detection } from "@/lib/types";

interface StoredConfig {
  upload_id: string;
  filename: string;
  n_rows: number;
  target: string;
  task: string;
  target_overridden: boolean;
  exclude_columns: string[];
  synthesizer: string | null;
  advanced: Record<string, unknown> | null;
  analysis: Analysis;
  detection: Detection;
}

const STAGES = [
  { key: "analyzing", label: "Analyzing your dataset", hint: "Understanding columns, types and relationships." },
  { key: "fitting", label: "Fitting the synthesizer", hint: "CTGAN can take a few minutes on larger datasets — this runs once." },
  { key: "sampling", label: "Sampling synthetic rows", hint: "Generating records from the fitted model." },
  { key: "validating", label: "Validating & checking uniqueness", hint: "Deduplication and format checks — no refitting needed." },
  { key: "complete", label: "Done", hint: "Your synthetic dataset is ready." },
];

function PreviewTable({ rows }: { rows: Record<string, unknown>[] }) {
  if (!rows.length) return null;
  const cols = Object.keys(rows[0]);
  return (
    <div className="overflow-auto rounded-md border">
      <Table>
        <TableHeader className="bg-muted">
          <TableRow>
            {cols.map((c) => (
              <TableHead key={c}>{c}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r, i) => (
            <TableRow key={i}>
              {cols.map((c) => (
                <TableCell key={c} className="whitespace-nowrap">
                  {r[c] === null || r[c] === undefined ? "—" : String(r[c])}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

export default function UploadPreviewPage() {
  const router = useRouter();
  const [config, setConfig] = useState<StoredConfig | null>(null);
  const [preview, setPreview] = useState<GenerationPreview | null>(null);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [stage, setStage] = useState(0);
  const [slow, setSlow] = useState(false);

  useEffect(() => {
    const raw = sessionStorage.getItem("datanova-config");
    if (!raw) {
      router.replace("/create/upload");
      return;
    }
    const cfg: StoredConfig = JSON.parse(raw);
    setConfig(cfg);
    api
      .preview(cfg.upload_id, {
        n_rows: cfg.n_rows,
        target: cfg.target,
        task: cfg.task,
        exclude_columns: cfg.exclude_columns,
        synthesizer: cfg.synthesizer,
        advanced: cfg.advanced,
      } as GeneratePreviewBody)
      .then(setPreview)
      .catch((e) => setError(e instanceof Error ? e.message : "Preview failed"));
  }, [router]);

  async function runGeneration() {
    if (!config) return;
    setGenerating(true);
    setError(null);
    setStage(0);
    setSlow(false);
    const startedAt = Date.now();
    const slowTimer = setTimeout(() => setSlow(true), 30_000);
    // Poll the backend's live generation status while the request runs.
    const pollTimer = setInterval(async () => {
      try {
        const res = await fetch(
  `${API_BASE}/api/upload/${config.upload_id}/generation-status`
      );
        if (!res.ok) return;
        const data = await res.json();
        const idx = STAGES.findIndex((s) => s.key === data.stage);
        if (idx >= 0) setStage(idx);
      } catch {
        /* status polling is best-effort */
      }
    }, 1_500);
    try {
      const res = await api.generate(config.upload_id, {
        n_rows: config.n_rows,
        target: config.target,
        task: config.task,
        exclude_columns: config.exclude_columns,
        synthesizer: config.synthesizer,
        advanced: config.advanced,
      } as GeneratePreviewBody);
      clearTimeout(slowTimer);
      setStage(STAGES.length - 1);
      setResult(res);
    } catch (e) {
      clearTimeout(slowTimer);
      setError(e instanceof Error ? e.message : "Generation failed");
    } finally {
      clearInterval(pollTimer);
      setGenerating(false);
      void startedAt;
    }
  }

  function continueToEvaluate() {
    sessionStorage.setItem("datanova-generation", JSON.stringify(result));
    router.push("/create/upload/evaluate");
  }

  if (!config) return null;

  return (
    <div className="flex min-h-screen flex-col">
      <StepHeader
        step={result ? "Preview" : "Generate"}
        title={result ? "Generation complete" : "Generation preview"}
        subtitle={
          result
            ? "Review the generated sample rows, then continue to evaluation."
            : "Confirm the plan before DataNova fits a synthesizer on your data."
        }
      />

      <div className="mx-auto w-full max-w-6xl flex-1 space-y-6 p-6">
        {!result && preview && (
          <>
            <div className="grid gap-4 md:grid-cols-2">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">What will happen</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="flex justify-between"><span className="text-muted-foreground">Rows to generate</span><span className="font-medium">{preview.rows_requested.toLocaleString()}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Target</span><span className="font-medium">{preview.target ?? "—"}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Task</span><span className="font-medium">{preview.task ?? "—"}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Features used</span><span className="font-medium">{preview.n_features}</span></div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Excluded columns</span>
                    <span className="max-w-[60%] text-right font-medium">{preview.excluded_columns.join(", ") || "none"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Approach</span>
                    <Badge>{preview.approach.synthesizer}</Badge>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Why this synthesizer?</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm text-muted-foreground">
                  <p>{preview.approach.reason}</p>
                  {preview.health_warnings.length > 0 && (
                    <Alert variant="warning">
                      <CircleAlert className="h-4 w-4" />
                      <AlertTitle>Health warnings carry over</AlertTitle>
                      <AlertDescription>
                        <ul className="list-inside list-disc">
                          {preview.health_warnings.slice(0, 3).map((w, i) => (
                            <li key={i}>{w}</li>
                          ))}
                        </ul>
                      </AlertDescription>
                    </Alert>
                  )}
                  <ul className="list-inside list-disc text-xs">
                    {preview.notes.map((n, i) => (
                      <li key={i}>{n}</li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Your real data (first 5 rows)</CardTitle>
              </CardHeader>
              <CardContent>
                <PreviewTable rows={preview.sample_preview} />
              </CardContent>
            </Card>
          </>
        )}

        {generating && (
          <Card>
            <CardContent className="flex flex-col items-center gap-4 py-12">
              <Loader2 className="h-10 w-10 animate-spin text-brand-red" />
              <p className="font-medium">Generating your dataset…</p>
              <ol className="w-full max-w-md space-y-2">
                {STAGES.slice(0, 4).map((s, i) => {
                  const state = i < stage ? "done" : i === stage ? "active" : "todo";
                  return (
                    <li key={s.key} className="flex items-start gap-2.5 text-sm">
                      {state === "done" ? (
                        <CircleCheck className="mt-0.5 h-4 w-4 shrink-0 text-brand-red" />
                      ) : state === "active" ? (
                        <Loader2 className="mt-0.5 h-4 w-4 shrink-0 animate-spin text-brand-red" />
                      ) : (
                        <span className="mt-0.5 h-4 w-4 shrink-0 rounded-full border border-border" />
                      )}
                      <span>
                        <span className={state === "todo" ? "text-muted-foreground" : "font-medium"}>{s.label}</span>
                        {state === "active" && (
                          <span className="block text-xs text-muted-foreground">{s.hint}</span>
                        )}
                      </span>
                    </li>
                  );
                })}
              </ol>
              {slow && (
                <p className="max-w-md text-center text-xs text-muted-foreground">
                  This is taking longer than expected. The synthesizer is still working — complex datasets simply
                  need more fitting time. You can safely keep this tab open.
                </p>
              )}
            </CardContent>
          </Card>
        )}

        {error && (
          <Alert variant="destructive">
            <CircleAlert className="h-4 w-4" />
            <AlertTitle>Generation error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {result && (
          <>
            <div className="grid gap-4 sm:grid-cols-4">
              <Card>
                <CardContent className="p-4">
                  <div className="text-xs text-muted-foreground">Rows generated</div>
                  <div className="text-xl font-bold">{result.n_rows_generated.toLocaleString()}</div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4">
                  <div className="text-xs text-muted-foreground">Synthesizer</div>
                  <div className="text-xl font-bold">{result.approach}</div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4">
                  <div className="text-xs text-muted-foreground">Fit time</div>
                  <div className="text-xl font-bold">{result.fit_seconds}s</div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4">
                  <div className="text-xs text-muted-foreground">Sample time</div>
                  <div className="text-xl font-bold">{result.sample_seconds}s</div>
                </CardContent>
              </Card>
            </div>

            {result.validation_warnings.length > 0 && (
              <Alert variant="warning">
                <CircleAlert className="h-4 w-4" />
                <AlertTitle>Generation validation warnings</AlertTitle>
                <AlertDescription>
                  <ul className="list-inside list-disc">
                    {result.validation_warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}

            {/* Generation Quality Check — computed by the backend validator */}
            {result.quality_check && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <ShieldCheck className="h-4 w-4 text-brand-red" /> Generation Quality Check{" "}
                    <Badge variant={result.quality_check.passed ? "success" : "destructive"}>
                      {result.quality_check.passed ? "Passed" : "Issues found"}
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid gap-3 sm:grid-cols-4">
                    <div className="rounded-lg border p-3">
                      <div className="text-xs text-muted-foreground">Rows generated</div>
                      <div className="text-xl font-bold">{result.quality_check.summary.rows.toLocaleString()}</div>
                    </div>
                    <div className="rounded-lg border p-3">
                      <div className="text-xs text-muted-foreground">Unique rows</div>
                      <div className="text-xl font-bold text-brand-red">{result.quality_check.summary.unique_rows.toLocaleString()}</div>
                    </div>
                    <div className="rounded-lg border p-3">
                      <div className="text-xs text-muted-foreground">Duplicate rows</div>
                      <div className="text-xl font-bold">{result.quality_check.summary.duplicate_rows}</div>
                    </div>
                    <div className="rounded-lg border p-3">
                      <div className="text-xs text-muted-foreground">Format violations</div>
                      <div className="text-xl font-bold">{result.quality_check.summary.format_violations}</div>
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
                </CardContent>
              </Card>
            )}

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Example synthetic rows (first {result.preview.length})</CardTitle>
              </CardHeader>
              <CardContent>
                <PreviewTable rows={result.preview} />
              </CardContent>
            </Card>

            {/* Optional realistic data-quality issues — applied to a SEPARATE
                copy after generation; the clean dataset stays available. */}
            <MessyOptionsSession uploadId={result.upload_id} rows={result.n_rows_generated} />
          </>
        )}

        {!preview && !error && (
          <div className="space-y-3">
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-48 w-full" />
          </div>
        )}
      </div>

      <ActionBar>
        <Button variant="outline" onClick={() => router.push("/create/upload")}>
          Change generation settings
        </Button>
        {!result ? (
          <Button onClick={runGeneration} disabled={generating || !preview}>
            {generating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Generate synthetic data
          </Button>
        ) : (
          <Button onClick={continueToEvaluate}>Continue to evaluation</Button>
        )}
      </ActionBar>
    </div>
  );
}
