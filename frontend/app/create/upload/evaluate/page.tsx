"use client";

/**
 * Mode A — Evaluate + Export step.
 * Runs evaluation and the ML utility experiment, shows explanations,
 * then saves the project and offers CSV/PDF downloads.
 */
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { CircleAlert, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useToast } from "@/components/ui/toast";
import { StepHeader } from "@/components/workflow/step-header";
import { ActionBar } from "@/components/workflow/page-shell";
import { EvaluationView } from "@/components/workflow/evaluation-view";
import { api } from "@/lib/api";
import type { Evaluation, Explanation, MLUtility, ModelRecommendation, SaveProjectRequest } from "@/lib/types";
import { INTERPRETATION_NOTES } from "@/lib/notes";

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
  analysis: unknown;
  detection: unknown;
}

interface StoredGeneration {
  approach: string;
  reason: string;
  fit_seconds: number;
  sample_seconds: number;
  n_rows_generated: number;
  validation_warnings: string[];
}

export default function UploadEvaluatePage() {
  const router = useRouter();
  const { toast } = useToast();

  const [projectName, setProjectName] = useState("");
  const [config, setConfig] = useState<StoredConfig | null>(null);
  const [generation, setGeneration] = useState<StoredGeneration | null>(null);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [ml, setMl] = useState<MLUtility | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [rec, setRec] = useState<ModelRecommendation | null>(null);
  const [modelKey, setModelKey] = useState<string>("__recommended__");
  const [error, setError] = useState<string | null>(null);
  const [phase, setPhase] = useState<"evaluating" | "ml" | "done">("evaluating");
  const [saving, setSaving] = useState(false);
  const [savedId, setSavedId] = useState<number | null>(null);

  useEffect(() => {
    const rawCfg = sessionStorage.getItem("datanova-config");
    const rawGen = sessionStorage.getItem("datanova-generation");
    if (!rawCfg || !rawGen) {
      router.replace("/create/upload");
      return;
    }
    const cfg: StoredConfig = JSON.parse(rawCfg);
    const gen: StoredGeneration = JSON.parse(rawGen);
    setConfig(cfg);
    setGeneration(gen);

    // 1) statistical evaluation
    let evalResult: Evaluation | null = null;
    api
      .evaluate({ upload_id: cfg.upload_id, target: cfg.target })
      .then((ev) => {
        evalResult = ev;
        setEvaluation(ev);
        setPhase("ml");
        // 2) model recommendation + ML utility
        return api.recommendModel({ upload_id: cfg.upload_id, task: cfg.task, target: cfg.target });
      })
      .then((recommendation) => {
        setRec(recommendation);
        return api.runMlUtility({
          upload_id: cfg.upload_id,
          target: cfg.target,
          task: cfg.task,
          model_key: recommendation.model_key,
        });
      })
      .then((mlRes) => {
        setMl(mlRes);
        // 3) explanations from the computed results
        return api.explain({
          evaluation: evalResult,
          ml_utility: mlRes,
          generation_approach: gen.approach,
          generation_reason: gen.reason,
          warnings: gen.validation_warnings,
        });
      })
      .then(setExplanation)
      .catch((e) => setError(e instanceof Error ? e.message : "Evaluation failed"))
      .finally(() => setPhase("done"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [router]);

  async function rerunWithModel(key: string) {
    if (!config) return;
    setModelKey(key);
    setError(null);
    try {
      const mlRes = await api.runMlUtility({
        upload_id: config.upload_id,
        target: config.target,
        task: config.task,
        model_key: key === "__recommended__" ? rec?.model_key ?? null : key,
      });
      setMl(mlRes);
      const exp = await api.explain({
        evaluation,
        ml_utility: mlRes,
        generation_approach: generation?.approach,
        generation_reason: generation?.reason,
        warnings: generation?.validation_warnings,
      });
      setExplanation(exp);
    } catch (e) {
      setError(e instanceof Error ? e.message : "ML test failed");
    }
  }

  async function saveAndExport() {
    if (!config || !generation) return;
    setSaving(true);
    try {
      const res = await api.saveProject({
        name: projectName.trim() || `${config.filename.replace(/\.csv$/i, "")} — expanded`,
        mode: "upload",
        upload_id: config.upload_id,
        dataset_filename: config.filename,
        // Persist the full health check + detection so the project record (and
        // the PDF report) contains the complete dataset information.
        rows_real: config.analysis ? (config.analysis as { rows?: number }).rows ?? undefined : undefined,
        columns: config.analysis ? (config.analysis as { columns?: number }).columns ?? undefined : undefined,
        analysis: (config.analysis ?? null) as SaveProjectRequest["analysis"],
        detection: (config.detection ?? null) as SaveProjectRequest["detection"],
        target: config.target,
        task: config.task,
        target_overridden: config.target_overridden,
        synthesizer: generation.approach,
        generation_reason: generation.reason,
        rows_requested: config.n_rows,
        rows_synthetic: generation.n_rows_generated,
        fit_seconds: generation.fit_seconds,
        sample_seconds: generation.sample_seconds,
        evaluation,
        ml_utility: ml,
        explanation,
        warnings: generation.validation_warnings,
      });
      setSavedId(res.project_id);
      toast({
        title: "Project saved",
        description: "Generating the PDF report…",
        variant: "success",
      });
      // auto-generate the PDF report right away
      await api.generateReport(res.project_id);
      toast({ title: "PDF report ready", description: "Open the project to download it.", variant: "success" });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Saving project failed");
      toast({
        title: "Could not save project",
        description: e instanceof Error ? e.message : undefined,
        variant: "destructive",
      });
    } finally {
      setSaving(false);
    }
  }

  if (!config || !generation) return null;

  return (
    <div className="flex min-h-screen flex-col">
      <StepHeader
        step="Evaluate"
        title="Compare real vs synthetic"
        subtitle="Statistical comparison, distribution charts and the ML utility experiment — all computed from your data."
      />

      <div className="mx-auto w-full max-w-6xl flex-1 space-y-6 p-6">
        {(phase === "evaluating" || phase === "ml") && (
          <Card>
            <CardContent className="flex items-center gap-3 py-10">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
              <span className="text-sm text-muted-foreground">
                {phase === "evaluating" ? "Running statistical evaluation…" : "Running ML utility experiments…"}
              </span>
            </CardContent>
          </Card>
        )}

        {error && (
          <Alert variant="destructive">
            <CircleAlert className="h-4 w-4" />
            <AlertTitle>Evaluation error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {evaluation && <EvaluationView evaluation={evaluation} mlUtility={ml} explanation={explanation} />}

        {/* Model recommendation + override */}
        {rec && rec.model_key && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Baseline model</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-sm text-muted-foreground">{rec.reason}</p>
              {/* Recommended model + optional alternatives, always visible but compact */}
              <div className="grid gap-2 sm:max-w-sm">
                <Label htmlFor="model-override">Baseline model</Label>
                <Select value={modelKey} onValueChange={rerunWithModel}>
                  <SelectTrigger id="model-override">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__recommended__">
                      Recommended: {rec.label}
                    </SelectItem>
                    {rec.alternatives.map((a) => (
                      <SelectItem key={a.key} value={a.key}>
                        {a.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {rec.alternatives.map((a) => (
                  <p key={a.key} className="text-xs text-muted-foreground">
                    <span className="font-medium">{a.label}:</span> {a.strength}
                  </p>
                ))}
                <p className="text-xs text-muted-foreground">
                  Recommendations are sensible starting points, not universally best models — try other baselines if
                  you need to.
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        <Alert variant="warning">
          <CircleAlert className="h-4 w-4" />
          <AlertTitle>Read the results honestly</AlertTitle>
          <AlertDescription className="text-xs leading-relaxed">
            {INTERPRETATION_NOTES.similarityVsUtility} {INTERPRETATION_NOTES.notPrivate}
          </AlertDescription>
        </Alert>
      </div>

      <ActionBar>
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <Label htmlFor="project-name" className="shrink-0 text-sm text-muted-foreground">Project name</Label>
          <Input
            id="project-name"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            placeholder={`${config.filename.replace(/\.csv$/i, "")} — expanded`}
            className="max-w-sm"
          />
        </div>
        <Button variant="outline" onClick={() => router.push("/create/upload/preview")}>
          Back
        </Button>
        <Button variant="outline" onClick={() => config.upload_id && api.downloadSessionCsv(config.upload_id)}>
          Download synthetic CSV
        </Button>
        <Button onClick={saveAndExport} disabled={saving || !evaluation}>
          {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
          {savedId ? "Saved — generate PDF report" : "Save project & report"}
        </Button>
        {savedId && (
          <Button variant="secondary" onClick={() => router.push(`/projects/${savedId}`)}>
            Open project
          </Button>
        )}
      </ActionBar>
    </div>
  );
}
