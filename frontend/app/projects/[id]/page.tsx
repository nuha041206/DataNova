"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, CircleAlert, CircleCheck, FileDown, FileSpreadsheet, FileText, FileX2, Info, Loader2, Wand2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { EvaluationView } from "@/components/workflow/evaluation-view";
import { api } from "@/lib/api";
import type { ProjectDetail, ProjectFiles } from "@/lib/types";

export default function ProjectDetailPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [files, setFiles] = useState<ProjectFiles | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [generatingReport, setGeneratingReport] = useState(false);

  useEffect(() => {
    api
      .getProject(Number(params.id))
      .then(setProject)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load project"));
    api
      .projectFiles(Number(params.id))
      .then(setFiles)
      .catch(() => setFiles(null));
  }, [params.id]);

  if (error) {
    return (
      <div className="mx-auto max-w-3xl space-y-4 p-6">
        <Alert variant="destructive">
          <CircleAlert className="h-4 w-4" />
          <AlertTitle>Could not load project</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
        <Button asChild variant="outline">
          <Link href="/projects">Back to history</Link>
        </Button>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="mx-auto max-w-6xl space-y-4 p-6">
        <Skeleton className="h-24 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  const hasEvaluation = Boolean(project.evaluation && project.evaluation.column_results);

  return (
    <div className="flex min-h-screen flex-col">
      <div className="border-b bg-background px-6 py-5">
        <Button asChild variant="ghost" size="sm" className="-ml-2 mb-2">
          <Link href="/projects">
            <ArrowLeft className="h-4 w-4" /> History
          </Link>
        </Button>
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-2xl font-bold tracking-tight">{project.name}</h1>
          <Badge variant={project.mode === "upload" ? "default" : "secondary"}>
            {project.mode === "upload" ? "expanded dataset" : "from requirements"}
          </Badge>
          {project.has_report && <Badge variant="success">PDF ready</Badge>}
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          Created {new Date(project.created_at).toLocaleString()}
          {project.dataset_filename ? ` · ${project.dataset_filename}` : ""}
          {project.target ? ` · target: ${project.target}` : ""}
          {project.task ? ` · ${project.task}` : ""}
          {project.synthesizer ? ` · ${project.synthesizer}` : ""}
        </p>
      </div>

      <div className="mx-auto w-full max-w-6xl flex-1 space-y-6 p-6">
        {/* Files on disk — existence-aware so downloads never surprise-fail */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Files</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex items-center gap-2">
              {files?.synthetic_csv.exists ? (
                <CircleCheck className="h-4 w-4 text-brand-red" />
              ) : (
                <FileX2 className="h-4 w-4 text-muted-foreground" />
              )}
              <FileSpreadsheet className="h-4 w-4 text-muted-foreground" />
              <span>
                Synthetic CSV — {" "}
                {files?.synthetic_csv.exists
                  ? files.synthetic_csv.file
                  : project.rows_synthetic
                    ? "missing on disk (created before local file persistence)"
                    : "not saved with this project"}
              </span>
            </div>
            <div className="flex items-center gap-2">
              {files?.report_pdf.exists ? (
                <CircleCheck className="h-4 w-4 text-brand-red" />
              ) : (
                <FileX2 className="h-4 w-4 text-muted-foreground" />
              )}
              <FileText className="h-4 w-4 text-muted-foreground" />
              <span>
                PDF report — {" "}
                {files?.report_pdf.exists
                  ? `generated ${new Date(files.report_pdf.generated_at ?? "").toLocaleString()}`
                  : "not generated yet"}
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Generation summary */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Generation summary</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1.5 text-sm text-muted-foreground">
            {project.synthesizer && (
              <p>
                <b className="text-foreground">Synthesizer:</b> {project.synthesizer}
              </p>
            )}
            {project.generation_reason && <p>{project.generation_reason}</p>}
            <p>
              <b className="text-foreground">Rows:</b>{" "}
              {project.rows_synthetic?.toLocaleString() ?? "—"} generated
              {project.rows_requested ? ` (requested ${project.rows_requested.toLocaleString()})` : ""}
              {project.fit_seconds != null ? ` · fit ${project.fit_seconds}s` : ""}
              {project.sample_seconds != null ? ` · sample ${project.sample_seconds}s` : ""}
            </p>
          </CardContent>
        </Card>

        {project.requirement_spec && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <Wand2 className="h-4 w-4 text-primary" /> Original requirement
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <p className="rounded-md bg-muted px-3 py-2 italic text-muted-foreground">
                “{project.requirement_spec.raw_text}”
              </p>
              <div className="flex flex-wrap gap-1.5">
                <Badge variant="outline">{project.requirement_spec.rows} rows</Badge>
                <Badge variant="outline">{project.requirement_spec.task}</Badge>
                {project.requirement_spec.target && <Badge variant="outline">target: {project.requirement_spec.target}</Badge>}
                {project.requirement_spec.features.map((f) => (
                  <Badge key={f.name} variant="secondary">{f.name}</Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {project.analysis && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Dataset health (as recorded)</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex flex-wrap gap-1.5">
                <Badge variant="outline">{project.analysis.rows.toLocaleString()} rows</Badge>
                <Badge variant="outline">{project.analysis.columns} columns</Badge>
                <Badge variant="outline">{project.analysis.total_missing_cells} missing cells</Badge>
                <Badge variant="outline">{project.analysis.duplicate_rows} duplicate rows</Badge>
                {project.analysis.id_like_columns.length > 0 && (
                  <Badge variant="outline">ID-like: {project.analysis.id_like_columns.join(", ")}</Badge>
                )}
              </div>
              {project.analysis.warnings.length > 0 && (
                <ul className="list-inside list-disc text-xs text-muted-foreground">
                  {project.analysis.warnings.slice(0, 5).map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              )}
            </CardContent>
          </Card>
        )}

        {project.detection && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Target detection (as recorded)</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              {project.detection.suggested_target ?? "—"} · {project.detection.confidence} confidence ·{" "}
              {project.detection.reason}
              {project.target_overridden ? " (user overrode the suggestion)" : ""}
            </CardContent>
          </Card>
        )}

        {hasEvaluation && project.evaluation && (
          <EvaluationView
            evaluation={project.evaluation}
            mlUtility={project.ml_utility}
            explanation={project.explanation}
          />
        )}

        {!hasEvaluation && project.mode === "requirements" && (
          <Alert variant="info">
            <Info className="h-4 w-4" />
            <AlertTitle>Requirement-based project</AlertTitle>
            <AlertDescription className="text-xs">
              This dataset was generated from requirements without a real source dataset, so no real-vs-synthetic
              comparison exists. DataNova does not fabricate similarity scores.
            </AlertDescription>
          </Alert>
        )}

        {project.warnings.length > 0 && (
          <Alert variant="warning">
            <CircleAlert className="h-4 w-4" />
            <AlertTitle>Recorded warnings</AlertTitle>
            <AlertDescription>
              <ul className="list-inside list-disc">
                {project.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </AlertDescription>
          </Alert>
        )}
      </div>

      <div className="sticky bottom-0 border-t bg-background/95 px-6 py-3 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-end gap-3">
          <Button variant="outline" onClick={() => api.downloadProjectCsv(project.id)} disabled={!project.rows_synthetic}>
            <FileDown className="h-4 w-4" /> Synthetic CSV
          </Button>
          {project.has_report ? (
            <Button onClick={() => api.downloadReport(project.id)}>
              <FileText className="h-4 w-4" /> Download PDF report
            </Button>
          ) : (
            <Button
              onClick={async () => {
                setGeneratingReport(true);
                try {
                  await api.generateReport(project.id);
                  router.refresh();
                  setProject({ ...project, has_report: true });
                } finally {
                  setGeneratingReport(false);
                }
              }}
              disabled={generatingReport}
            >
              {generatingReport ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileText className="h-4 w-4" />}
              Generate PDF report
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
