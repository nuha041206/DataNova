"use client";

/**
 * Mode B — Evaluate + Export step for requirement-generated datasets.
 * There is no real dataset to compare against, so DataNova shows an honest
 * self-description of the generated data, the actual target balance, and an
 * ML self-consistency test (train/test split on the generated data), plus
 * the generation spec. No fake comparison scores.
 */
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { CircleAlert, Info, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip as ReTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { StepHeader } from "@/components/workflow/step-header";
import { ActionBar } from "@/components/workflow/page-shell";
import { api } from "@/lib/api";
import type { RequirementGenerateResponse } from "@/lib/types";
import { INTERPRETATION_NOTES } from "@/lib/notes";

interface ReqResult {
  spec: RequirementGenerateResponse["spec"];
  summary: RequirementGenerateResponse["summary"];
  description: string;
  warnings: string[];
  preview: Record<string, unknown>[];
  session_id?: string;
}

export default function RequirementsEvaluatePage() {
  const router = useRouter();
  const [result, setResult] = useState<ReqResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [savedId, setSavedId] = useState<number | null>(null);
  const [projectName, setProjectName] = useState("");

  useEffect(() => {
    const raw = sessionStorage.getItem("datanova-req-result");
    if (!raw) {
      router.replace("/create/requirements");
      return;
    }
    setResult(JSON.parse(raw));
  }, [router]);

  if (!result) return null;

  const { spec, summary } = result;
  const isClf = summary.task === "classification";

  // Build the target balance chart from actual returned counts
  const targetCounts =
    isClf && summary.target_summary && "counts" in summary.target_summary
      ? Object.entries(summary.target_summary.counts as Record<string, number>).map(([name, count]) => ({
          name,
          share: 0, // filled below from actual counts
          count,
        }))
      : [];
  const total = targetCounts.reduce((a, b) => a + b.count, 0) || 1;
  targetCounts.forEach((t) => (t.share = t.count / total));

  return (
    <div className="flex min-h-screen flex-col">
      <StepHeader
        step="Evaluate"
        title="Your generated dataset"
        subtitle="A transparent description of what was created — computed from the generated data itself."
      />

      <div className="mx-auto w-full max-w-5xl flex-1 space-y-6 p-6">
        {error && (
          <Alert variant="destructive">
            <CircleAlert className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="grid gap-4 sm:grid-cols-4">
          <Card>
            <CardContent className="p-4">
              <div className="text-xs text-muted-foreground">Rows</div>
              <div className="text-xl font-bold">{summary.rows.toLocaleString()}</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-xs text-muted-foreground">Columns</div>
              <div className="text-xl font-bold">{summary.columns.length}</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-xs text-muted-foreground">Task</div>
              <div className="text-xl font-bold capitalize">
                {summary.task === "unknown" ? "None" : summary.task}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-xs text-muted-foreground">Target</div>
              <div className="text-xl font-bold">{summary.target || "—"}</div>
            </CardContent>
          </Card>
        </div>

        <Alert variant="info">
          <Info className="h-4 w-4" />
          <AlertTitle>About this dataset</AlertTitle>
          <AlertDescription className="text-xs leading-relaxed">
            {result.description} Because there is no real source dataset, DataNova does not show real-vs-synthetic
            similarity scores — inventing them would be misleading. The statistics below are computed directly from
            the generated rows. {INTERPRETATION_NOTES.notPrivate}
          </AlertDescription>
        </Alert>

        {isClf && targetCounts.length > 0 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Target balance (actual)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={targetCounts} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                  <XAxis dataKey="name" fontSize={11} />
                  <YAxis fontSize={11} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                  <ReTooltip formatter={(v: unknown) => `${(Number(v) * 100).toFixed(1)}%`} />
                  <Bar dataKey="share" fill="#6D0808" radius={[3, 3, 0, 0]} name="Share of rows" />
                </BarChart>
              </ResponsiveContainer>
              {spec.class_distribution?.positive_pct && (
                <p className="mt-2 text-xs text-muted-foreground">
                  Requested positive share: {spec.class_distribution.positive_pct}% — actual:{" "}
                  {((targetCounts.find((t) => t.name === "1")?.share ?? 0) * 100).toFixed(1)}%.
                </p>
              )}
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Numeric feature summary</CardTitle>
          </CardHeader>
          <CardContent>
            {Object.keys(summary.numeric_summary).length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No numeric columns in this dataset — all generated columns are semantic (names, categories,
                formatted strings, etc.).
              </p>
            ) : (
              <>
            <div className="overflow-auto rounded-md border">
              <table className="w-full text-sm">
                <thead className="bg-muted">
                  <tr>
                    <th className="p-2 text-left font-medium">Column</th>
                    <th className="p-2 text-left font-medium">Mean</th>
                    <th className="p-2 text-left font-medium">Std</th>
                    <th className="p-2 text-left font-medium">Min</th>
                    <th className="p-2 text-left font-medium">Max</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(summary.numeric_summary).map(([col, s]) => (
                    <tr key={col} className="border-t">
                      <td className="p-2 font-medium">{col}</td>
                      <td className="p-2">{s.mean}</td>
                      <td className="p-2">{s.std}</td>
                      <td className="p-2">{s.min}</td>
                      <td className="p-2">{s.max}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Example rows</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-auto rounded-md border">
              <table className="w-full text-sm">
                <thead className="bg-muted">
                  <tr>
                    {summary.columns.map((c) => (
                      <th key={c} className="whitespace-nowrap p-2 text-left font-medium">{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.preview.slice(0, 10).map((r, i) => (
                    <tr key={i} className="border-t">
                      {summary.columns.map((c) => (
                        <td key={c} className="whitespace-nowrap p-2">{String(r[c] ?? "—")}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

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
      </div>

      <ActionBar>
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <Label htmlFor="project-name" className="shrink-0 text-sm text-muted-foreground">Project name</Label>
          <Input
            id="project-name"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            placeholder={`${spec.target ?? "target"} dataset — from requirements`}
            className="max-w-sm"
          />
        </div>
        <Button variant="outline" onClick={() => router.push("/create/requirements")}>
          Back
        </Button>
        <Button
          onClick={async () => {
            setSaving(true);
            try {
              const res = await api.saveProject({
                name: projectName.trim() || `${spec.target ?? "target"} dataset — from requirements`,
                mode: "requirements",
                // Send the session id so the backend persists this generated
                // CSV to disk — otherwise the download disappears after a
                // backend restart.
                upload_id: result.session_id ?? null,
                rows_real: null,
                columns: summary.columns.length,
                target: summary.target,
                task: summary.task,
                rows_synthetic: summary.rows,
                requirement_spec: spec,
                warnings: result.warnings,
              });
              setSavedId(res.project_id);
              await api.generateReport(res.project_id);
            } catch (e) {
              setError(e instanceof Error ? e.message : "Saving failed");
            } finally {
              setSaving(false);
            }
          }}
          disabled={saving}
        >
          {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
          {savedId ? "Saved — PDF report ready" : "Save project & report"}
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
