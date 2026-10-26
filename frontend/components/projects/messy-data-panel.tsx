"use client";

/**
 * "Make Data Messy" — user-controlled realistic messiness for the stored
 * synthetic dataset. The original clean dataset is never modified: the messy
 * result is written to a separate *_messy.csv file.
 *
 * All percentages are set by the user; nothing is corrupted automatically.
 * Estimates are shown BEFORE generating; actual counts are shown after.
 */
import { useCallback, useEffect, useState } from "react";
import { CircleAlert, CircleCheck, FlaskConical, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";
import type { MessyConfig, MessyPreviewResponse, MessyResult } from "@/lib/types";

const FIELDS: { key: keyof MessyConfig; label: string; hint: string }[] = [
  { key: "missing_values", label: "Missing values", hint: "Cells become NaN or empty strings (identifiers are protected)." },
  { key: "duplicate_rows", label: "Duplicate rows", hint: "Appends real duplicate records; rows grow, nothing is overwritten." },
  { key: "whitespace", label: "Whitespace errors", hint: "Leading/trailing/tab characters in text columns." },
  { key: "capitalization", label: "Inconsistent capitalization", hint: "mumbai / MUMBAI / MuMbAi style variants." },
  { key: "format_inconsistency", label: "Formatting inconsistencies", hint: "Phone/date/email format variants, per semantic type." },
  { key: "invalid_values", label: "Invalid / noisy values", hint: "Ages like \"twenty five\" or -2; applied sparingly." },
  { key: "outliers", label: "Outliers", hint: "Controlled extreme values in numeric columns." },
];

const DEFAULTS: MessyConfig = {
  missing_values: 5,
  duplicate_rows: 3,
  whitespace: 2,
  capitalization: 2,
  format_inconsistency: 2,
  invalid_values: 1,
  outliers: 0,
};

export function MessyDataPanel({ projectId }: { projectId: number }) {
  const [cfg, setCfg] = useState<MessyConfig>(DEFAULTS);
  const [preview, setPreview] = useState<MessyPreviewResponse | null>(null);
  const [result, setResult] = useState<MessyResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refreshPreview = useCallback((c: MessyConfig) => {
    api.messyPreview(projectId, c).then(setPreview).catch(() => setPreview(null));
  }, [projectId]);

  useEffect(() => {
    refreshPreview(cfg);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  function update(key: keyof MessyConfig, raw: string) {
    const v = Math.max(0, Math.min(100, Number(raw) || 0));
    const next = { ...cfg, [key]: v };
    setCfg(next);
    refreshPreview(next);
  }

  async function generate() {
    setBusy(true);
    setError(null);
    try {
      const res = await api.messyGenerate(projectId, cfg);
      setResult(res);
      refreshPreview(cfg);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Messy data generation failed");
    } finally {
      setBusy(false);
    }
  }

  const est = preview?.estimates;
  const ds = preview?.dataset;
  const anyRequested = Object.values(cfg).some((v) => v > 0);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <FlaskConical className="h-4 w-4 text-brand-red" /> Make Data Messy
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <p className="text-muted-foreground">
          Simulates realistic uncleaned data for cleaning / preprocessing practice. You control every
          percentage; the <b>original clean dataset is never modified</b> — the messy copy is saved as a
          separate file.
        </p>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {FIELDS.map(({ key, label, hint }) => (
            <div key={key} className="rounded-lg border p-3">
              <Label htmlFor={`messy-${key}`} className="text-xs font-medium">{label}</Label>
              <div className="mt-1.5 flex items-center gap-1.5">
                <Input
                  id={`messy-${key}`}
                  type="number"
                  min={0}
                  max={100}
                  step={0.5}
                  value={cfg[key]}
                  onChange={(e) => update(key, e.target.value)}
                  className="h-8 w-20"
                  title={hint}
                />
                <span className="text-xs text-muted-foreground">%</span>
              </div>
              <p className="mt-1 text-[11px] leading-snug text-muted-foreground">{hint}</p>
            </div>
          ))}
        </div>

        {ds && est && (
          <div className="rounded-lg border bg-muted/40 p-3 text-xs">
            <div className="font-medium">Current dataset</div>
            <div className="mt-1 text-muted-foreground">
              Rows: {ds.rows.toLocaleString()} · Columns: {ds.columns} · Total cells: {ds.total_cells.toLocaleString()}
            </div>
            <div className="mt-2 font-medium">Estimated impact at these settings</div>
            <ul className="mt-1 list-inside list-disc text-muted-foreground">
              <li>Missing-value cells: ~{est.missing_cells.toLocaleString()}</li>
              <li>Duplicate rows added: ~{est.duplicate_rows.toLocaleString()}</li>
              <li>Whitespace cells: ~{est.whitespace_cells.toLocaleString()}</li>
              <li>Capitalization cells: ~{est.capitalization_cells.toLocaleString()}</li>
              <li>Format-variant cells: ~{est.format_cells.toLocaleString()}</li>
              <li>Invalid/noisy cells: ~{est.invalid_cells.toLocaleString()}</li>
              <li>Outlier cells: ~{est.outlier_cells.toLocaleString()}</li>
            </ul>
            <p className="mt-1.5 text-muted-foreground">
              Semantic columns detected — text-like: {est.text_columns.length} · numeric: {est.numeric_columns.length} ·
              format-sensitive (phone/email/date): {est.formatable_columns.length}
            </p>
          </div>
        )}

        {error && (
          <Alert variant="destructive">
            <CircleAlert className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {result && (
          <Alert variant="info">
            <CircleCheck className="h-4 w-4" />
            <AlertTitle>Messy Data Summary — actual changes</AlertTitle>
            <AlertDescription>
              <ul className="list-inside list-disc">
                <li>Missing cells introduced: {result.summary.counts.missing_values.toLocaleString()} (target ~{result.summary.missing_cells_target.toLocaleString()})</li>
                <li>Duplicate rows added: {result.summary.counts.duplicate_rows.toLocaleString()} — rows now {result.summary.rows_after_duplicates.toLocaleString()}</li>
                <li>Whitespace inconsistencies: {result.summary.counts.whitespace.toLocaleString()}</li>
                <li>Capitalization inconsistencies: {result.summary.counts.capitalization.toLocaleString()}</li>
                <li>Formatting inconsistencies: {result.summary.counts.format_inconsistencies.toLocaleString()}</li>
                <li>Invalid/noisy values: {result.summary.counts.invalid_values.toLocaleString()}</li>
                <li>Outliers: {result.summary.counts.outliers.toLocaleString()}</li>
              </ul>
              <p className="mt-1.5">
                Saved as <Badge variant="secondary">{result.file}</Badge> — the clean dataset is unchanged.
              </p>
            </AlertDescription>
          </Alert>
        )}

        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={generate} disabled={busy || !anyRequested}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <FlaskConical className="h-4 w-4" />}
            Generate Messy Dataset
          </Button>
          {result && (
            <Button variant="outline" onClick={() => api.downloadMessyCsv(projectId)}>
              Download messy CSV
            </Button>
          )}
        </div>
        {!anyRequested && (
          <p className="text-xs text-muted-foreground">Set at least one percentage above 0 to enable generation.</p>
        )}
      </CardContent>
    </Card>
  );
}
