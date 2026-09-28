"use client";

/**
 * Shared "Realistic Data Quality Issues" selector used by Mode 1 (after a
 * successful generation) and Mode 2 (in the requirements editor).
 * Checkbox + percentage per issue, plain-language explanations, and honest
 * reporting of ACTUAL applied percentages after generation.
 */
import { useState } from "react";
import { CircleAlert, CircleCheck, FlaskConical, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { api } from "@/lib/api";
import type { MessyConfig, SessionMessyResult } from "@/lib/types";

export const MESSY_ISSUES: { key: keyof MessyConfig; label: string; explanation: string }[] = [
  {
    key: "missing_values",
    label: "Missing values",
    explanation: "Randomly removes values from selected columns to simulate incomplete real-world data.",
  },
  {
    key: "duplicate_rows",
    label: "Duplicate records",
    explanation: "Repeats some records to simulate accidental duplicate entries.",
  },
  {
    key: "whitespace",
    label: "Whitespace errors",
    explanation: "Adds stray spaces or tabs around text values, like data copied from different sources.",
  },
  {
    key: "capitalization",
    label: "Inconsistent capitalization",
    explanation: "Introduces variations such as mumbai, MUMBAI or MuMbAi in text columns.",
  },
  {
    key: "format_inconsistency",
    label: "Formatting inconsistencies",
    explanation: "Realistic variants for phones (+91 …, 0987…), dates (12/01/2026, 12-Jan-2026) and emails.",
  },
  {
    key: "invalid_values",
    label: "Invalid / inconsistent values",
    explanation: "Introduces controlled values that do not follow the normal pattern (ages like &apos;twenty five&apos; or -2).",
  },
  {
    key: "outliers",
    label: "Outliers",
    explanation: "Adds a small number of unusually high or low values in numeric columns.",
  },
];

export function MessyOptionsSession({ uploadId, rows }: { uploadId: string; rows: number }) {
  const [enabled, setEnabled] = useState<Record<string, boolean>>({});
  const [pcts, setPcts] = useState<MessyConfig>({
    missing_values: 5, duplicate_rows: 3, whitespace: 0,
    capitalization: 0, format_inconsistency: 2, invalid_values: 1, outliers: 0,
  });
  const [result, setResult] = useState<SessionMessyResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const anySelected = Object.entries(enabled).some(([, on]) => on);
  const cells = rows * 8; // rough scope for the estimate text

  async function apply() {
    setBusy(true);
    setError(null);
    try {
      const cfg: MessyConfig = {
        missing_values: enabled.missing_values ? pcts.missing_values : 0,
        duplicate_rows: enabled.duplicate_rows ? pcts.duplicate_rows : 0,
        whitespace: enabled.whitespace ? pcts.whitespace : 0,
        capitalization: enabled.capitalization ? pcts.capitalization : 0,
        format_inconsistency: enabled.format_inconsistency ? pcts.format_inconsistency : 0,
        invalid_values: enabled.invalid_values ? pcts.invalid_values : 0,
        outliers: enabled.outliers ? pcts.outliers : 0,
      };
      setResult(await api.sessionMessy(uploadId, cfg));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not apply data-quality issues");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <FlaskConical className="h-4 w-4 text-brand-red" /> Realistic Data Quality Issues
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <p className="text-muted-foreground">
          Do you want the generated dataset to contain realistic data-quality problems? The clean
          dataset stays available — issues are applied to a separate copy and the{" "}
          <b>actual percentages</b> are reported afterwards.
        </p>

        <div className="space-y-2">
          {MESSY_ISSUES.map(({ key, label, explanation }) => (
            <div key={key} className="rounded-lg border p-3">
              <div className="flex flex-wrap items-center gap-3">
                <label className="flex items-center gap-2 text-sm font-medium">
                  <input
                    type="checkbox"
                    checked={!!enabled[key]}
                    onChange={(e) => setEnabled((s) => ({ ...s, [key]: e.target.checked }))}
                  />
                  {label}
                </label>
                {enabled[key] && (
                  <span className="flex items-center gap-1.5">
                    <Input
                      type="number"
                      min={0}
                      max={100}
                      step={0.5}
                      value={pcts[key]}
                      onChange={(e) =>
                        setPcts((p) => ({ ...p, [key]: Math.max(0, Math.min(100, Number(e.target.value) || 0)) }))}
                      className="h-8 w-20"
                      aria-label={`${label} percentage`}
                    />
                    <span className="text-xs text-muted-foreground">%</span>
                  </span>
                )}
              </div>
              <p className="mt-1 text-xs text-muted-foreground">{explanation}</p>
            </div>
          ))}
        </div>

        {anySelected && rows > 0 && (
          <p className="rounded-md border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
            For {rows.toLocaleString()} rows: 5% missing values ≈ {(cells * 0.05).toLocaleString()} affected
            cells; 3% duplicates ≈ {Math.round(rows * 0.03).toLocaleString()} extra rows. Exact counts depend
            on your dataset&apos;s shape and are reported after applying.
          </p>
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
            <AlertTitle>Applied — actual results</AlertTitle>
            <AlertDescription>
              <ul className="list-inside list-disc">
                <li>Missing cells: {result.summary.counts.missing_values.toLocaleString()} (actual {result.actual_percentages.missing_values}% of all cells)</li>
                <li>Duplicate rows added: {result.summary.counts.duplicate_rows.toLocaleString()} (actual {result.actual_percentages.duplicate_rows}%) — rows now {result.summary.rows_after_duplicates.toLocaleString()}</li>
                <li>Whitespace issues: {result.summary.counts.whitespace.toLocaleString()} · capitalization: {result.summary.counts.capitalization.toLocaleString()} · format variants: {result.summary.counts.format_inconsistencies.toLocaleString()}</li>
                <li>Invalid values: {result.summary.counts.invalid_values.toLocaleString()} · outliers: {result.summary.counts.outliers.toLocaleString()}</li>
              </ul>
            </AlertDescription>
          </Alert>
        )}

        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={apply} disabled={busy || !anySelected}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <FlaskConical className="h-4 w-4" />}
            Apply to a separate copy
          </Button>
          {result && (
            <Button variant="outline" onClick={() => api.downloadSessionMessyCsv(uploadId)}>
              Download messy CSV
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
