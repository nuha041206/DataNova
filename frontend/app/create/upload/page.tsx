"use client";

/**
 * Mode A — Configure step.
 * Upload CSV -> health check -> target detection (override) -> generation
 * settings -> continue to preview.
 */
import { useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  CircleAlert,
  Loader2,
  RotateCcw,
  UploadCloud,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { StepHeader } from "@/components/workflow/step-header";
import { ActionBar } from "@/components/workflow/page-shell";
import { HealthCheckCards } from "@/components/workflow/health-check-cards";
import { DetectionCard } from "@/components/workflow/detection-card";
import { api } from "@/lib/api";
import { SEMANTIC_LABELS, type Detection, type SchemaInfo, type UploadResponse } from "@/lib/types";
import { INTERPRETATION_NOTES } from "@/lib/notes";

export default function UploadConfigurePage() {
  const router = useRouter();
  const fileInput = useRef<HTMLInputElement>(null);

  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [detection, setDetection] = useState<Detection | null>(null);
  const [schema, setSchema] = useState<Record<string, SchemaInfo>>({});
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // generation settings
  const [nRows, setNRows] = useState("1000");
  const [excludeIds, setExcludeIds] = useState(true);
  const [synthOverride, setSynthOverride] = useState<string>("");
  const [epochs, setEpochs] = useState("120");

  const excludedColumns = useMemo(() => {
    if (!upload || !excludeIds) return [];
    return upload.analysis.id_like_columns;
  }, [upload, excludeIds]);

  async function onFilePicked(file: File) {
    setBusy(true);
    setError(null);
    try {
      const res = await api.uploadDataset(file);
      setUpload(res);
      setDetection(res.detection);
      setSchema(res.schema ?? {});
      // Sensible default: generate as many rows as the real dataset has
      setNRows(String(Math.min(Math.max(res.analysis.rows, 100), 5000)));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  async function overrideTarget(target: string | null) {
    if (!upload) return;
    setBusy(true);
    setError(null);
    try {
      const det = await api.redetect(upload.upload_id, target);
      setDetection(det);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Detection failed");
    } finally {
      setBusy(false);
    }
  }

  function goToPreview() {
    if (!upload || !detection) return;
    const target = detection.suggested_target;
    if (!target) {
      setError("Select a target column first, or choose 'No target' in the override.");
      return;
    }
    router.push(`/create/upload/preview`, {
      // pass config through sessionStorage: small, local, no server state
    });
    sessionStorage.setItem(
      "datanova-config",
      JSON.stringify({
        upload_id: upload.upload_id,
        filename: upload.filename,
        n_rows: parseInt(nRows, 10) || 1000,
        target: detection.suggested_target,
        task: detection.task,
        target_overridden: detection.confidence === "user",
        exclude_columns: excludedColumns,
        synthesizer: synthOverride || null,
        advanced: synthOverride === "CTGAN" || synthOverride === "TVAE" ? { epochs: parseInt(epochs, 10) || 120 } : null,
        analysis: upload.analysis,
        detection,
      })
    );
  }

  return (
    <div className="flex min-h-screen flex-col">
      <StepHeader
        step="Configure"
        title="Expand an existing dataset"
        subtitle="Upload a CSV, review its health, confirm the target, then set how many rows to generate."
      />

      <div className="mx-auto w-full max-w-6xl flex-1 space-y-6 p-6">
        {/* Upload zone */}
        {!upload && (
          <Card
            className="border-2 border-dashed"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files?.[0];
              if (f) onFilePicked(f);
            }}
          >
            <CardContent className="flex flex-col items-center gap-3 py-14 text-center">
              {busy ? (
                <>
                  <Loader2 className="h-10 w-10 animate-spin text-primary" />
                  <p className="text-sm text-muted-foreground">Uploading &amp; analyzing…</p>
                </>
              ) : (
                <>
                  <UploadCloud className="h-12 w-12 text-muted-foreground/50" />
                  <div>
                    <p className="font-medium">Drag &amp; drop your CSV here</p>
                    <p className="text-sm text-muted-foreground">or browse — up to 50 MB, .csv files</p>
                  </div>
                  <Button onClick={() => fileInput.current?.click()} disabled={busy}>
                    Choose a CSV File
                  </Button>
                  <input
                    ref={fileInput}
                    type="file"
                    accept=".csv,text/csv"
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) onFilePicked(f);
                    }}
                  />
                </>
              )}
            </CardContent>
          </Card>
        )}

        {error && (
          <Alert variant="destructive">
            <CircleAlert className="h-4 w-4" />
            <AlertTitle>Something went wrong</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Analysis + detection */}
        {upload && detection && (
          <>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h2 className="text-lg font-semibold">{upload.filename}</h2>
                <p className="text-xs text-muted-foreground">
                  {upload.analysis.rows.toLocaleString()} rows · {upload.analysis.columns} columns
                </p>
              </div>
              <Button variant="outline" size="sm" onClick={() => { setUpload(null); setDetection(null); }}>
                <RotateCcw className="h-4 w-4" /> Replace File
              </Button>
            </div>

            <HealthCheckCards analysis={upload.analysis} />

            {/* Detected schema: technical + semantic types per column */}
            {Object.keys(schema).length > 0 && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Detected Schema — Technical &amp; Semantic Types</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto rounded-md border">
                    <table className="w-full text-xs">
                      <thead className="bg-muted text-left">
                        <tr>
                          <th className="p-2 font-medium">Column</th>
                          <th className="p-2 font-medium">Technical type</th>
                          <th className="p-2 font-medium">Semantic type</th>
                          <th className="p-2 font-medium">Why</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(schema).map(([col, info]) => (
                          <tr key={col} className="border-t">
                            <td className="p-2 font-medium">{col}</td>
                            <td className="p-2">{info.technical_type}</td>
                            <td className="p-2">{SEMANTIC_LABELS[info.semantic_type] ?? info.semantic_type}</td>
                            <td className="max-w-[280px] truncate p-2 text-muted-foreground" title={info.detection_reason}>
                              {info.detection_reason}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    Columns with format-critical semantics (names, phones, emails, pincodes, addresses) are
                    regenerated with dedicated format-aware generators after SDV sampling, so every value is valid.
                  </p>
                </CardContent>
              </Card>
            )}

            <DetectionCard detection={detection} columns={upload.analysis.column_profiles.map((c) => c.name)} onTargetChange={overrideTarget} />

            {/* Generation Settings */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="font-display text-base">Generation Settings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-1.5">
                    <Label htmlFor="nrows">Additional rows to generate</Label>
                    <Input
                      id="nrows"
                      type="number"
                      min={1}
                      max={200000}
                      value={nRows}
                      onChange={(e) => setNRows(e.target.value)}
                    />
                    <p className="text-xs text-muted-foreground">
                      How many synthetic rows to add — up to 10,000+. Larger counts take longer; duplicates are
                      removed and regenerated automatically.
                    </p>
                  </div>
                  <label className="flex items-start gap-2 rounded-md border p-3">
                    <input
                      type="checkbox"
                      className="mt-0.5"
                      checked={excludeIds}
                      onChange={(e) => setExcludeIds(e.target.checked)}
                    />
                    <span>
                      <span className="text-sm font-medium">Exclude ID-like columns</span>
                      <span className="block text-xs text-muted-foreground">
                        {excludedColumns.length
                          ? excludedColumns.join(", ")
                          : "No ID-like columns detected — nothing will be excluded."}
                      </span>
                    </span>
                  </label>
                </div>

                {/* Technical options — collapsed by default so the main flow stays simple */}
                <details className="rounded-md border bg-muted/40">
                  <summary className="cursor-pointer select-none px-4 py-2.5 text-sm font-medium">
                    Synthesizer options (optional)
                  </summary>
                  <div className="grid gap-4 px-4 pb-4 pt-1 sm:grid-cols-2">
                    <div className="space-y-1.5">
                      <Label htmlFor="synth">Synthesizer</Label>
                      <select
                        id="synth"
                        className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm"
                        value={synthOverride}
                        onChange={(e) => setSynthOverride(e.target.value)}
                      >
                        <option value="">Auto (recommended)</option>
                        <option value="GaussianCopula">GaussianCopula — fast, transparent</option>
                        <option value="CTGAN">CTGAN — deep learning, slower</option>
                        <option value="TVAE">TVAE — autoencoder, slower</option>
                      </select>
                      <p className="text-xs text-muted-foreground">
                        Leave on Auto and DataNova picks a suitable synthesizer based on your dataset size and types.
                      </p>
                    </div>
                    <div className="space-y-1.5">
                      <Label htmlFor="epochs">CTGAN/TVAE epochs</Label>
                      <Input id="epochs" type="number" min={20} max={1000} value={epochs} onChange={(e) => setEpochs(e.target.value)} />
                      <p className="text-xs text-muted-foreground">
                        Training epochs for neural synthesizers (default 120). More epochs = slower, possibly better.
                      </p>
                    </div>
                  </div>
                </details>

                <p className="text-xs text-muted-foreground">{INTERPRETATION_NOTES.notPrivate}</p>
              </CardContent>
            </Card>
          </>
        )}
      </div>

      <ActionBar>
        <Button variant="outline" onClick={() => router.push("/create")} disabled={!upload}>
          Back
        </Button>
        <Button onClick={goToPreview} disabled={!upload || !detection || busy}>
          Continue to Preview
        </Button>
      </ActionBar>
    </div>
  );
}
