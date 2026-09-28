"use client";

/**
 * Settings — real application settings UI.
 * Status rows show facts (read-only); editable controls only where a setting
 * actually exists. Technical filesystem paths live in a collapsed section.
 */
import { useEffect, useState } from "react";
import {
  Activity,
  ChevronDown,
  CircleCheck,
  CircleX,
  Database,
  Info,
  RotateCcw,
  Server,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { API_BASE, api } from "@/lib/api";
import { useBackendStatus } from "@/lib/backend-status";

function StatusRow({
  label,
  value,
  ok,
}: {
  label: string;
  value: React.ReactNode;
  ok?: boolean;
}) {
  return (
    <div className="flex items-center justify-between rounded-md border px-3 py-2.5">
      <span className="text-sm">{label}</span>
      <span className="flex items-center gap-1.5 text-sm font-medium">
        {ok === true && <CircleCheck className="h-4 w-4 text-brand-red" />}
        {ok === false && <CircleX className="h-4 w-4 text-red-600" />}
        {value}
      </span>
    </div>
  );
}

export default function SettingsPage() {
  const backend = useBackendStatus();
  const [apiInfo, setApiInfo] = useState<{ app?: string; version?: string } | null>(null);
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    if (backend !== "online") return;
    api.health().then(setApiInfo).catch(() => setApiInfo(null));
  }, [backend]);

  function clearLocalPreferences() {
    // Only browser-side preferences live here; project data is never touched.
    try {
      window.sessionStorage.removeItem("datanova-config");
      window.sessionStorage.removeItem("datanova-generation");
      window.sessionStorage.removeItem("datanova-reqspec");
      window.sessionStorage.removeItem("datanova-reqgen");
    } catch {
      /* storage unavailable */
    }
    setConfirming(false);
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Preferences and connection status for DataNova on this computer.
        </p>
      </div>

      {/* ------------------------------------------------ GENERAL */}
      <Card>
        <CardHeader>
          <CardTitle>General</CardTitle>
          <CardDescription>Interface behaviour. Preferences apply to this browser only.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          <StatusRow
            label="Default workflow"
            value="Configure → Generate → Preview → Evaluate → Export"
          />
          <StatusRow label="Row limit per generation" value="Up to 200,000 rows" />
          <StatusRow label="CSV upload limit" value="50 MB" />
          <StatusRow label="Interface language" value="English" />
        </CardContent>
      </Card>

      {/* ------------------------------------------------ DATA */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-4 w-4 text-brand-red" /> Data &amp; storage
          </CardTitle>
          <CardDescription>
            DataNova runs locally. Project history and generated files stay on this computer — no account is used.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          <StatusRow label="Storage" value="Local (this computer)" ok />
          <StatusRow label="Account" value="Not required" />
          <StatusRow label="Project history" value="Stored in a local database" />
          <p className="px-1 pt-1 text-xs text-muted-foreground">
            Removing DataNova&apos;s local data folder will also remove your project history. Uploaded datasets and
            generated CSVs stay inside that folder.
          </p>

          {/* Technical details — collapsed so normal users never see file paths */}
          <details className="mt-2 rounded-md border bg-muted/40">
            <summary className="flex cursor-pointer select-none items-center gap-1.5 px-3 py-2 text-xs font-medium text-muted-foreground">
              <ChevronDown className="h-3.5 w-3.5" /> Technical details
            </summary>
            <div className="space-y-2 px-3 pb-3 text-xs text-muted-foreground">
              <div className="flex items-center justify-between gap-4">
                <span>Uploaded &amp; generated CSVs</span>
                <code className="rounded bg-background px-1.5 py-0.5">datanova/backend/data/</code>
              </div>
              <div className="flex items-center justify-between gap-4">
                <span>PDF reports</span>
                <code className="rounded bg-background px-1.5 py-0.5">datanova/backend/data/reports/</code>
              </div>
              <div className="flex items-center justify-between gap-4">
                <span>Project database (SQLite)</span>
                <code className="rounded bg-background px-1.5 py-0.5">datanova/backend/data/db/datanova.db</code>
              </div>
            </div>
          </details>
        </CardContent>
      </Card>

      {/* ------------------------------------------------ CONNECTION */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Server className="h-4 w-4 text-brand-red" /> Connection
          </CardTitle>
          <CardDescription>The web app talks to a local API server to analyse and generate data.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          <StatusRow
            label="Backend status"
            value={backend === "checking" ? "Checking…" : backend === "online" ? "Connected" : "Not reachable"}
            ok={backend === "online" ? true : backend === "offline" ? false : undefined}
          />
          <StatusRow
            label="API address"
            value={<code className="rounded bg-muted px-1.5 py-0.5 text-xs">{API_BASE}</code>}
          />
          <StatusRow
            label="Interactive API docs"
            value={
              backend === "online" ? (
                <a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer" className="text-brand-red underline">
                  Open /docs
                </a>
              ) : (
                "Available when backend is running"
              )
            }
          />
          {backend === "offline" && (
            <div className="flex items-start gap-2 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900">
              <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              Start the backend with <code className="mx-1 rounded bg-amber-100 px-1">uvicorn app.main:app --port 8000</code>{" "}
              inside <code className="rounded bg-amber-100 px-1">datanova/backend</code>.
            </div>
          )}
        </CardContent>
      </Card>

      {/* ------------------------------------------------ APPLICATION */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-brand-red" /> Application
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <StatusRow label="Application" value={apiInfo?.app ?? "DataNova"} />
          <StatusRow label="Version" value={apiInfo?.version ?? "1.0.0"} />
          <Separator className="my-2" />
          {confirming ? (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm text-muted-foreground">
                Clear in-progress workflow data from this browser? Saved projects are not affected.
              </span>
              <Button size="sm" variant="outline" onClick={() => setConfirming(false)}>
                Cancel
              </Button>
              <Button size="sm" variant="destructive" onClick={clearLocalPreferences}>
                Clear
              </Button>
            </div>
          ) : (
            <div>
              <Button variant="outline" onClick={() => setConfirming(true)}>
                <RotateCcw className="h-4 w-4" /> Clear in-progress workflow data
              </Button>
              <p className="mt-1.5 text-xs text-muted-foreground">
                Resets unfinished upload/generation sessions held by this browser. Your saved project history is kept.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Loading state for app info while backend answers */}
      {backend === "checking" && (
        <div className="flex justify-center py-2">
          <Skeleton className="h-4 w-40" />
        </div>
      )}
    </div>
  );
}
