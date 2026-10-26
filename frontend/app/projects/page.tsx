"use client";

/**
 * Project History — every saved workflow on this computer.
 * Local-first: no account exists, so wording never implies a cloud user.
 */
import { useEffect, useState } from "react";
import Link from "next/link";
import { CircleAlert, FileDown, FileSpreadsheet, HardDrive, History, Pencil, Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useToast } from "@/components/ui/toast";
import { api } from "@/lib/api";
import type { ProjectSummary } from "@/lib/types";
import { useBackendStatus } from "@/lib/backend-status";

export default function ProjectsPage() {
  const backend = useBackendStatus();
  const { toast } = useToast();
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<ProjectSummary | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [renaming, setRenaming] = useState<ProjectSummary | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [savingRename, setSavingRename] = useState(false);

  useEffect(() => {
    if (backend !== "online") return;
    api
      .listProjects()
      .then((r) => setProjects(r.projects))
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load projects"));
  }, [backend]);

  async function saveRename() {
    if (!renaming) return;
    const name = renameValue.trim();
    if (!name) return;
    setSavingRename(true);
    try {
      const res = await api.renameProject(renaming.id, name);
      setProjects((prev) => prev?.map((p) => (p.id === res.id ? { ...p, name: res.name } : p)) ?? null);
      toast({ title: "Project renamed", description: `Downloads will now use “${res.name}”.`, variant: "success" });
      setRenaming(null);
    } catch (e) {
      toast({
        title: "Rename failed",
        description: e instanceof Error ? e.message : "Could not rename the project.",
        variant: "destructive",
      });
    } finally {
      setSavingRename(false);
    }
  }

  async function remove() {
    if (!pendingDelete) return;
    setDeleting(true);
    try {
      await api.deleteProject(pendingDelete.id);
      setProjects((prev) => prev?.filter((p) => p.id !== pendingDelete.id) ?? null);
      toast({ title: "Project deleted", description: `“${pendingDelete.name}” was removed from this computer.`, variant: "success" });
    } catch (e) {
      toast({
        title: "Delete failed",
        description: e instanceof Error ? e.message : "Could not delete the project.",
        variant: "destructive",
      });
    } finally {
      setDeleting(false);
      setPendingDelete(null);
    }
  }

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Project History</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Every workflow you have saved on this computer — with its results, files and reports.
          </p>
        </div>
        <Button asChild>
          <Link href="/create">
            <Plus className="h-4 w-4" /> New project
          </Link>
        </Button>
      </div>

      {/* Local-first storage notice */}
      <div className="flex items-start gap-2.5 rounded-lg border border-brand-muted/60 bg-brand-muted/10 px-4 py-3 text-sm">
        <HardDrive className="mt-0.5 h-4 w-4 shrink-0 text-brand-red" />
        <span>
          <b>Local projects.</b> Projects created on this computer are stored locally and are available from this
          browser/application. No account is required, and nothing is uploaded to a cloud service. Removing
          DataNova&apos;s local data folder removes this history.
        </span>
      </div>

      {backend === "offline" && (
        <Alert variant="warning">
          <CircleAlert className="h-4 w-4" />
          <AlertTitle>Backend offline</AlertTitle>
          <AlertDescription>Start the FastAPI backend to see your saved projects.</AlertDescription>
        </Alert>
      )}

      {error && (
        <Alert variant="destructive">
          <CircleAlert className="h-4 w-4" />
          <AlertTitle>Could not load projects</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {projects === null && backend === "online" && !error && (
        <div className="space-y-2">
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-16 w-full" />
        </div>
      )}

      {projects?.length === 0 && (
        <Card className="border-dashed">
          <CardContent className="flex flex-col items-center gap-2 py-12 text-center">
            <History className="h-10 w-10 text-muted-foreground/40" />
            <p className="font-medium">No saved projects on this computer yet</p>
            <p className="text-sm text-muted-foreground">
              Finish a generation workflow and save it — it will appear here, stored locally.
            </p>
          </CardContent>
        </Card>
      )}

      {/* Rename dialog */}
      <Dialog open={renaming !== null} onOpenChange={(open) => !open && setRenaming(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename project</DialogTitle>
            <DialogDescription>
              Future CSV and PDF downloads will be named after the new project name.
            </DialogDescription>
          </DialogHeader>
          <Input
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            placeholder={renaming?.name}
            onKeyDown={(e) => e.key === "Enter" && saveRename()}
            autoFocus
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setRenaming(null)} disabled={savingRename}>
              Cancel
            </Button>
            <Button onClick={saveRename} disabled={savingRename || !renameValue.trim()}>
              {savingRename ? "Saving…" : "Save name"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete confirmation dialog */}
      <Dialog open={pendingDelete !== null} onOpenChange={(open) => !open && setPendingDelete(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete “{pendingDelete?.name}”?</DialogTitle>
            <DialogDescription>
              This permanently removes the saved project, its stored results and its generated files from this
              computer. This cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPendingDelete(null)} disabled={deleting}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={remove} disabled={deleting}>
              Delete project
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <div className="space-y-2">
        {projects?.map((p) => (
          <Card key={p.id} className="flex items-center justify-between p-4 transition-shadow hover:shadow-md">
            <Link href={`/projects/${p.id}`} className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium">{p.name}</span>
                <Badge variant={p.mode === "upload" ? "default" : "secondary"}>
                  {p.mode === "upload" ? "expanded dataset" : "from requirements"}
                </Badge>
                {p.target && <Badge variant="outline">target: {p.target}</Badge>}
                {p.task && <Badge variant="outline">{p.task}</Badge>}
                {p.has_report && <Badge variant="success">PDF ready</Badge>}
              </div>
              <div className="mt-1 text-xs text-muted-foreground">
                {new Date(p.created_at).toLocaleString()}
                {p.dataset_filename ? ` · ${p.dataset_filename}` : ""}
                {p.rows_real ? ` · ${p.rows_real.toLocaleString()} real rows` : ""}
                {p.columns ? ` · ${p.columns} columns` : ""}
                {p.rows_synthetic ? ` · ${p.rows_synthetic.toLocaleString()} synthetic rows` : ""}
                {p.synthesizer ? ` · ${p.synthesizer}` : ""}
              </div>
            </Link>
            <div className="flex shrink-0 items-center gap-1.5">
              {p.rows_synthetic ? (
                <Button variant="ghost" size="icon" title="Download synthetic CSV" onClick={() => api.downloadProjectCsv(p.id)}>
                  <FileSpreadsheet className="h-4 w-4" />
                </Button>
              ) : null}
              {p.has_report && (
                <Button variant="ghost" size="icon" title="Download PDF report" onClick={() => api.downloadReport(p.id)}>
                  <FileDown className="h-4 w-4" />
                </Button>
              )}
              <Button
                variant="ghost"
                size="icon"
                title="Rename project"
                onClick={() => {
                  setRenaming(p);
                  setRenameValue(p.name);
                }}
              >
                <Pencil className="h-4 w-4" />
              </Button>
              <Button variant="ghost" size="icon" title="Delete project" onClick={() => setPendingDelete(p)}>
                <Trash2 className="h-4 w-4 text-muted-foreground" />
              </Button>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
