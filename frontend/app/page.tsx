"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  CircleAlert,
  Database,
  FileDown,
  FlaskConical,
  Plus,
  Sparkles,
  Wand2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import type { ProjectSummary } from "@/lib/types";
import { useBackendStatus } from "@/lib/backend-status";

function StatCard({
  icon: Icon,
  label,
  value,
  sub,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  sub: string;
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        <p className="text-xs text-muted-foreground">{sub}</p>
      </CardContent>
    </Card>
  );
}

export default function DashboardPage() {
  const backend = useBackendStatus();
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);

  useEffect(() => {
    if (backend !== "online") return;
    let cancelled = false;
    api
      .listProjects()
      .then((r) => !cancelled && setProjects(r.projects))
      .catch(() => !cancelled && setProjects([]));
    return () => {
      cancelled = true;
    };
  }, [backend]);

  const nProjects = projects?.length ?? 0;
  const nSyntheticRows =
    projects?.reduce((acc, p) => acc + (p.rows_synthetic ?? 0), 0) ?? 0;
  const withReports = projects?.filter((p) => p.has_report).length ?? 0;

  return (
    <div className="mx-auto max-w-6xl space-y-8 p-6">
      {/* Offline banner */}
      {backend === "offline" && (
        <div className="flex items-center gap-3 rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <CircleAlert className="h-5 w-5 shrink-0" />
          <div>
            
<span className="font-semibold">Backend not reachable.</span>{" "}
Please check the connection to the deployed backend.

          </div>
          <Button asChild size="sm" variant="outline" className="ml-auto">
            <Link href="/about">Setup help</Link>
          </Button>
        </div>
      )}

      {/* Hero — light cream surface, dark red reserved for accents */}
      <section className="overflow-hidden rounded-2xl border bg-brand-cream p-8 shadow-lg">
        <div className="flex flex-wrap items-start justify-between gap-6">
          <div className="max-w-xl">
            <div className="mb-2 inline-flex items-center gap-2 rounded-full bg-brand-red/10 px-3 py-1 text-xs font-medium text-brand-red">
              <Sparkles className="h-3.5 w-3.5" /> Synthetic Data Generation &amp; Quality Evaluation
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-brand-dark">
              Create useful synthetic datasets, understand their quality.
            </h1>
            <p className="mt-2 text-sm text-brand-dark/70">
              Expand an existing dataset with SDV, or describe what you need in plain English — then compare,
              evaluate and export with full transparency.
            </p>
          </div>
          <div className="flex w-full flex-col gap-3 lg:w-auto">
            <Button asChild size="lg" className="bg-brand-red text-white hover:bg-brand-dark">
              <Link href="/create">
                <Plus className="h-5 w-5" /> Create New Project
              </Link>
            </Button>
            <Button asChild size="lg" variant="outline" className="border-brand-dark/30 bg-transparent text-brand-dark hover:bg-brand-red/10 hover:text-brand-dark">
              <Link href="/projects">
                <ArrowRight className="h-5 w-5" /> View History
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="grid gap-4 sm:grid-cols-3">
        <StatCard icon={Database} label="Projects" value={projects ? String(nProjects) : "…"} sub="saved in history" />
        <StatCard
          icon={Activity}
          label="Synthetic rows generated"
          value={projects ? nSyntheticRows.toLocaleString() : "…"}
          sub="across all projects"
        />
        <StatCard icon={FileDown} label="PDF reports" value={projects ? String(withReports) : "…"} sub="generated evaluations" />
      </section>

      {/* Two modes */}
      <section className="grid gap-4 md:grid-cols-2">
        <Card className="group transition-shadow hover:shadow-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Database className="h-5 w-5 text-primary" /> Expand an Existing Dataset
            </CardTitle>
            <CardDescription>
              Upload a real CSV. DataNova checks its health, detects the target and task, then generates matching
              synthetic rows with SDV.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild>
              <Link href="/create?mode=upload">
                Start with a CSV <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card className="group transition-shadow hover:shadow-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Wand2 className="h-5 w-5 text-primary" /> Tell DataNova What You Want
            </CardTitle>
            <CardDescription>
              Describe the dataset in plain English — rows, columns, target, class balance. DataNova interprets it
              and asks before it guesses.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild variant="outline">
              <Link href="/create?mode=requirements">
                Describe a dataset <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
          </CardContent>
        </Card>
      </section>

      {/* Recent projects */}
      <section>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Recent projects</h2>
          <Button asChild variant="ghost" size="sm">
            <Link href="/projects">
              View all <ArrowRight className="h-4 w-4" />
            </Link>
          </Button>
        </div>
        {projects === null ? (
          <div className="space-y-2">
            <Skeleton className="h-14 w-full" />
            <Skeleton className="h-14 w-full" />
            <Skeleton className="h-14 w-full" />
          </div>
        ) : projects.length === 0 ? (
          <Card className="border-dashed">
            <CardContent className="flex flex-col items-center gap-2 py-10 text-center">
              <FlaskConical className="h-10 w-10 text-muted-foreground/40" />
              <p className="font-medium">No projects yet</p>
              <p className="max-w-sm text-sm text-muted-foreground">
                Create your first project to generate and evaluate synthetic data. Your history will appear here.
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-2">
            {projects.slice(0, 4).map((p) => (
              <Link key={p.id} href={`/projects/${p.id}`}>
                <Card className="mb-2 flex items-center justify-between p-4 transition-shadow hover:shadow-md">
                  <div>
                    <div className="font-medium">{p.name}</div>
                    <div className="text-xs text-muted-foreground">
                      {p.mode === "upload" ? "Expanded dataset" : "From requirements"}
                      {p.target ? ` · target: ${p.target}` : ""}
                      {p.task ? ` · ${p.task}` : ""}
                      {" · "}
                      {new Date(p.created_at).toLocaleString()}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {p.rows_synthetic ? (
                      <Badge variant="secondary">{p.rows_synthetic.toLocaleString()} synthetic rows</Badge>
                    ) : null}
                    {p.has_report ? <Badge variant="success">Report</Badge> : null}
                  </div>
                </Card>
              </Link>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
