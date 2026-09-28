"use client";

/**
 * Methodology & About — visual, in-app explanation of DataNova.
 * Structured cards/workflow instead of pasted documentation.
 */
import Link from "next/link";
import {
  Activity,
  BarChart3,
  ClipboardList,
  Database,
  FileDown,
  FileText,
  FlaskConical,
  GitCompareArrows,
  ListChecks,
  Scale,
  Target,
  TriangleAlert,
  Wand2,
  XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";

const WORKFLOW = [
  { icon: Database, title: "Real dataset or requirements", text: "Upload a CSV (Mode A) or describe what you need in plain English (Mode B)." },
  { icon: ClipboardList, title: "Dataset analysis", text: "Types, missing values, duplicates, cardinality and warnings — computed from your file." },
  { icon: Target, title: "Target & task detection", text: "Explainable heuristics suggest the target column and whether the task is classification or regression. You can always override." },
  { icon: Wand2, title: "Synthetic generation (SDV)", text: "GaussianCopula, CTGAN or TVAE is chosen from your data; ID-like columns are excluded and re-added as placeholders." },
  { icon: Scale, title: "Real vs synthetic evaluation", text: "Per-column KS/TVD similarities, association structure and target comparison." },
  { icon: Activity, title: "ML utility test", text: "Two experiments on the same real hold-out set measure whether the synthetic data trains useful models." },
  { icon: FileDown, title: "Report & export", text: "Synthetic CSV download plus a PDF evaluation report generated from the stored results." },
];

const METHODOLOGY = [
  {
    icon: ClipboardList,
    title: "Dataset analysis",
    text: "Every column is profiled: inferred type, missing values, unique counts, mean/std/quantiles or top categories, constant and ID-like columns, duplicates. Nothing is modelled before this health check is shown.",
  },
  {
    icon: Wand2,
    title: "SDV synthetic generation",
    text: "GaussianCopula fits per-column marginal distributions plus correlations; CTGAN/TVAE learn deeper patterns with neural networks. DataNova picks a default from dataset size, types and missingness — you can override. Excluded ID columns are restored with placeholder values (SYN000001…).",
  },
  {
    icon: BarChart3,
    title: "Statistical & distribution comparison",
    text: "Numerical columns use the Kolmogorov–Smirnov statistic (similarity = 1 − KS); categorical columns use Total Variation Distance (similarity = 1 − TVD). Both express how closely synthetic distributions track the real ones, from 0 to 1.",
  },
  {
    icon: GitCompareArrows,
    title: "Correlation comparison",
    text: "Absolute Pearson correlations (numeric pairs) and Cramér's V (categorical pairs) are computed for both datasets. The matrices are compared pair-by-pair and summarised as mean/max absolute difference.",
  },
  {
    icon: Target,
    title: "Target comparison",
    text: "For classification, class proportions are compared (TVD-based similarity). For regression, the target distribution is compared with the same KS-based approach. The target is what your models will predict, so it gets its own check.",
  },
];

const SCORE_COMPONENTS = [
  { label: "Statistical similarity", weight: "30%", detail: "Per-column KS / TVD similarity" },
  { label: "Distribution shapes", weight: "30%", detail: "Same per-column detail, emphasised separately" },
  { label: "Correlation structure", weight: "25%", detail: "Association matrices, real vs synthetic" },
  { label: "Target distribution", weight: "15%", detail: "Class shares or target statistics" },
];

const LIMITATIONS = [
  { icon: XCircle, text: "Synthetic data is NOT automatically private, anonymous or safe." },
  { icon: XCircle, text: "Statistical similarity does not guarantee privacy — memorised patterns can still leak." },
  { icon: XCircle, text: "Statistical similarity does not guarantee ML usefulness, and vice versa." },
  { icon: TriangleAlert, text: "ML utility depends on the dataset, model, preprocessing and evaluation setup." },
  { icon: TriangleAlert, text: "Privacy testing (e.g. membership inference, re-identification) is a separate task DataNova does not perform." },
];

const TECH_STACK = [
  { group: "Frontend", items: ["Next.js 14", "React 18", "TypeScript", "Tailwind CSS", "shadcn/ui components", "Recharts"] },
  { group: "Backend", items: ["Python", "FastAPI", "Uvicorn", "pandas", "scikit-learn"] },
  { group: "Generation & reports", items: ["SDV (GaussianCopula / CTGAN / TVAE)", "ReportLab", "matplotlib", "SQLAlchemy + SQLite"] },
];

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-5xl space-y-10 p-6">
      {/* -------------------------------------------------- ABOUT DATANOVA */}
      <section className="space-y-4">
        <div className="flex flex-wrap items-center gap-4">
          <img
            src="/logo.png"
            alt="DataNova logo"
            className="h-16 w-16 rounded-xl bg-white object-contain shadow-sm ring-1 ring-border"
          />
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Methodology &amp; About</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              What DataNova does, how every number is computed, and what the results can — and cannot — tell you.
            </p>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">What is DataNova?</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              A local-first synthetic-data studio: it generates realistic-looking tabular data and then measures how
              good that data actually is. Built by Nuha Khan — the DataNova Synthetic Data Studio.
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">What problem does it solve?</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              Real datasets are often small, imbalanced or sensitive. DataNova expands them — or creates lookalikes
              from a description — so students can practise modelling without touching more real data.
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Why synthetic data?</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              More rows for training, balanced classes for rare outcomes, shareable stand-ins for sensitive data —
              but only after evaluation. That evaluation is DataNova&apos;s core.
            </CardContent>
          </Card>
        </div>
      </section>

      {/* -------------------------------------------------- HOW IT WORKS */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">How DataNova works</h2>
        <div className="space-y-2">
          {WORKFLOW.map((step, i) => (
            <div key={step.title} className="flex items-start gap-3">
              <div className="flex flex-col items-center">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-brand-dark text-white">
                  <step.icon className="h-4 w-4" />
                </div>
                {i < WORKFLOW.length - 1 && <div className="h-6 w-px bg-border" />}
              </div>
              <div className="flex-1 rounded-lg border bg-background px-4 py-3">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-brand-red">STEP {i + 1}</span>
                  <span className="text-sm font-semibold">{step.title}</span>
                </div>
                <p className="mt-0.5 text-sm text-muted-foreground">{step.text}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* -------------------------------------------------- METHODOLOGY */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Methodology</h2>
        <div className="grid gap-4 md:grid-cols-2">
          {METHODOLOGY.map((m) => (
            <Card key={m.title}>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-base">
                  <m.icon className="h-4 w-4 text-brand-red" /> {m.title}
                </CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">{m.text}</CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* -------------------------------------------------- ML UTILITY */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">ML utility experiment</h2>
        <Card>
          <CardContent className="space-y-4 pt-5">
            <p className="text-sm text-muted-foreground">
              Statistical similarity alone does not prove the data is useful for machine learning. DataNova therefore
              runs two experiments with the <b>same model and the same real hold-out test set</b>:
            </p>
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-lg border-2 border-brand-dark/30 bg-brand-dark/5 p-4">
                <Badge>Experiment A</Badge>
                <p className="mt-2 text-sm font-semibold">Train on real → test on real</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  The baseline: what the model can achieve when trained on genuine data.
                </p>
              </div>
              <div className="rounded-lg border-2 border-brand-red/30 bg-brand-red/5 p-4">
                <Badge variant="secondary">Experiment B</Badge>
                <p className="mt-2 text-sm font-semibold">Train on synthetic → test on real</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Whether models trained on synthetic data match the baseline. Comparing B to A on an identical
                  hold-out set isolates the effect of the synthetic data itself.
                </p>
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Metrics: accuracy / precision / recall / F1 for classification; MAE / MSE / RMSE / R² for regression.
              Reported as <b>retention</b> (B ÷ A) where meaningful.
            </p>
          </CardContent>
        </Card>
      </section>

      {/* -------------------------------------------------- DATANOVA SCORE */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">DataNova Score</h2>
        <Alert variant="warning">
          <TriangleAlert className="h-4 w-4" />
          <AlertTitle>Project-defined metric — not an industry standard</AlertTitle>
          <AlertDescription className="text-xs leading-relaxed">
            The DataNova Score is a transparent weighted average defined by this project. Its formula and components
            are always shown so you can judge each part yourself. There is no universal industry-standard synthetic
            data score.
          </AlertDescription>
        </Alert>
        <Card>
          <CardContent className="pt-5">
            <div className="space-y-2">
              {SCORE_COMPONENTS.map((c) => (
                <div key={c.label} className="flex items-center justify-between gap-4 rounded-md border px-3 py-2">
                  <div>
                    <div className="text-sm font-medium">{c.label}</div>
                    <div className="text-xs text-muted-foreground">{c.detail}</div>
                  </div>
                  <Badge variant="outline">{c.weight}</Badge>
                </div>
              ))}
            </div>
            <p className="mt-3 text-xs text-muted-foreground">
              Score = 0.30·statistical + 0.30·distribution + 0.25·correlation + 0.15·target. Shown with full component
              breakdown in the Compare tab and in every PDF report.
            </p>
          </CardContent>
        </Card>
      </section>

      {/* -------------------------------------------------- LIMITATIONS */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Limitations — read before trusting the output</h2>
        <div className="grid gap-2 md:grid-cols-2">
          {LIMITATIONS.map((l, i) => (
            <div key={i} className="flex items-start gap-2.5 rounded-md border bg-background px-3 py-2.5">
              <l.icon className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
              <span className="text-sm">{l.text}</span>
            </div>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          Results also depend on dataset size, synthesizer choice, random seed and evaluation configuration — treat
          every number as indicative, not definitive.
        </p>
      </section>

      {/* -------------------------------------------------- TECH STACK */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Tech stack</h2>
        <div className="grid gap-4 md:grid-cols-3">
          {TECH_STACK.map((t) => (
            <Card key={t.group}>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">{t.group}</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-wrap gap-1.5">
                {t.items.map((i) => (
                  <Badge key={i} variant="secondary">{i}</Badge>
                ))}
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      <Separator />
      <div className="flex flex-wrap items-center justify-between gap-3 pb-4">
        <p className="flex items-center gap-2 text-xs text-muted-foreground">
          <FlaskConical className="h-3.5 w-3.5" />
          DataNova — Synthetic Data Studio · Author: Nuha Khan
        </p>
        <div className="flex gap-2">
          <Button asChild variant="outline" size="sm">
            <Link href="/settings">
              <ListChecks className="h-4 w-4" /> Settings
            </Link>
          </Button>
          <Button asChild size="sm">
            <Link href="/create">
              <FileText className="h-4 w-4" /> Create a project
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
}
