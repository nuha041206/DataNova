"use client";

/**
 * Create Project — mode chooser.
 * Balanced two-card layout with feature lists and a "how it works" strip,
 * designed to fill the page without meaningless filler.
 */
import { Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  ArrowRight,
  ClipboardCheck,
  Database,
  FileDown,
  Gauge,
  ShieldCheck,
  Sparkles,
  Table2,
  Wand2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const FLOW = ["Configure", "Understand schema", "Generate", "Validate", "Evaluate", "Export"];

function ModeChooser() {
  const router = useRouter();
  const search = useSearchParams();
  const preselected = search.get("mode");

  return (
    <div className="mx-auto w-full max-w-5xl space-y-10 px-6 py-10">
      {/* Page header */}
      <header className="mx-auto max-w-2xl text-center">
        <div className="mb-3 inline-flex items-center gap-2 rounded-full bg-brand-muted/20 px-3 py-1 text-xs font-medium text-brand-red">
          <Sparkles className="h-3.5 w-3.5" /> Synthetic Data Studio
        </div>
        <h1 className="text-3xl font-bold tracking-tight">Create a New Project</h1>
        <p className="mt-2 text-base text-muted-foreground">
          Choose how you want DataNova to generate your synthetic dataset — from an existing file, or from a plain
          description of what you need.
        </p>
      </header>

      {/* Mode cards */}
      <div className="grid gap-6 md:grid-cols-2">
        <Card className="group flex flex-col transition-all hover:border-brand-red/60 hover:shadow-lg">
          <CardHeader className="pb-4">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-brand-dark text-white">
              <Database className="h-6 w-6" />
            </div>
            <CardTitle className="text-xl">Expand an Existing Dataset</CardTitle>
            <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
              Use an existing CSV and generate additional records while preserving its structure, distributions and
              relationships.
            </p>
          </CardHeader>
          <CardContent className="flex flex-1 flex-col justify-between gap-6">
            <ul className="space-y-2.5 text-sm text-muted-foreground">
              {[
                { icon: Gauge, text: "Dataset health analysis" },
                { icon: ClipboardCheck, text: "Automatic target detection" },
                { icon: Table2, text: "Semantic datatype detection" },
                { icon: Sparkles, text: "SDV-based generation" },
                { icon: ShieldCheck, text: "Real vs synthetic evaluation" },
              ].map(({ icon: Icon, text }) => (
                <li key={text} className="flex items-center gap-2.5">
                  <Icon className="h-4 w-4 shrink-0 text-brand-red" />
                  {text}
                </li>
              ))}
            </ul>
            <Button
              className="w-full"
              size="lg"
              variant={preselected === "upload" ? "default" : "outline"}
              onClick={() => router.push("/create/upload")}
            >
              Start with a CSV <ArrowRight className="h-4 w-4" />
            </Button>
          </CardContent>
        </Card>

        <Card className="group flex flex-col transition-all hover:border-brand-red/60 hover:shadow-lg">
          <CardHeader className="pb-4">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-brand-red text-white">
              <Wand2 className="h-6 w-6" />
            </div>
            <CardTitle className="text-xl">Tell DataNova What You Want</CardTitle>
            <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
              Describe the dataset you need and DataNova converts your requirements into an editable schema — with
              names, phones, addresses and more.
            </p>
          </CardHeader>
          <CardContent className="flex flex-1 flex-col justify-between gap-6">
            <ul className="space-y-2.5 text-sm text-muted-foreground">
              {[
                { icon: Wand2, text: "Natural-language requirements" },
                { icon: Table2, text: "Datatype & semantic-type editing" },
                { icon: ShieldCheck, text: "2000+ unique record generation" },
                { icon: Sparkles, text: "Indian data support" },
                { icon: ClipboardCheck, text: "Generation quality validation" },
              ].map(({ icon: Icon, text }) => (
                <li key={text} className="flex items-center gap-2.5">
                  <Icon className="h-4 w-4 shrink-0 text-brand-red" />
                  {text}
                </li>
              ))}
            </ul>
            <Button
              className="w-full"
              size="lg"
              variant={preselected === "requirements" ? "default" : "outline"}
              onClick={() => router.push("/create/requirements")}
            >
              Describe a Dataset <ArrowRight className="h-4 w-4" />
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* How it works */}
      <section className="rounded-2xl border bg-background px-6 py-8">
        <h2 className="text-center text-lg font-semibold">How DataNova works</h2>
        <p className="mt-1 text-center text-sm text-muted-foreground">
          Every project follows the same transparent pipeline.
        </p>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-x-2 gap-y-4">
          {FLOW.map((step, i) => (
            <div key={step} className="flex items-center gap-2">
              <div className="flex items-center gap-2 rounded-full border bg-muted/40 px-4 py-2">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-brand-dark text-[11px] font-bold text-white">
                  {i + 1}
                </span>
                <span className="text-sm font-medium">{step}</span>
              </div>
              {i < FLOW.length - 1 && <ArrowRight className="h-4 w-4 text-muted-foreground/50" />}
            </div>
          ))}
        </div>
      </section>

      {/* Honest footnote */}
      <p className="mx-auto max-w-2xl text-center text-xs leading-relaxed text-muted-foreground">
        <FileDown className="mr-1 inline h-3.5 w-3.5" />
        All generated datasets are clearly synthetic — emails use example.com domains, addresses are pattern-assembled
        and no real person&apos;s data is reproduced. DataNova is a rules-based generator, not a trained AI model.
      </p>
    </div>
  );
}

export default function CreatePage() {
  return (
    <Suspense>
      <ModeChooser />
    </Suspense>
  );
}
