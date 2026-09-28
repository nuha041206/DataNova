"use client";

/**
 * Shared evaluation view used by the workflow (session) and the project
 * detail page. Renders score, per-column comparison, charts, ML utility
 * experiments and generated explanations.
 */
import { useMemo } from "react";
import { Info } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  CategoryProportionChart,
  CorrelationHeatmaps,
  NumericDistributionChart,
  TargetProportionChart,
} from "@/components/charts/comparison-charts";
import type { ClassificationMetrics, Evaluation, Finding, MLUtility, RegressionMetrics } from "@/lib/types";
import { cn } from "@/lib/utils";

function simColor(sim: number | null): string {
  if (sim === null) return "text-muted-foreground";
  if (sim >= 0.9) return "text-[#4C6B57]";
  if (sim >= 0.7) return "text-[#A15C00]";
  return "text-red-600";
}

function FindingsList({ findings }: { findings: Finding[] }) {
  if (!findings.length) return null;
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {findings.map((f, i) => (
        <div
          key={i}
          className={cn(
            "rounded-lg border p-3",
            f.kind === "positive" && "border-[#BFC7B6] bg-[#E9EDDF]",
            f.kind === "warning" && "border-amber-200 bg-amber-50",
            f.kind === "neutral" && "bg-muted/40"
          )}
        >
          <div className="text-sm font-semibold">{f.title}</div>
          <p className="mt-1 text-sm text-muted-foreground">{f.text}</p>
        </div>
      ))}
    </div>
  );
}

export function ScoreCard({ evaluation }: { evaluation: Evaluation }) {
  const score = evaluation.data_nova_score;
  const pct = score.value === null ? 0 : Math.round(score.value * 100);
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">DataNova Similarity Score</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-baseline gap-2">
          <span className="text-4xl font-bold" data-testid="datanova-score">
            {score.value === null ? "—" : score.value.toFixed(3)}
          </span>
          <span className="text-sm text-muted-foreground">/ 1.000</span>
        </div>
        <Progress value={pct} />
        <div className="grid gap-2 sm:grid-cols-2">
          {Object.entries(score.components).map(([k, v]) => (
            <div key={k} className="flex items-center justify-between rounded-md border px-3 py-1.5 text-sm">
              <span className="capitalize text-muted-foreground">
                {k} <span className="text-xs">({Math.round((score.weights[k] ?? 0) * 100)}%)</span>
              </span>
              <span className={cn("font-semibold", simColor(v))}>{v === null ? "—" : v.toFixed(3)}</span>
            </div>
          ))}
        </div>
        <Alert variant="info">
          <Info className="h-4 w-4" />
          <AlertDescription className="text-xs">{score.definition}</AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  );
}

function ColumnComparisonTable({ evaluation }: { evaluation: Evaluation }) {
  return (
    <div className="max-h-[26rem] overflow-auto rounded-md border">
      <Table>
        <TableHeader className="sticky top-0 bg-muted">
          <TableRow>
            <TableHead>Column</TableHead>
            <TableHead>Type</TableHead>
            <TableHead>Similarity</TableHead>
            <TableHead>Real → Synthetic (mean or top category)</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {evaluation.column_results.map((c) => {
            let detail = "—";
            const fmtNum = (v: number | null | undefined) =>
              v === null || v === undefined ? "—" : Math.abs(v) >= 1000 ? v.toFixed(1) : Number(v.toFixed(4)).toString();
            if (c.stats) {
              detail = `${fmtNum(c.stats.mean.real)} → ${fmtNum(c.stats.mean.synthetic)}`;
            } else if (c.proportions?.length) {
              const p = c.proportions[0];
              detail = `'${p.category}': ${(p.real_share * 100).toFixed(1)}% → ${(p.synthetic_share * 100).toFixed(1)}%`;
            }
            return (
              <TableRow key={c.column}>
                <TableCell className="font-medium">{c.column}</TableCell>
                <TableCell>
                  <Badge variant="secondary" className="capitalize">{c.type}</Badge>
                </TableCell>
                <TableCell className={cn("font-semibold", simColor(c.similarity))}>
                  {c.similarity === null ? "—" : c.similarity.toFixed(3)}
                </TableCell>
                <TableCell className="text-muted-foreground">{detail}</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

function ConfusionMatrix({ cm }: { cm: { labels: string[]; matrix: number[][] } }) {
  return (
    <div className="overflow-auto">
      <table className="text-xs">
        <thead>
          <tr>
            <th className="p-1" />
            {cm.labels.map((l) => (
              <th key={l} className="p-1 font-medium">{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cm.matrix.map((row, i) => (
            <tr key={i}>
              <th className="p-1 text-right font-medium">{cm.labels[i]}</th>
              {row.map((v, j) => (
                <td
                  key={j}
                  className="border p-1 text-center"
                  style={{ backgroundColor: `rgba(109,8,8,${Math.min(v / Math.max(...row.flat(), 1), 0.9)})` }}
                >
                  <span className={v / Math.max(...row.flat(), 1) > 0.5 ? "text-white" : ""}>{v}</span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function MLUtilityView({ ml }: { ml: MLUtility }) {
  const isClf = ml.task === "classification";
  // Narrow the metric union with the actual task type (no `any`).
  const aClf = ml.experiment_a.metrics as ClassificationMetrics;
  const bClf = ml.experiment_b.metrics as ClassificationMetrics;
  const aReg = ml.experiment_a.metrics as RegressionMetrics;
  const bReg = ml.experiment_b.metrics as RegressionMetrics;

  const rows: { name: string; a: number; b: number; higherBetter: boolean }[] = isClf
    ? [
        { name: "Accuracy", a: aClf.accuracy, b: bClf.accuracy, higherBetter: true },
        { name: "Precision (macro)", a: aClf.precision, b: bClf.precision, higherBetter: true },
        { name: "Recall (macro)", a: aClf.recall, b: bClf.recall, higherBetter: true },
        { name: "F1 (macro)", a: aClf.f1, b: bClf.f1, higherBetter: true },
      ]
    : [
        { name: "MAE", a: aReg.mae, b: bReg.mae, higherBetter: false },
        { name: "MSE", a: aReg.mse, b: bReg.mse, higherBetter: false },
        { name: "RMSE", a: aReg.rmse, b: bReg.rmse, higherBetter: false },
        { name: "R²", a: aReg.r2, b: bReg.r2, higherBetter: true },
      ];

  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Experiment A — {ml.experiment_a.description}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1.5">
              {rows.map((r) => (
                <div key={r.name} className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">{r.name}</span>
                  <span className="font-semibold">{r.a}</span>
                </div>
              ))}
            </div>
            {isClf && aClf.confusion_matrix && (
              <div className="mt-3">
                <div className="mb-1 text-xs text-muted-foreground">Confusion matrix (A)</div>
                <ConfusionMatrix cm={aClf.confusion_matrix} />
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Experiment B — {ml.experiment_b.description}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1.5">
              {rows.map((r) => (
                <div key={r.name} className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">{r.name}</span>
                  <span
                    className={cn(
                      "font-semibold",
                      isClf ? (r.b < r.a * 0.85 ? "text-[#A15C00]" : "text-[#4C6B57]") : "text-foreground"
                    )}
                  >
                    {r.b}
                  </span>
                </div>
              ))}
            </div>
            {isClf && bClf.confusion_matrix && (
              <div className="mt-3">
                <div className="mb-1 text-xs text-muted-foreground">Confusion matrix (B)</div>
                <ConfusionMatrix cm={bClf.confusion_matrix} />
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Setup &amp; retention</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <p className="text-muted-foreground">
            Model: <span className="font-medium text-foreground">{ml.model.label}</span> · split{" "}
            {ml.split.real_train_rows}/{ml.split.real_test_rows} (test {ml.split.test_size * 100}%, seed{" "}
            {ml.split.random_state}) · synthetic training rows: {ml.split.synthetic_train_rows.toLocaleString()}
            {ml.skipped_columns?.length ? ` · skipped columns (missing in synthetic): ${ml.skipped_columns.join(", ")}` : ""}
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            {Object.entries(ml.retention).map(([k, v]) => (
              <div key={k} className="flex items-center justify-between rounded-md border px-3 py-1.5">
                <span className="text-muted-foreground">{k} retention (B vs A)</span>
                <span className="font-semibold">{v === null ? "—" : `${Math.round(v * 100)}%`}</span>
              </div>
            ))}
          </div>
          <Alert variant="info">
            <Info className="h-4 w-4" />
            <AlertDescription className="text-xs">{ml.interpretation_note}</AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    </div>
  );
}

/** Full evaluation tab set: Compare / Charts / ML Utility / Why? */
export function EvaluationView({
  evaluation,
  mlUtility,
  explanation,
}: {
  evaluation: Evaluation;
  mlUtility: MLUtility | null;
  explanation: { evaluation: Finding[]; ml_utility: Finding[]; generation: Finding[] } | null;
}) {
  const numericCols = useMemo(
    () => evaluation.column_results.filter((c) => c.type === "numerical"),
    [evaluation]
  );
  const catCols = useMemo(
    () => evaluation.column_results.filter((c) => c.type !== "numerical"),
    [evaluation]
  );

  return (
    <Tabs defaultValue="compare" className="space-y-4">
      <TabsList>
        <TabsTrigger value="compare">Compare</TabsTrigger>
        <TabsTrigger value="charts">Charts</TabsTrigger>
        {mlUtility && <TabsTrigger value="ml">ML Utility</TabsTrigger>}
        {explanation && <TabsTrigger value="why">Why these results?</TabsTrigger>}
      </TabsList>

      <TabsContent value="compare" className="space-y-4">
        <div className="grid gap-4 lg:grid-cols-2">
          <ScoreCard evaluation={evaluation} />
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Overview</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Rows (real / synthetic)</span>
                <span className="font-medium">
                  {evaluation.row_counts.real.toLocaleString()} / {evaluation.row_counts.synthetic.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Correlation difference (mean abs)</span>
                <span className="font-medium">{evaluation.correlation_comparison.mean_abs_difference ?? "—"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Target</span>
                <span className="font-medium">{evaluation.target_comparison.target ?? "—"}</span>
                {evaluation.target_comparison.similarity !== null && (
                  <Badge variant="secondary" className="ml-2">
                    similarity {evaluation.target_comparison.similarity.toFixed(3)}
                  </Badge>
                )}
              </div>
              {(evaluation.column_overlap.only_in_real.length > 0 ||
                evaluation.column_overlap.only_in_synthetic.length > 0) && (
                <p className="text-xs text-muted-foreground">
                  Columns only in real: {evaluation.column_overlap.only_in_real.join(", ") || "none"} · only in
                  synthetic: {evaluation.column_overlap.only_in_synthetic.join(", ") || "none"}
                </p>
              )}
            </CardContent>
          </Card>
        </div>
        <ColumnComparisonTable evaluation={evaluation} />
      </TabsContent>

      <TabsContent value="charts" className="space-y-6">
        {evaluation.target_comparison.class_proportions?.length ? (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Target distribution — {evaluation.target_comparison.target}</CardTitle>
            </CardHeader>
            <CardContent>
              <TargetProportionChart evaluation={evaluation} />
            </CardContent>
          </Card>
        ) : null}

        {numericCols.slice(0, 6).map((c) => (
          <Card key={c.column}>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">
                {c.column} <span className="text-sm font-normal text-muted-foreground">(quantiles, KS similarity {c.ks?.similarity ?? "—"})</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <NumericDistributionChart col={c} />
            </CardContent>
          </Card>
        ))}

        {catCols.slice(0, 4).map((c) => (
          <Card key={c.column}>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">
                {c.column} <span className="text-sm font-normal text-muted-foreground">(category shares, TVD similarity {c.tvd?.similarity ?? "—"})</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <CategoryProportionChart col={c} />
            </CardContent>
          </Card>
        ))}

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Correlation / association structure</CardTitle>
          </CardHeader>
          <CardContent>
            <CorrelationHeatmaps evaluation={evaluation} />
            <p className="mt-2 text-xs text-muted-foreground">
              Mean absolute difference: {evaluation.correlation_comparison.mean_abs_difference ?? "—"} (0 = identical
              structure). Numeric pairs use |Pearson r|; pairs involving categories use Cramér&apos;s V.
            </p>
          </CardContent>
        </Card>
      </TabsContent>

      {mlUtility && (
        <TabsContent value="ml">
          <MLUtilityView ml={mlUtility} />
        </TabsContent>
      )}

      {explanation && (
        <TabsContent value="why" className="space-y-6">
          <section>
            <h3 className="mb-2 font-semibold">Evaluation</h3>
            <FindingsList findings={explanation.evaluation} />
          </section>
          {explanation.ml_utility.length > 0 && (
            <section>
              <h3 className="mb-2 font-semibold">ML utility</h3>
            </section>
          )}
          <FindingsList findings={explanation.ml_utility} />
          {explanation.generation.length > 0 && (
            <section>
              <h3 className="mb-2 font-semibold">Generation</h3>
              <FindingsList findings={explanation.generation} />
            </section>
          )}
        </TabsContent>
      )}
    </Tabs>
  );
}
