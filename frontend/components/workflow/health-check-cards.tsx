"use client";

/** Dataset Health Check display: summary stats + per-column table + warnings. */
import { AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { Analysis } from "@/lib/types";
import { cn } from "@/lib/utils";

function fmt(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined) return "—";
  if (Number.isInteger(v)) return String(v);
  return Number(v).toFixed(digits);
}

export function HealthCheckCards({ analysis }: { analysis: Analysis }) {
  const stats = [
    { label: "Rows", value: analysis.rows.toLocaleString() },
    { label: "Columns", value: String(analysis.columns) },
    { label: "Missing cells", value: analysis.total_missing_cells.toLocaleString() },
    { label: "Duplicate rows", value: String(analysis.duplicate_rows) },
    { label: "Memory", value: `${analysis.memory_mb} MB` },
    {
      label: "Column types",
      value: Object.entries(analysis.dtypes_summary)
        .map(([k, v]) => `${k}: ${v}`)
        .join(", "),
    },
  ];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {stats.map((s) => (
          <Card key={s.label}>
            <CardContent className="p-4">
              <div className="text-xs text-muted-foreground">{s.label}</div>
              <div className="mt-1 truncate text-lg font-semibold" title={s.value}>
                {s.value}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {analysis.warnings.length > 0 && (
        <Alert variant="warning">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Health check found {analysis.warnings.length} thing(s) to review</AlertTitle>
          <AlertDescription>
            <ul className="list-inside list-disc space-y-0.5">
              {analysis.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
            <p className="mt-2 text-xs">
              DataNova never modifies your data automatically — review these and decide yourself.
            </p>
          </AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Column profiles</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="max-h-96 overflow-auto rounded-md border">
            <Table>
              <TableHeader className="sticky top-0 bg-muted">
                <TableRow>
                  <TableHead>Column</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Missing</TableHead>
                  <TableHead>Unique</TableHead>
                  <TableHead>Mean</TableHead>
                  <TableHead>Std</TableHead>
                  <TableHead>Min</TableHead>
                  <TableHead>Max</TableHead>
                  <TableHead>Flags</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {analysis.column_profiles.map((c) => (
                  <TableRow key={c.name}>
                    <TableCell className="font-medium">{c.name}</TableCell>
                    <TableCell>
                      <Badge variant="secondary" className="capitalize">
                        {c.inferred_type}
                      </Badge>
                    </TableCell>
                    <TableCell className={cn(c.missing_pct > 40 && "font-semibold text-amber-700")}>
                      {c.missing} ({fmt(c.missing_pct, 1)}%)
                    </TableCell>
                    <TableCell>{c.unique}</TableCell>
                    <TableCell>{fmt(c.mean)}</TableCell>
                    <TableCell>{fmt(c.std)}</TableCell>
                    <TableCell>{fmt(c.min)}</TableCell>
                    <TableCell>{fmt(c.max)}</TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {c.looks_like_id && <Badge variant="outline">ID?</Badge>}
                        {c.is_constant && <Badge variant="outline">constant</Badge>}
                        {c.is_high_cardinality && <Badge variant="outline">high cardinality</Badge>}
                        {c.target_name_hint && <Badge variant="success">target?</Badge>}
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
