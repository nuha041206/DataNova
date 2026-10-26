"use client";

/**
 * Target & task detection card.
 * Shows the suggestion with confidence + reason, and lets the user override
 * target and task manually (the detection is only ever a suggestion).
 */
import { Target } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { Detection } from "@/lib/types";
import { INTERPRETATION_NOTES } from "@/lib/notes";

const CONFIDENCE_VARIANT: Record<string, "success" | "warning" | "secondary"> = {
  high: "success",
  medium: "warning",
  low: "secondary",
  user: "success",
};

export function DetectionCard({
  detection,
  columns,
  onTargetChange,
}: {
  detection: Detection;
  columns: string[];
  onTargetChange: (target: string | null) => void;
}) {
  const task = detection.task;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Target className="h-4 w-4 text-primary" /> Target &amp; task detection
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <div>
            <div className="text-xs text-muted-foreground">Suggested target</div>
            <div className="text-lg font-semibold">{detection.suggested_target ?? "None found"}</div>
          </div>
          <Badge variant={CONFIDENCE_VARIANT[detection.confidence] ?? "secondary"}>
            {detection.confidence === "user" ? "manually set" : `${detection.confidence} confidence`}
          </Badge>
          {task !== "unknown" && <Badge variant="outline">{task}</Badge>}
        </div>

        <p className="rounded-md bg-muted px-3 py-2 text-sm text-muted-foreground">
          <span className="font-medium text-foreground">Why: </span>
          {detection.reason}
        </p>

        {/* Manual override — always available */}
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="target-override">Target column (override)</Label>
            <Select
              value={detection.suggested_target ?? ""}
              onValueChange={(v) => onTargetChange(v === "__none__" ? null : v)}
            >
              <SelectTrigger id="target-override">
                <SelectValue placeholder="Select target column" />
              </SelectTrigger>
              <SelectContent>
                {columns.map((c) => (
                  <SelectItem key={c} value={c}>
                    {c}
                  </SelectItem>
                ))}
                <SelectItem value="__none__">No target (unsupervised)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label>Task type (auto from target)</Label>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex h-9 items-center rounded-md border px-3 text-sm text-muted-foreground">
                  {task === "unknown" ? "unknown — pick a target" : task}
                </div>
              </TooltipTrigger>
              <TooltipContent>
                Task is derived from the target column&apos;s type and cardinality: numeric with many unique values
                means regression, otherwise classification.
              </TooltipContent>
            </Tooltip>
          </div>
        </div>

        <p className="text-xs text-muted-foreground">{INTERPRETATION_NOTES.detectionIsSuggestion}</p>

        {detection.candidates.length > 1 && (
          <div className="rounded-md border px-3 py-2">
            <div className="mb-1 text-xs font-medium text-muted-foreground">Other candidates considered</div>
            <ul className="space-y-1 text-xs text-muted-foreground">
              {detection.candidates.slice(1, 4).map((c) => (
                <li key={c.column}>
                  <span className="font-medium text-foreground">{c.column}</span> ({c.task}, score{" "}
                  {c.score.toFixed(2)}) — {c.reason}
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
