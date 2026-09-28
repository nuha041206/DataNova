"use client";

/**
 * Visual workflow progress for the guided pipeline:
 * Configure -> Generate -> Preview -> Evaluate -> Export
 */
import { Check } from "lucide-react";
import { cn } from "@/lib/utils";

export const WORKFLOW_STEPS = ["Configure", "Generate", "Preview", "Evaluate", "Export"] as const;
export type WorkflowStep = (typeof WORKFLOW_STEPS)[number];

export function StepIndicator({ current }: { current: WorkflowStep }) {
  const currentIdx = WORKFLOW_STEPS.indexOf(current);
  return (
    <ol className="flex flex-wrap items-center gap-y-2 text-sm">
      {WORKFLOW_STEPS.map((step, i) => {
        const done = i < currentIdx;
        const active = i === currentIdx;
        return (
          <li key={step} className="flex items-center">
            <div className="flex items-center gap-2">
              <span
                className={cn(
                  "flex h-7 w-7 items-center justify-center rounded-full border text-xs font-semibold",
                  done && "border-primary bg-primary text-primary-foreground",
                  active && "border-primary text-primary",
                  !done && !active && "border-muted-foreground/30 text-muted-foreground"
                )}
              >
                {done ? <Check className="h-4 w-4" /> : i + 1}
              </span>
              <span
                className={cn(
                  "font-medium",
                  active ? "text-foreground" : "text-muted-foreground",
                  done && "text-foreground"
                )}
              >
                {step}
              </span>
            </div>
            {i < WORKFLOW_STEPS.length - 1 && (
              <span className={cn("mx-3 hidden h-px w-8 sm:block", done ? "bg-primary" : "bg-border")} />
            )}
          </li>
        );
      })}
    </ol>
  );
}
