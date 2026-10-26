"use client";

import { StepIndicator, type WorkflowStep } from "./step-indicator";

/** Header block used by each workflow page. */
export function StepHeader({
  step,
  title,
  subtitle,
}: {
  step: WorkflowStep;
  title: string;
  subtitle: string;
}) {
  return (
    <div className="border-b bg-background px-6 py-5">
      <StepIndicator current={step} />
      <h1 className="mt-4 text-2xl font-bold tracking-tight">{title}</h1>
      <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p>
    </div>
  );
}
