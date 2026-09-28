"use client";

/**
 * Minimal toast system (shadcn-inspired) built on plain React state:
 *   const { toast } = useToast();
 *   toast({ title: "Saved", description: "…", variant: "success" });
 *
 * Render <ToastViewport /> once in the root layout.
 * Deliberately dependency-free so students can read every line.
 */
import * as React from "react";
import { CircleAlert, CircleCheck, Info } from "lucide-react";
import { cn } from "@/lib/utils";

type ToastVariant = "default" | "success" | "warning" | "destructive";

interface ToastItem {
  id: number;
  title: string;
  description?: string;
  variant: ToastVariant;
}

interface ToastOptions {
  title: string;
  description?: string;
  variant?: ToastVariant;
  /** Auto-dismiss after this many ms; default 5000. */
  duration?: number;
}

type ToastContextValue = {
  toast: (opts: ToastOptions) => void;
  dismiss: (id: number) => void;
};

const ToastContext = React.createContext<ToastContextValue | null>(null);

let nextId = 1;

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = React.useState<ToastItem[]>([]);

  const dismiss = React.useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const toast = React.useCallback(
    ({ title, description, variant = "default", duration = 5000 }: ToastOptions) => {
      const id = nextId++;
      setToasts((prev) => [...prev.slice(-4), { id, title, description, variant }]);
      window.setTimeout(() => dismiss(id), duration);
    },
    [dismiss]
  );

  const value = React.useMemo(() => ({ toast, dismiss }), [toast, dismiss]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <ToastViewport toasts={toasts} onDismiss={dismiss} />
    </ToastContext.Provider>
  );
}

const VARIANT_STYLES: Record<ToastVariant, string> = {
  default: "border bg-background text-foreground",
  success: "border-[#BFC7B6] bg-[#E9EDDF] text-[#3F4438]",
  warning: "border-amber-300 bg-amber-50 text-amber-900",
  destructive: "border-red-300 bg-red-50 text-red-900",
};

const VARIANT_ICONS: Record<ToastVariant, React.ComponentType<{ className?: string }>> = {
  default: Info,
  success: CircleCheck,
  warning: CircleAlert,
  destructive: CircleAlert,
};

function ToastViewport({
  toasts,
  onDismiss,
}: {
  toasts: ToastItem[];
  onDismiss: (id: number) => void;
}) {
  return (
    <div
      aria-live="polite"
      role="region"
      aria-label="Notifications"
      className="pointer-events-none fixed bottom-4 right-4 z-[100] flex w-full max-w-sm flex-col gap-2"
    >
      {toasts.map((t) => {
        const Icon = VARIANT_ICONS[t.variant];
        return (
          <button
            key={t.id}
            type="button"
            onClick={() => onDismiss(t.id)}
            className={cn(
              "pointer-events-auto w-full rounded-lg border p-4 text-left shadow-lg transition-all hover:opacity-95",
              VARIANT_STYLES[t.variant]
            )}
          >
            <span className="flex gap-3">
              <Icon className="mt-0.5 h-4 w-4 shrink-0 opacity-70" />
              <span className="min-w-0">
                <span className="block text-sm font-semibold leading-tight">{t.title}</span>
                {t.description && (
                  <span className="mt-0.5 block text-xs leading-relaxed opacity-80">{t.description}</span>
                )}
              </span>
            </span>
          </button>
        );
      })}
    </div>
  );
}

/** Hook giving access to the `toast` function. Must be used inside <ToastProvider>. */
export function useToast(): ToastContextValue {
  const ctx = React.useContext(ToastContext);
  if (!ctx) {
    throw new Error("useToast must be used inside <ToastProvider>");
  }
  return ctx;
}

/** Convenience type so call sites read nicely: `const { toast } = useToast()`. */
export type { ToastOptions };
