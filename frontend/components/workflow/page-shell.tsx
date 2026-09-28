"use client";

/** Sticky bottom action bar for workflow pages. */
export function ActionBar({ children }: { children: React.ReactNode }) {
  return (
    <div className="sticky bottom-0 border-t bg-background/95 px-6 py-3 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      <div className="flex items-center justify-end gap-3">{children}</div>
    </div>
  );
}
