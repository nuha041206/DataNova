"use client";

import * as React from "react";
import { API_BASE } from "./api";

/** Poll backend health so the UI can show a clear offline banner. */
export function useBackendStatus() {
  const [status, setStatus] = React.useState<"checking" | "online" | "offline">("checking");

  React.useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(4000) });
        if (!cancelled) setStatus(res.ok ? "online" : "offline");
      } catch {
        if (!cancelled) setStatus("offline");
      }
    };
    check();
    const t = setInterval(check, 20000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, []);

  return status;
}
