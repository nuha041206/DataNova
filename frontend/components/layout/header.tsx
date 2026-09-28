"use client";

/**
 * Top page header — complements the sidebar on every page.
 * Shows the current section and the live backend status pill.
 */
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Settings } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { useBackendStatus } from "@/lib/backend-status";

const SECTION_TITLES: { match: (p: string) => boolean; title: string }[] = [
  { match: (p) => p === "/", title: "Dashboard" },
  { match: (p) => p.startsWith("/create"), title: "Create Project" },
  { match: (p) => p.startsWith("/projects"), title: "Project History" },
  { match: (p) => p.startsWith("/settings"), title: "Settings" },
  { match: (p) => p.startsWith("/about"), title: "Methodology & About" },
];

export function Header() {
  const pathname = usePathname();
  const backend = useBackendStatus();

  const section = SECTION_TITLES.find((s) => s.match(pathname))?.title ?? "DataNova";

  return (
    <header className="sticky top-0 z-40 flex h-14 items-center gap-3 border-b bg-background/95 px-6 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      <h2 className="text-sm font-medium text-muted-foreground">{section}</h2>

      <div className="ml-auto flex items-center gap-3">
        <Badge variant={backend === "online" ? "success" : backend === "offline" ? "destructive" : "secondary"}>
          Backend {backend === "checking" ? "…" : backend}
        </Badge>
        <Link
          href="/settings"
          title="Settings"
          className="flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        >
          <Settings className="h-4 w-4" />
        </Link>
      </div>
    </header>
  );
}
