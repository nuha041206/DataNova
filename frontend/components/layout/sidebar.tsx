"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, Sparkles, History, FlaskConical, Settings } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/create", label: "New Project", icon: Sparkles },
  { href: "/projects", label: "Project History", icon: History },
  { href: "/settings", label: "Settings", icon: Settings },
  { href: "/about", label: "Methodology & About", icon: FlaskConical },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-border bg-brand-dark text-white md:flex">
      <Link href="/" className="flex items-center gap-2.5 px-5 pb-6 pt-6">
        {/* DataNova logo mark — transparent PNG, aspect ratio preserved */}
        <img
          src="/logo_mark.png"
          alt="DataNova logo"
          className="h-10 w-10 object-contain"
        />
        <div>
          <div className="font-display text-lg font-bold leading-none">DataNova</div>
          <div className="text-[11px] text-brand-cream/70">Synthetic Data Studio</div>
        </div>
      </Link>

      <nav className="flex-1 space-y-1 px-3">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium text-white/70 transition-colors hover:bg-white/10 hover:text-white",
                active && "bg-brand-red text-white hover:bg-brand-red hover:text-white"
              )}
            >
              <Icon className="h-4 w-4" />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-white/10 p-4">
        <p className="text-[11px] font-medium text-brand-cream/80">Author: Nuha Khan</p>
        <p className="mt-1 text-[11px] leading-relaxed text-white/50">
          Local-first: projects are stored on this computer — no account required. Evaluate synthetic data before
          you trust it.
        </p>
      </div>
    </aside>
  );
}
