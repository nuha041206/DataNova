import type { Metadata } from "next";
import { Inter, Playfair_Display } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ToastProvider } from "@/components/ui/toast";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const playfair = Playfair_Display({
  subsets: ["latin"],
  variable: "--font-playfair",
  display: "swap",
});

export const metadata: Metadata = {
  title: "DataNova — Synthetic Data Studio",
  description:
    "Generate synthetic datasets with SDV and evaluate their statistical similarity and ML usefulness.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${playfair.variable}`}>
      <body>
        <TooltipProvider delayDuration={200}>
          <ToastProvider>
            <div className="flex min-h-screen font-sans">
              <Sidebar />
              <div className="flex min-w-0 flex-1 flex-col">
                <Header />
                <main className="flex-1 overflow-x-hidden bg-muted/30">{children}</main>
              </div>
            </div>
          </ToastProvider>
        </TooltipProvider>
      </body>
    </html>
  );
}
