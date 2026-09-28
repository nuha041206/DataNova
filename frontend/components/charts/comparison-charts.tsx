"use client";

/**
 * Comparison charts for real vs synthetic evaluation.
 * All data comes from the evaluation payload — nothing is invented here.
 */
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip as ReTooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ColumnEval, Evaluation } from "@/lib/types";

/** Percent formatter for recharts tooltips (loosely typed for recharts v3). */
const pctFormatter = (v: unknown) => `${(Number(v) * 100).toFixed(1)}%`;

const REAL_COLOR = "#400202";   // DataNova darkest — real data
const SYNTH_COLOR = "#D9A441";  // warm amber — synthetic data (accessible contrast pair)

/** Overlay histogram for one numerical column, binned client-side. */
export function NumericDistributionChart({ col }: { col: ColumnEval }) {
  if (!col.stats) return null;
  const { real, synthetic } = col.stats.quantiles as unknown as {
    real: (number | null)[];
    synthetic: (number | null)[];
  };
  // Quantile overlay (5 points) when raw values are unavailable — honest and fast
  const labels = ["min", "Q25", "median", "Q75", "max"];
  const data = labels.map((label, i) => ({
    name: label,
    real: real[i],
    synthetic: synthetic[i],
  }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
        <XAxis dataKey="name" fontSize={11} />
        <YAxis fontSize={11} />
        <ReTooltip />
        <Legend />
        <Bar dataKey="real" fill={REAL_COLOR} radius={[3, 3, 0, 0]} name="Real" />
        <Bar dataKey="synthetic" fill={SYNTH_COLOR} radius={[3, 3, 0, 0]} name="Synthetic" />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Grouped bar chart of category shares for one categorical column. */
export function CategoryProportionChart({ col }: { col: ColumnEval }) {
  if (!col.proportions?.length) return null;
  const data = col.proportions.slice(0, 8).map((p) => ({
    name: String(p.category),
    real: p.real_share,
    synthetic: p.synthetic_share,
  }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
        <XAxis dataKey="name" fontSize={10} interval={0} angle={-18} textAnchor="end" height={48} />
        <YAxis fontSize={11} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
        <ReTooltip formatter={pctFormatter} />
        <Legend />
        <Bar dataKey="real" fill={REAL_COLOR} radius={[3, 3, 0, 0]} name="Real share" />
        <Bar dataKey="synthetic" fill={SYNTH_COLOR} radius={[3, 3, 0, 0]} name="Synthetic share" />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Class shares of the target variable (classification). */
export function TargetProportionChart({ evaluation }: { evaluation: Evaluation }) {
  const props = evaluation.target_comparison.class_proportions;
  if (!props?.length) return null;
  const data = props.map((p) => ({
    name: String(p.category),
    real: p.real_share,
    synthetic: p.synthetic_share,
  }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
        <XAxis dataKey="name" fontSize={11} />
        <YAxis fontSize={11} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
        <ReTooltip formatter={pctFormatter} />
        <Legend />
        <Bar dataKey="real" fill={REAL_COLOR} radius={[3, 3, 0, 0]} name="Real share" />
        <Bar dataKey="synthetic" fill={SYNTH_COLOR} radius={[3, 3, 0, 0]} name="Synthetic share" />
      </BarChart>
    </ResponsiveContainer>
  );
}

function heatColor(v: number): string {
  // 0 -> white, 1 -> DataNova brand red #6D0808
  const alpha = Math.min(Math.max(v, 0), 1);
  return `rgba(109, 8, 8, ${alpha.toFixed(3)})`;
}

/**
 * Side-by-side association heatmaps (real vs synthetic) rendered as CSS grids.
 * Values are |Pearson| for numeric pairs and Cramér's V otherwise (computed
 * by the backend).
 */
export function CorrelationHeatmaps({ evaluation }: { evaluation: Evaluation }) {
  const corr = evaluation.correlation_comparison;
  if (!corr.real_matrix || !corr.synthetic_matrix || !corr.columns) return null;
  const { columns, real_matrix, synthetic_matrix } = corr;

  const Grid = ({ matrix, title }: { matrix: number[][]; title: string }) => (
    <div>
      <div className="mb-1 text-xs font-medium text-muted-foreground">{title}</div>
      <div
        className="grid gap-px overflow-auto rounded-md border text-[10px]"
        style={{ gridTemplateColumns: `minmax(60px,auto) repeat(${columns.length}, minmax(28px, 1fr))` }}
      >
        <div />
        {columns.map((c) => (
          <div key={c} className="max-w-[70px] truncate bg-muted px-1 py-1 text-center" title={c}>
            {c}
          </div>
        ))}
        {matrix.map((row, i) => (
          <div key={i} className="contents">
            <div className="max-w-[70px] truncate bg-muted px-1 py-1 text-right" title={columns[i]}>
              {columns[i]}
            </div>
            {row.map((v, j) => (
              <div
                key={j}
                className="flex h-7 items-center justify-center text-[9px] text-foreground/70"
                style={{ backgroundColor: heatColor(v) }}
                title={`${columns[i]} ↔ ${columns[j]}: ${v}`}
              >
                {i === j ? "" : v.toFixed(2)}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Grid matrix={real_matrix} title="Real data — absolute associations" />
      <Grid matrix={synthetic_matrix} title="Synthetic data — absolute associations" />
    </div>
  );
}
