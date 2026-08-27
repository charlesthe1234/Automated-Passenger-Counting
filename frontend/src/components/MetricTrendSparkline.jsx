import { Activity, TrendingDown, TrendingUp } from "lucide-react";

function formatTime(value) {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function trendSummary(points) {
  if (points.length < 2) {
    return { label: "No trend", tone: "neutral", Icon: Activity };
  }

  const first = Number(points[0].passenger_count);
  const last = Number(points[points.length - 1].passenger_count);
  const delta = last - first;

  if (delta > 2) {
    return { label: `Rising +${delta}`, tone: "warning", Icon: TrendingUp };
  }
  if (delta < -2) {
    return { label: `Clearing ${delta}`, tone: "good", Icon: TrendingDown };
  }
  return { label: "Stable", tone: "neutral", Icon: Activity };
}

function toneClass(tone) {
  const tones = {
    good: "border-emerald-500/30 bg-emerald-500/10 text-emerald-100",
    warning: "border-amber-500/30 bg-amber-500/10 text-amber-100",
    neutral: "border-slate-700 bg-slate-800/70 text-slate-300",
  };
  return tones[tone] || tones.neutral;
}

function makePolyline(points, width, height, padding) {
  const counts = points.map((point) => Number(point.passenger_count)).filter(Number.isFinite);
  if (counts.length < 2) return "";

  const minCount = Math.min(...counts);
  const maxCount = Math.max(...counts);
  const range = Math.max(maxCount - minCount, 1);
  const innerWidth = width - padding * 2;
  const innerHeight = height - padding * 2;

  return points
    .map((point, index) => {
      const count = Number(point.passenger_count);
      const x = padding + (index / Math.max(points.length - 1, 1)) * innerWidth;
      const y = padding + (1 - (count - minCount) / range) * innerHeight;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export default function MetricTrendSparkline({ points = [] }) {
  const validPoints = points.filter((point) => Number.isFinite(Number(point.passenger_count)));
  const summary = trendSummary(validPoints);
  const Icon = summary.Icon;
  const latest = validPoints[validPoints.length - 1];
  const width = 360;
  const height = 120;
  const polyline = makePolyline(validPoints, width, height, 14);

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/70 p-3">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-slate-300">
          <Activity className="h-4 w-4 text-cyan-300" />
          60-Min Trend
        </div>
        <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-bold ${toneClass(summary.tone)}`}>
          <Icon className="h-3.5 w-3.5" />
          {summary.label}
        </span>
      </div>

      {validPoints.length >= 2 ? (
        <div className="grid gap-3">
          <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Passenger count trend sparkline" className="h-24 w-full">
            <defs>
              <linearGradient id="trendFill" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="rgb(103 232 249)" stopOpacity="0.32" />
                <stop offset="100%" stopColor="rgb(103 232 249)" stopOpacity="0" />
              </linearGradient>
            </defs>
            <polyline
              points={`14,106 ${polyline} 346,106`}
              fill="url(#trendFill)"
              stroke="none"
            />
            <polyline
              points={polyline}
              fill="none"
              stroke="rgb(103 232 249)"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="4"
            />
          </svg>
          <div className="flex items-center justify-between gap-3 text-xs text-slate-500">
            <span>{formatTime(validPoints[0].timestamp)}</span>
            <span className="font-bold text-slate-300">{latest?.passenger_count ?? "-"} people latest</span>
            <span>{formatTime(latest?.timestamp)}</span>
          </div>
        </div>
      ) : (
        <div className="rounded-md border border-dashed border-slate-700 px-3 py-4 text-sm text-slate-400">
          No trend data yet. Add at least two metric rows to draw the 60-minute sparkline.
        </div>
      )}
    </section>
  );
}
