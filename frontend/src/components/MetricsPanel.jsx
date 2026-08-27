import { Users } from "lucide-react";

function formatTime(value) {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export default function MetricsPanel({ metrics, compact = false }) {
  return (
    <section className={compact ? "" : "rounded-lg border border-slate-800 bg-slate-900/70 p-4"}>
      <div className="mb-3 flex items-center gap-2">
        <Users className="h-4 w-4 text-cyan-300" />
        <h2 className="text-sm font-bold uppercase tracking-wide text-slate-200">Latest Metrics</h2>
      </div>
      <div className="grid gap-2">
        {metrics.length ? (
          metrics.slice(0, 10).map((metric) => (
            <div key={metric.id} className="grid grid-cols-[1fr_auto] gap-3 rounded-md bg-slate-800/70 px-3 py-2">
              <div className="min-w-0">
                <div className="truncate text-xs text-slate-400">{formatTime(metric.timestamp)}</div>
                <div className="truncate text-xs text-slate-500">Run {metric.run_id}</div>
              </div>
              <div className="text-right">
                <div className="text-lg font-black text-white">{metric.passenger_count}</div>
                <div className="text-xs text-slate-500">people</div>
              </div>
            </div>
          ))
        ) : (
          <div className="rounded-md border border-dashed border-slate-700 px-3 py-4 text-sm text-slate-400">
            No metric rows yet.
          </div>
        )}
      </div>
    </section>
  );
}
