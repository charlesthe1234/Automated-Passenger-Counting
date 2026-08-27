import { Gauge } from "lucide-react";

const toneClasses = {
  safe: {
    label: "Safe",
    text: "text-emerald-100",
    badge: "border-emerald-500/30 bg-emerald-500/10 text-emerald-100",
    fill: "bg-emerald-400",
  },
  warning: {
    label: "Warning",
    text: "text-amber-100",
    badge: "border-amber-500/30 bg-amber-500/10 text-amber-100",
    fill: "bg-amber-400",
  },
  critical: {
    label: "Critical",
    text: "text-red-100",
    badge: "border-red-500/30 bg-red-500/10 text-red-100",
    fill: "bg-red-500",
  },
  unknown: {
    label: "Unknown",
    text: "text-slate-300",
    badge: "border-slate-700 bg-slate-800/70 text-slate-300",
    fill: "bg-slate-500",
  },
};

function normalizePercent(value) {
  const percent = Number(value);
  if (!Number.isFinite(percent)) return 0;
  return Math.min(100, Math.max(0, percent));
}

function formatPercent(value) {
  const percent = Number(value);
  if (!Number.isFinite(percent)) return "No capacity";
  return `${percent.toFixed(1)}%`;
}

function formatCapacity(zone) {
  if (zone.capacity === null || zone.capacity === undefined) {
    return `${zone.count} / no capacity`;
  }
  return `${zone.count} / ${zone.capacity}`;
}

export default function ZoneCapacityBars({ zones = [] }) {
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/70 p-3">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-slate-300">
          <Gauge className="h-4 w-4 text-cyan-300" />
          Zone Capacity
        </div>
        <div className="text-xs text-slate-500">Safe &lt;60% · Warning 60-85% · Critical &gt;85%</div>
      </div>

      {zones.length ? (
        <div className="grid gap-3 md:grid-cols-2">
          {zones.map((zone) => {
            const tone = toneClasses[zone.status] || toneClasses.unknown;
            const percent = normalizePercent(zone.percent_used);
            return (
              <article key={zone.zone_id} className="rounded-lg border border-slate-800 bg-slate-950 p-3">
                <div className="mb-2 flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-black text-white">{zone.zone_id}</div>
                    <div className="text-xs text-slate-500">{formatCapacity(zone)} people</div>
                  </div>
                  <span className={`rounded-full border px-2.5 py-1 text-xs font-bold ${tone.badge}`}>
                    {tone.label}
                  </span>
                </div>

                <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                  <div className={`h-full rounded-full ${tone.fill}`} style={{ width: `${percent}%` }} />
                </div>

                <div className="mt-2 flex items-center justify-between text-xs">
                  <span className={`font-bold ${tone.text}`}>{formatPercent(zone.percent_used)}</span>
                  <span className="text-slate-500">capacity used</span>
                </div>
              </article>
            );
          })}
        </div>
      ) : (
        <div className="rounded-md border border-dashed border-slate-700 px-3 py-4 text-sm text-slate-400">
          No zone data yet. Zone bars will appear after metric rows include camera-keyed zone counts.
        </div>
      )}
    </section>
  );
}
