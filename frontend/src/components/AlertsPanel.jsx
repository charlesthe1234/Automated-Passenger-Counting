import { AlertTriangle } from "lucide-react";

const severityTone = {
  critical: "bg-red-400",
  warning: "bg-amber-300",
  info: "bg-sky-300",
};

function formatTime(value) {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export default function AlertsPanel({ alerts, compact = false }) {
  return (
    <section className={compact ? "" : "rounded-lg border border-slate-800 bg-slate-900/70 p-4"}>
      <div className="mb-3 flex items-center gap-2">
        <AlertTriangle className="h-4 w-4 text-amber-300" />
        <h2 className="text-sm font-bold uppercase tracking-wide text-slate-200">Latest Alerts</h2>
      </div>
      <div className="grid gap-2">
        {alerts.length ? (
          alerts.slice(0, 5).map((alert) => {
            const severity = String(alert.severity || "info").toLowerCase();
            return (
              <div key={alert.id} className="rounded-md bg-slate-800/70 px-3 py-2">
                <div className="flex items-center gap-2">
                  <span className={`h-2.5 w-2.5 rounded-full ${severityTone[severity] || severityTone.info}`} />
                  <span className="text-xs font-bold uppercase text-slate-300">{severity}</span>
                  <span className="ml-auto text-xs text-slate-500">{formatTime(alert.timestamp)}</span>
                </div>
                <p className="mt-1 truncate text-sm text-slate-200">{alert.message}</p>
              </div>
            );
          })
        ) : (
          <div className="rounded-md border border-dashed border-slate-700 px-3 py-4 text-sm text-slate-400">
            No alerts recorded.
          </div>
        )}
      </div>
    </section>
  );
}
