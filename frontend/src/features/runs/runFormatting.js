export const STATUS_STYLES = {
  starting: "bg-amber-500/15 text-amber-100",
  active: "bg-emerald-500/15 text-emerald-100",
  ending: "bg-amber-500/15 text-amber-100",
  ended: "bg-slate-700/60 text-slate-200",
  failed: "bg-red-500/15 text-red-100",
  interrupted: "bg-orange-500/15 text-orange-100",
  external: "bg-sky-500/15 text-sky-100",
};

export const ORIGIN_LABELS = {
  managed: "Managed",
  legacy: "Imported legacy",
  external: "External / unmanaged",
};

export function formatDateTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

export function formatDuration(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

/** Live duration for an in-progress run, stored duration otherwise. */
export function liveDurationSeconds(run, nowMs) {
  if (!run) return 0;
  if (run.is_in_progress && run.started_at) {
    return Math.max(0, (nowMs - new Date(run.started_at).getTime()) / 1000);
  }
  return run.duration_seconds || 0;
}

export function isUnmanaged(run) {
  return run?.origin_type === "external";
}
