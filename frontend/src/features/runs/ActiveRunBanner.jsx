import { AlertTriangle, ArrowRight, LoaderCircle, Play, Square } from "lucide-react";
import { useEffect, useState } from "react";
import { useAuth } from "../../auth/AuthProvider.jsx";
import {
  ORIGIN_LABELS,
  STATUS_STYLES,
  formatDateTime,
  formatDuration,
  isUnmanaged,
  liveDurationSeconds,
} from "./runFormatting.js";

/**
 * Replaces the old CV Start/Stop panel. Everyone sees which run is live; only
 * an admin gets the controls.
 */
export default function ActiveRunBanner({
  activeRun,
  externalRun,
  cvStatus,
  isHistorical,
  onReturnToCurrentRun,
  onStartRun,
  onEndRun,
  busy,
  error,
}) {
  const { isAdmin } = useAuth();
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const run = activeRun || externalRun;
  const unmanaged = !activeRun && Boolean(externalRun);
  const cvState = cvStatus?.state || "offline";
  const cvBusy = ["loading", "starting", "stopping"].includes(cvState);

  return (
    <section className="mb-5 rounded-xl border border-slate-800 bg-slate-900/70 p-4 shadow-2xl">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            {cvBusy ? (
              <LoaderCircle className="h-5 w-5 animate-spin text-cyan-300" />
            ) : cvState === "failed" ? (
              <AlertTriangle className="h-5 w-5 text-red-300" />
            ) : (
              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  run ? "bg-emerald-400" : cvState === "ready" ? "bg-cyan-300" : "bg-slate-500"
                }`}
              />
            )}
            <h2 className="text-base font-black text-white">
              {run ? run.name || run.run_id : "No run in progress"}
            </h2>
            {run ? (
              <span
                className={`rounded-full px-2.5 py-1 text-xs font-bold ${
                  STATUS_STYLES[run.status] || "bg-slate-700/60 text-slate-200"
                }`}
              >
                {run.status}
              </span>
            ) : null}
            {unmanaged ? (
              <span className="rounded-full bg-sky-500/15 px-2.5 py-1 text-xs font-bold text-sky-100">
                {ORIGIN_LABELS.external}
              </span>
            ) : null}
          </div>

          {run ? (
            <>
              <p className="mt-1 font-mono text-xs text-slate-400">{run.run_id}</p>
              <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-400">
                <span>
                  {unmanaged ? "First data" : "Started"}:{" "}
                  {formatDateTime(unmanaged ? run.first_ingested_at : run.started_at)}
                </span>
                <span>Duration: {formatDuration(liveDurationSeconds(run, now))}</span>
                <span>Latest count: {run.latest_passenger_count}</span>
                <span>CV worker: {cvState}</span>
              </div>
              {unmanaged ? (
                <p className="mt-2 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-xs text-sky-100">
                  This data comes from a CV pipeline started outside the dashboard.
                  Run Manager did not start it and <strong>cannot stop it</strong> — stop
                  it where it was launched.
                </p>
              ) : null}
            </>
          ) : (
            <p className="mt-1 text-sm text-slate-400">
              {cvState === "ready"
                ? "The computer-vision worker is idle and ready. Start a run to begin capture."
                : cvState === "loading"
                  ? cvStatus?.loading_stage || "Preparing computer vision…"
                  : cvState === "failed"
                    ? cvStatus?.error || "The computer-vision worker is unavailable."
                    : "Waiting for the computer-vision worker."}
            </p>
          )}
        </div>

        <div className="flex min-w-[210px] flex-col gap-2">
          {isHistorical && run ? (
            <button
              type="button"
              onClick={onReturnToCurrentRun}
              className="inline-flex items-center justify-center gap-2 rounded-lg border border-cyan-300/50 bg-cyan-300/10 px-5 py-3 text-sm font-black text-cyan-100 transition hover:bg-cyan-300/20"
            >
              Go to Current Run
              <ArrowRight className="h-4 w-4" />
            </button>
          ) : null}

          {isAdmin && !activeRun ? (
            <button
              type="button"
              onClick={onStartRun}
              disabled={busy || cvState !== "ready"}
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-cyan-300 px-5 py-3 text-sm font-black text-slate-950 transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Play className="h-4 w-4" />
              Start New Run
            </button>
          ) : null}

          {isAdmin && activeRun ? (
            <button
              type="button"
              onClick={onEndRun}
              disabled={busy || activeRun.status === "ending"}
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-red-500 px-5 py-3 text-sm font-black text-white transition hover:bg-red-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Square className="h-4 w-4" />
              {activeRun.status === "ending" ? "Ending…" : "End Run"}
            </button>
          ) : null}

          {!isAdmin ? (
            <p className="text-center text-xs text-amber-200">
              Only an administrator can start or end a run.
            </p>
          ) : null}
          {error ? <p className="text-center text-xs text-red-300">{error}</p> : null}
        </div>
      </div>
    </section>
  );
}
