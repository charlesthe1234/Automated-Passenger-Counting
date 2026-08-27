import { CheckCircle2, Eye, History, RefreshCw, Trash2, TriangleAlert } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import ExportShiftReportButton from "../reports/ExportShiftReportButton.jsx";
import { endpoints, fetchJson } from "../../lib/api.js";
import DeleteRunDialog from "./DeleteRunDialog.jsx";
import {
  ORIGIN_LABELS,
  STATUS_STYLES,
  formatDateTime,
  formatDuration,
  isUnmanaged,
} from "./runFormatting.js";

const PAGE_SIZE = 25;

export default function RunHistoryView({ selectedHistoricalRun, onSelectHistoricalRun, onChanged }) {
  const [runs, setRuns] = useState([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deletionSummary, setDeletionSummary] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const payload = await fetchJson(endpoints.runs({ limit: PAGE_SIZE, offset }));
      setRuns(payload.items || []);
      setTotal(payload.total || 0);
      setError("");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not load run history.");
    } finally {
      setLoading(false);
    }
  }, [offset]);

  useEffect(() => {
    load();
  }, [load]);

  function handleDeleted(summary) {
    setDeleteTarget(null);
    setDeletionSummary(summary);
    if (selectedHistoricalRun?.run_id === summary.run_id) {
      onSelectHistoricalRun(null);
    }
    load();
    onChanged?.();
  }

  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  return (
    <section className="grid gap-4">
      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 shadow-2xl">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="flex items-center gap-2 text-base font-black text-white">
              <History className="h-4 w-4 text-cyan-300" />
              Run History
            </h2>
            <p className="mt-1 text-sm text-slate-400">
              Every managed, imported, and externally detected run. Select one to view its
              stored data, or export it as a report.
            </p>
          </div>
          <button
            type="button"
            onClick={load}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm font-bold text-slate-200 transition hover:border-slate-500 hover:text-white"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </button>
        </div>

        {selectedHistoricalRun ? (
          <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-cyan-400/30 bg-cyan-400/10 px-3 py-2">
            <p className="text-sm text-cyan-100">
              Viewing historical run{" "}
              <span className="font-mono font-bold">{selectedHistoricalRun.run_id}</span>. Live
              tactical data is hidden while a past run is selected.
            </p>
            <button
              type="button"
              onClick={() => onSelectHistoricalRun(null)}
              className="rounded-lg border border-cyan-300/40 px-2.5 py-1 text-xs font-bold text-cyan-100 hover:bg-cyan-300/10"
            >
              Return to live
            </button>
          </div>
        ) : null}
      </div>

      {deletionSummary ? (
        <div
          className={`rounded-lg border px-4 py-3 text-sm ${
            deletionSummary.file_cleanup_failures
              ? "border-amber-500/30 bg-amber-500/10 text-amber-100"
              : "border-emerald-500/30 bg-emerald-500/10 text-emerald-100"
          }`}
        >
          <p className="flex items-center gap-2 font-bold">
            {deletionSummary.file_cleanup_failures ? (
              <TriangleAlert className="h-4 w-4" />
            ) : (
              <CheckCircle2 className="h-4 w-4" />
            )}
            Deleted {deletionSummary.run_id}
          </p>
          <p className="mt-1 text-xs">
            {deletionSummary.deleted_metrics} metrics · {deletionSummary.deleted_alerts} alerts ·{" "}
            {deletionSummary.deleted_evacuees} evacuees ·{" "}
            {deletionSummary.deleted_gallery_views} gallery views ·{" "}
            {deletionSummary.deleted_images} image files removed
          </p>
          {deletionSummary.file_cleanup_warnings?.map((warning) => (
            <p key={warning} className="mt-1 text-xs">
              {warning} The database records were deleted successfully.
            </p>
          ))}
          <button
            type="button"
            onClick={() => setDeletionSummary(null)}
            className="mt-2 text-xs underline"
          >
            Dismiss
          </button>
        </div>
      ) : null}

      {error ? (
        <p role="alert" className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-100">
          {error}
        </p>
      ) : null}

      <div className="grid gap-3">
        {loading ? (
          <p className="rounded-xl border border-slate-800 bg-slate-900/70 px-4 py-6 text-center text-sm text-slate-400">
            Loading run history…
          </p>
        ) : runs.length === 0 ? (
          <p className="rounded-xl border border-slate-800 bg-slate-900/70 px-4 py-6 text-center text-sm text-slate-400">
            No runs recorded yet. Start one from Operations.
          </p>
        ) : (
          runs.map((run) => (
            <article
              key={run.run_id}
              className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 shadow-2xl"
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="text-sm font-black text-white">{run.name || run.run_id}</h3>
                    <span
                      className={`rounded-full px-2.5 py-1 text-xs font-bold ${
                        STATUS_STYLES[run.status] || "bg-slate-700/60 text-slate-200"
                      }`}
                    >
                      {run.status}
                    </span>
                    <span className="rounded-full bg-slate-800 px-2.5 py-1 text-xs font-bold text-slate-300">
                      {ORIGIN_LABELS[run.origin_type] || run.origin_type}
                    </span>
                    {run.is_demo ? (
                      <span className="rounded-full bg-violet-500/15 px-2.5 py-1 text-xs font-bold text-violet-100">
                        Demo
                      </span>
                    ) : null}
                  </div>
                  <p className="mt-1 font-mono text-xs text-slate-400">{run.run_id}</p>
                  {isUnmanaged(run) ? (
                    <p className="mt-1 text-xs text-sky-200">Not controlled by Run Manager</p>
                  ) : null}
                  {run.failure_reason ? (
                    <p className="mt-1 text-xs text-amber-200">{run.failure_reason}</p>
                  ) : null}
                </div>

                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => onSelectHistoricalRun(run)}
                    className="inline-flex items-center gap-2 rounded-lg border border-slate-700 px-2.5 py-1.5 text-xs font-bold text-slate-200 hover:border-cyan-300 hover:text-white"
                  >
                    <Eye className="h-3.5 w-3.5" />
                    View
                  </button>
                  <ExportShiftReportButton runId={run.run_id} compact />
                  <button
                    type="button"
                    disabled={!run.can_delete}
                    title={run.can_delete ? undefined : "An in-progress run cannot be deleted."}
                    onClick={() => setDeleteTarget(run)}
                    className="inline-flex items-center gap-2 rounded-lg border border-red-500/40 bg-red-500/10 px-2.5 py-1.5 text-xs font-bold text-red-100 transition hover:bg-red-500/20 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    Delete
                  </button>
                </div>
              </div>

              <dl className="mt-4 grid gap-2 text-xs sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
                <div className="rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2.5">
                  <dt className="block font-bold uppercase tracking-wide text-slate-500">
                    {isUnmanaged(run) ? "First data: " : "Started: "}
                  </dt>
                  <dd className="mt-1 block text-slate-200">
                    {formatDateTime(isUnmanaged(run) ? run.first_ingested_at : run.started_at)}
                  </dd>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2.5">
                  <dt className="block font-bold uppercase tracking-wide text-slate-500">
                    {isUnmanaged(run) ? "Last data: " : "Ended: "}
                  </dt>
                  <dd className="mt-1 block text-slate-200">
                    {formatDateTime(isUnmanaged(run) ? run.last_ingested_at : run.ended_at)}
                  </dd>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2.5">
                  <dt className="block font-bold uppercase tracking-wide text-slate-500">Duration</dt>
                  <dd className="mt-1 block text-slate-200">{formatDuration(run.duration_seconds)}</dd>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2.5">
                  <dt className="block font-bold uppercase tracking-wide text-slate-500">Alerts</dt>
                  <dd className="mt-1 block text-slate-200">{run.alert_count}</dd>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2.5">
                  <dt className="block font-bold uppercase tracking-wide text-slate-500">Evacuees</dt>
                  <dd className="mt-1 block text-slate-200">{run.evacuee_count}</dd>
                </div>
              </dl>
            </article>
          ))
        )}
      </div>

      {total > PAGE_SIZE ? (
        <div className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-900/70 px-4 py-3">
          <button
            type="button"
            disabled={offset === 0}
            onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
            className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs font-bold text-slate-200 disabled:opacity-40"
          >
            Previous
          </button>
          <span className="text-xs text-slate-400">
            Page {currentPage} of {pageCount} · {total} runs
          </span>
          <button
            type="button"
            disabled={offset + PAGE_SIZE >= total}
            onClick={() => setOffset(offset + PAGE_SIZE)}
            className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs font-bold text-slate-200 disabled:opacity-40"
          >
            Next
          </button>
        </div>
      ) : null}

      {deleteTarget ? (
        <DeleteRunDialog
          run={deleteTarget}
          onClose={() => setDeleteTarget(null)}
          onDeleted={handleDeleted}
        />
      ) : null}
    </section>
  );
}
