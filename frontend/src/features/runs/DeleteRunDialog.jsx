import { AlertTriangle, LoaderCircle, Trash2, X } from "lucide-react";
import { useState } from "react";
import { endpoints, fetchJson } from "../../lib/api.js";

export default function DeleteRunDialog({ run, onClose, onDeleted }) {
  const [typed, setTyped] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  // Exact, case-sensitive match. The backend enforces this too.
  const matches = typed === run.run_id;

  async function handleSubmit(event) {
    event.preventDefault();
    if (!matches || submitting) return;
    setSubmitting(true);
    setError("");
    try {
      const summary = await fetchJson(endpoints.runDelete(run.run_id), {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirm_run_id: typed }),
      });
      onDeleted(summary);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not delete the run.");
      setSubmitting(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto bg-slate-950/80 px-4 py-8">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-lg rounded-xl border border-red-500/40 bg-slate-900 p-5 shadow-2xl"
      >
        <div className="mb-3 flex items-start justify-between gap-3">
          <h3 className="flex items-center gap-2 text-base font-black text-white">
            <AlertTriangle className="h-4 w-4 text-red-300" />
            Permanently delete this run
          </h3>
          <button type="button" onClick={onClose} aria-label="Close" className="text-slate-400 hover:text-white">
            <X className="h-4 w-4" />
          </button>
        </div>

        <p className="mb-3 text-sm text-slate-300">
          This permanently removes the following for{" "}
          <span className="font-mono font-bold text-white">{run.run_id}</span>. It cannot be undone.
        </p>
        <ul className="mb-3 space-y-1 rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-slate-300">
          <li>{run.metric_count} metric records</li>
          <li>{run.alert_count} alerts</li>
          <li>{run.evacuee_count} evacuee identities</li>
          <li>{run.gallery_view_count} gallery images (passenger evidence)</li>
          <li>{run.observation_count} legacy observations</li>
        </ul>
        <p className="mb-3 text-xs text-slate-400">
          Other runs are unaffected, and the database file itself is never deleted.
        </p>

        <label htmlFor="confirm-run-id" className="mb-1 block text-xs font-bold text-slate-300">
          Type <span className="font-mono text-white">{run.run_id}</span> to confirm
        </label>
        <input
          id="confirm-run-id"
          autoFocus
          autoComplete="off"
          value={typed}
          onChange={(event) => setTyped(event.target.value)}
          className="mb-3 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-sm text-white outline-none focus:border-red-400"
        />

        {error ? <p className="mb-3 text-xs text-red-300">{error}</p> : null}

        <div className="flex gap-2">
          <button
            type="submit"
            disabled={!matches || submitting}
            className="inline-flex flex-1 items-center justify-center gap-2 rounded-lg bg-red-500 px-4 py-2.5 text-sm font-black text-white transition hover:bg-red-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {submitting ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
            {submitting ? "Deleting…" : "Delete Run"}
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-slate-700 px-4 py-2.5 text-sm font-bold text-slate-300 hover:text-white"
          >
            Cancel
          </button>
        </div>
      </form>
    </div>
  );
}
