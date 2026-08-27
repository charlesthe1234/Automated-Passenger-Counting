import { LoaderCircle, Play, X } from "lucide-react";
import { useState } from "react";
import { useAuth } from "../../auth/AuthProvider.jsx";
import { endpoints, fetchJson } from "../../lib/api.js";

const RUN_ID_PATTERN = /^[A-Za-z0-9_-]{1,80}$/;

export default function StartRunDialog({ onClose, onStarted }) {
  const { user } = useAuth();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [runId, setRunId] = useState("");
  const [isDemo, setIsDemo] = useState(false);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const runIdInvalid = runId.trim() !== "" && !RUN_ID_PATTERN.test(runId.trim());

  async function handleSubmit(event) {
    event.preventDefault();
    if (submitting || runIdInvalid) return;
    setSubmitting(true);
    setError("");
    try {
      const started = await fetchJson(endpoints.runStart, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: runId.trim() || null,
          name: name.trim() || null,
          description: description.trim() || null,
          is_demo: isDemo,
        }),
      });
      onStarted(started);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not start the run.");
      setSubmitting(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto bg-slate-950/80 px-4 py-8">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-lg rounded-xl border border-slate-700 bg-slate-900 p-5 shadow-2xl"
      >
        <div className="mb-4 flex items-start justify-between gap-3">
          <h3 className="text-base font-black text-white">Start New Run</h3>
          <button type="button" onClick={onClose} aria-label="Close" className="text-slate-400 hover:text-white">
            <X className="h-4 w-4" />
          </button>
        </div>

        <label htmlFor="run-name" className="mb-1 block text-xs font-bold text-slate-300">
          Display name (optional)
        </label>
        <input
          id="run-name"
          autoFocus
          maxLength={120}
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="Morning evacuation exercise"
          className="mb-3 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-300"
        />

        <label htmlFor="run-description" className="mb-1 block text-xs font-bold text-slate-300">
          Description (optional)
        </label>
        <textarea
          id="run-description"
          rows={2}
          maxLength={1000}
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          className="mb-3 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-300"
        />

        <label htmlFor="run-id" className="mb-1 block text-xs font-bold text-slate-300">
          Run ID override (optional)
        </label>
        <input
          id="run-id"
          maxLength={80}
          value={runId}
          onChange={(event) => setRunId(event.target.value)}
          placeholder="Leave blank to generate one automatically"
          className={`w-full rounded-lg border bg-slate-950 px-3 py-2 font-mono text-sm text-white outline-none ${
            runIdInvalid ? "border-red-500/60" : "border-slate-700 focus:border-cyan-300"
          }`}
        />
        <p className="mb-3 mt-1 text-xs text-slate-500">
          {runIdInvalid
            ? "Only letters, numbers, underscore, and hyphen are allowed."
            : "A run ID cannot be changed later. Blank generates one like run_20260731_143000_a1b2."}
        </p>

        <label className="mb-3 flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={isDemo}
            onChange={(event) => setIsDemo(event.target.checked)}
            className="h-4 w-4 rounded border-slate-600 bg-slate-950"
          />
          This is a demo or test run
        </label>

        <p className="mb-3 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-400">
          Starting as <span className="font-bold text-slate-200">{user?.display_name}</span>. All
          data captured during this run is recorded against it.
        </p>

        {error ? <p className="mb-3 text-xs text-red-300">{error}</p> : null}

        <div className="flex gap-2">
          <button
            type="submit"
            disabled={submitting || runIdInvalid}
            className="inline-flex flex-1 items-center justify-center gap-2 rounded-lg bg-cyan-300 px-4 py-2.5 text-sm font-black text-slate-950 transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {submitting ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {submitting ? "Starting…" : "Start Run"}
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
