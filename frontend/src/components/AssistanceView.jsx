import { AlertTriangle, RefreshCw, Search, ShieldCheck, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import EvacueeCard from "../features/passenger-assistance/EvacueeCard.jsx";
import EvacueeGalleryModal from "../features/passenger-assistance/EvacueeGalleryModal.jsx";
import { AGE_GROUP_FILTERS, selectedAgeGroup } from "../features/passenger-assistance/ageGroups.js";
import { useAuth } from "../auth/AuthProvider.jsx";
import { endpoints, fetchJson, withQuery } from "../lib/api.js";
import usePoll from "../lib/usePoll.js";

const POLL_MS = 3000;

const initialFilters = {
  gender: "",
  min_age: "",
  max_age: "",
  camera_id: "",
  run_id: "",
};

const emptySummary = {
  total_analyzed: 0,
  males: 0,
  females: 0,
  unknown: 0,
  minors: 0,
};

function DemographicsSummary({ summary }) {
  const items = [
    { label: "Males", value: summary.males },
    { label: "Females", value: summary.females },
    { label: "Unknown", value: summary.unknown },
  ];

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/70 p-4">
      <div className="grid gap-4 lg:grid-cols-[minmax(230px,0.8fr)_1.2fr] lg:items-stretch">
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4">
          <div className="text-sm font-bold uppercase tracking-wide text-slate-400">Total Analyzed</div>
          <div className="mt-1 text-6xl font-bold leading-none text-red-500">{summary.total_analyzed}</div>
          <p className="mt-2 text-sm text-slate-400">Run-level baseline from unique ReID evacuee identities.</p>
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          {items.map((item) => (
            <div key={item.label} className="rounded-lg border border-slate-800 bg-slate-950 p-3">
              <div className="text-3xl font-black text-white">{item.value}</div>
              <div className="mt-1 text-xs font-bold uppercase tracking-wide text-slate-500">{item.label}</div>
            </div>
          ))}
        </div>
      </div>
      <div className="mt-3 rounded-lg border border-amber-400/50 bg-amber-500/10 p-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-amber-400/15 text-amber-200">
            <AlertTriangle className="h-5 w-5" />
          </div>
          <div>
            <div className="text-3xl font-black text-white">{summary.minors}</div>
            <div className="text-xs font-bold uppercase tracking-wide text-amber-200">
              Minors (&lt;18) overlapping demographic flag
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function AssistanceView({
  cameras = [],
  runId = "",
  isHistorical = false,
  runReady = true,
  hasNoRuns = false,
}) {
  const { isAdmin } = useAuth();
  const [filters, setFilters] = useState(initialFilters);
  const [evacuees, setEvacuees] = useState([]);
  const [selectedEvacuee, setSelectedEvacuee] = useState(null);
  const [summary, setSummary] = useState(emptySummary);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const [summaryError, setSummaryError] = useState("");

  const effectiveFilters = useMemo(
    () => ({ ...filters, run_id: filters.run_id || runId }),
    [filters, runId],
  );
  const queryUrl = useMemo(
    () => withQuery(endpoints.evacuees, effectiveFilters),
    [effectiveFilters],
  );
  const summaryUrl = useMemo(
    () => withQuery(endpoints.evacueesSummary, { run_id: effectiveFilters.run_id }),
    [effectiveFilters.run_id],
  );

  async function loadEvacuees() {
    try {
      const data = await fetchJson(queryUrl);
      setEvacuees(Array.isArray(data) ? data : []);
      setError("");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not load evacuees.");
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    if (runReady && !effectiveFilters.run_id) {
      setEvacuees([]);
      setSummary(emptySummary);
      setIsLoading(false);
    }
  }, [runReady, effectiveFilters.run_id]);

  usePoll(
    async (signal) => {
      try {
        const data = await fetchJson(queryUrl, { signal });
        setEvacuees(Array.isArray(data) ? data : []);
        setError("");
      } catch (nextError) {
        if (nextError?.name === "AbortError") return;
        setError(nextError instanceof Error ? nextError.message : "Could not load evacuees.");
      } finally {
        setIsLoading(false);
      }
    },
    {
      intervalMs: POLL_MS,
      enabled: runReady && Boolean(effectiveFilters.run_id),
      deps: [queryUrl, runReady, effectiveFilters.run_id],
    },
  );

  usePoll(
    async (signal) => {
      try {
        const data = await fetchJson(summaryUrl, { signal });
        setSummary({ ...emptySummary, ...(data || {}) });
        setSummaryError("");
      } catch (nextError) {
        if (nextError?.name === "AbortError") return;
        setSummary(emptySummary);
        setSummaryError(nextError instanceof Error ? nextError.message : "Could not load summary.");
      }
    },
    {
      intervalMs: POLL_MS,
      enabled: runReady && Boolean(effectiveFilters.run_id),
      deps: [summaryUrl, runReady, effectiveFilters.run_id],
    },
  );

  useEffect(() => {
    setSelectedEvacuee((current) => {
      if (!current) return null;
      return evacuees.find((evacuee) => evacuee.id === current.id) || current;
    });
  }, [evacuees]);

  function updateFilter(key, value) {
    setFilters((current) => ({ ...current, [key]: value }));
  }

  function updateAgeGroup(value) {
    const group = AGE_GROUP_FILTERS.find((option) => option.value === value) || AGE_GROUP_FILTERS[0];
    setFilters((current) => ({ ...current, min_age: group.minAge, max_age: group.maxAge }));
  }

  async function clearDemoLogs() {
    const confirmed = window.confirm("Clear all saved evacuee identities, gallery views, and legacy demo observations?");
    if (!confirmed) return;

    try {
      await fetchJson(endpoints.evacuees, { method: "DELETE" });
      await fetchJson(endpoints.observations, { method: "DELETE" });
      setEvacuees([]);
      setSelectedEvacuee(null);
      setSummary(emptySummary);
      setError("");
      setSummaryError("");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not clear observations.");
    }
  }

  return (
    <section className="grid gap-5">
      <DemographicsSummary summary={summary} />
      {summaryError ? (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm font-bold text-amber-100">
          {summaryError}
        </div>
      ) : null}

      <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="mb-2 inline-flex items-center gap-2 text-sm font-bold uppercase tracking-wide text-cyan-200">
              <ShieldCheck className="h-4 w-4" />
              Assistance Filter
            </div>
            <h2 className="text-2xl font-black text-white">Passenger Assistance</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
              Filter unique ReID evacuee records by model-estimated age group, gender, and last camera. Open a thumbnail
              to compare the available front, side, back, and baseline views manually.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={loadEvacuees}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm font-bold text-slate-200 transition hover:border-slate-500 hover:text-white"
            >
              <RefreshCw className="h-4 w-4" />
              Refresh
            </button>
            {/* Legacy pre-Run-Manager bulk clear. The backend requires an
                admin session plus CSRF; hiding it from staff avoids offering a
                control that would always be rejected. */}
            {isAdmin ? (
              <button
                type="button"
                onClick={clearDemoLogs}
                className="inline-flex items-center gap-2 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm font-bold text-red-100 transition hover:bg-red-500/20"
              >
                <Trash2 className="h-4 w-4" />
                Clear Logs
              </button>
            ) : null}
          </div>
        </div>

        <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <label className="grid gap-1 text-sm">
            <span className="font-bold text-slate-300">Gender</span>
            <select
              value={filters.gender}
              onChange={(event) => updateFilter("gender", event.target.value)}
              className="h-10 rounded-md border border-slate-700 bg-slate-950 px-3 text-sm text-white outline-none focus:border-cyan-300"
            >
              <option value="">Any gender</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="unknown">Unknown / Other</option>
            </select>
          </label>

          <label className="grid gap-1 text-sm">
            <span className="font-bold text-slate-300">Age Group</span>
            <select
              value={selectedAgeGroup(filters)}
              onChange={(event) => updateAgeGroup(event.target.value)}
              className="h-10 rounded-md border border-slate-700 bg-slate-950 px-3 text-sm text-white outline-none placeholder:text-slate-600 focus:border-cyan-300"
            >
              {AGE_GROUP_FILTERS.map((group) => (
                <option key={group.value || "any"} value={group.value}>
                  {group.label}
                </option>
              ))}
            </select>
          </label>

          <label className="grid gap-1 text-sm">
            <span className="font-bold text-slate-300">Camera</span>
            <select
              value={filters.camera_id}
              onChange={(event) => updateFilter("camera_id", event.target.value)}
              className="h-10 rounded-md border border-slate-700 bg-slate-950 px-3 text-sm text-white outline-none focus:border-cyan-300"
            >
              <option value="">Any camera</option>
              {cameras.map((camera) => (
                <option key={camera.camera_id} value={camera.camera_id}>
                  {camera.camera_id}
                </option>
              ))}
            </select>
          </label>

          <label className="grid gap-1 text-sm">
            <span className="font-bold text-slate-300">Run ID</span>
            <input
              type="text"
              value={filters.run_id}
              onChange={(event) => updateFilter("run_id", event.target.value)}
              placeholder="default"
              className="h-10 rounded-md border border-slate-700 bg-slate-950 px-3 text-sm text-white outline-none placeholder:text-slate-600 focus:border-cyan-300"
            />
          </label>
        </div>
        <p className="mt-3 text-xs text-slate-500">Age groups are based on the model estimate and may require manual verification.</p>
      </div>

      {error ? (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm font-bold text-red-100">
          {error}
        </div>
      ) : null}

      <div className="flex items-center justify-between gap-3">
        <div className="inline-flex items-center gap-2 text-sm font-bold uppercase tracking-wide text-slate-300">
          <Search className="h-4 w-4 text-cyan-300" />
          {isLoading ? "Loading evacuees" : `${evacuees.length} matching evacuees`}
        </div>
        <div className="text-xs text-slate-500">Updated every 3 seconds</div>
      </div>

      {evacuees.length ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {evacuees.map((evacuee) => (
            <EvacueeCard key={evacuee.id} evacuee={evacuee} onOpen={setSelectedEvacuee} />
          ))}
        </div>
      ) : (
        <div className="rounded-lg border border-dashed border-slate-700 bg-slate-900/40 px-4 py-12 text-center">
          <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-slate-800">
            <Search className="h-5 w-5 text-slate-400" />
          </div>
          <div className="text-lg font-bold text-white">No matching evacuees</div>
          <p className="mt-1 text-sm text-slate-400">
            Evacuees will appear after the ReID pipeline creates a master identity and uploads its gallery views.
          </p>
        </div>
      )}
      {selectedEvacuee ? (
        <EvacueeGalleryModal evacuee={selectedEvacuee} onClose={() => setSelectedEvacuee(null)} />
      ) : null}
    </section>
  );
}
