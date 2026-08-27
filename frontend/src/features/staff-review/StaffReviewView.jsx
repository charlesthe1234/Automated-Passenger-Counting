import { LoaderCircle, RefreshCw, ShieldCheck, UsersRound } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { endpoints, fetchJson, withQuery } from "../../lib/api.js";
import usePoll from "../../lib/usePoll.js";
import StaffCard from "./StaffCard.jsx";
import StaffGalleryModal from "./StaffGalleryModal.jsx";

const POLL_MS = 3000;

const ROLE_FILTERS = [
  { id: "all", label: "All Staff" },
  { id: "cag", label: "CAG" },
  { id: "scdf", label: "SCDF" },
];

export default function StaffReviewView({
  runId = "",
  isHistorical = false,
  runReady = true,
  hasNoRuns = false,
}) {
  const [staff, setStaff] = useState([]);
  const [roleFilter, setRoleFilter] = useState("all");
  const [selectedPerson, setSelectedPerson] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const queryUrl = useMemo(() => withQuery(endpoints.staff, { run_id: runId }), [runId]);

  async function loadStaff() {
    if (!runId) return;
    setIsLoading(true);
    try {
      const data = await fetchJson(queryUrl);
      setStaff(Array.isArray(data) ? data : []);
      setError("");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not load staff records.");
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    setRoleFilter("all");
    setSelectedPerson(null);
    setStaff([]);
    setError("");
    setIsLoading(Boolean(runId));
  }, [runId, runReady]);

  usePoll(
    async (signal) => {
      try {
        const data = await fetchJson(queryUrl, { signal });
        setStaff(Array.isArray(data) ? data : []);
        setError("");
      } catch (nextError) {
        if (nextError?.name === "AbortError") return;
        setError(nextError instanceof Error ? nextError.message : "Could not load staff records.");
      } finally {
        setIsLoading(false);
      }
    },
    {
      intervalMs: POLL_MS,
      enabled: runReady && Boolean(runId),
      deps: [queryUrl, runReady, runId],
    },
  );

  useEffect(() => {
    setSelectedPerson((current) => {
      if (!current) return null;
      return staff.find((person) => person.id === current.id) || current;
    });
  }, [staff]);

  const visibleStaff = useMemo(
    () => (roleFilter === "all" ? staff : staff.filter((person) => person.role === roleFilter)),
    [roleFilter, staff],
  );

  let emptyTitle = "No staff detected in this run";
  let emptyMessage = "People predicted as CAG or SCDF will appear here with their available evidence.";
  if (!runId) {
    emptyTitle = hasNoRuns ? "No runs recorded yet" : "No run selected";
    emptyMessage = hasNoRuns
      ? "Start a run from Operations before reviewing staff predictions."
      : "Select a run before reviewing staff predictions.";
  } else if (roleFilter !== "all") {
    emptyTitle = `No ${roleFilter.toUpperCase()} staff detected`;
    emptyMessage = `No one in this run is currently predicted as ${roleFilter.toUpperCase()}.`;
  }

  return (
    <section className="grid gap-5">
      <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-4">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="mb-2 inline-flex items-center gap-2 text-sm font-bold uppercase tracking-wide text-cyan-200">
              <ShieldCheck className="h-4 w-4" />
              {isHistorical ? "Historical staff review" : "Staff identification review"}
            </div>
            <h2 className="text-2xl font-black text-white">Staff Review</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
              Review every person currently predicted as CAG or SCDF for the selected run. These classifications are
              model-generated and should be checked against the available evidence.
            </p>
          </div>
          <button
            type="button"
            onClick={loadStaff}
            disabled={!runId || isLoading}
            className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm font-bold text-slate-200 transition hover:border-slate-500 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isLoading ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            Refresh
          </button>
        </div>

        <div className="mt-4 inline-flex flex-wrap gap-2" role="group" aria-label="Filter staff by predicted role">
          {ROLE_FILTERS.map((filter) => {
            const selected = roleFilter === filter.id;
            const selectedStyle =
              filter.id === "cag"
                ? "border-yellow-300 bg-yellow-400/15 text-yellow-100"
                : filter.id === "scdf"
                  ? "border-orange-400 bg-orange-500/15 text-orange-100"
                  : "border-cyan-300 bg-cyan-300/15 text-cyan-100";
            return (
              <button
                key={filter.id}
                type="button"
                onClick={() => setRoleFilter(filter.id)}
                aria-pressed={selected}
                className={`rounded-lg border px-4 py-2 text-sm font-black transition ${
                  selected
                    ? selectedStyle
                    : "border-slate-700 bg-slate-950 text-slate-400 hover:border-slate-500 hover:text-white"
                }`}
              >
                {filter.label}
              </button>
            );
          })}
        </div>

        <div className="mt-4 rounded-lg border border-amber-400/30 bg-amber-500/10 px-4 py-3 text-sm leading-6 text-amber-100">
          Bright or high-visibility clothing can influence the model. A person shown here is not necessarily confirmed staff.
        </div>
      </div>

      {error ? (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm font-bold text-red-100">
          {error}
        </div>
      ) : null}

      <div className="text-right text-xs text-slate-500">Updated automatically every 3 seconds</div>

      {isLoading && !staff.length ? (
        <div className="flex min-h-52 items-center justify-center rounded-lg border border-slate-800 bg-slate-900/40 text-sm font-bold text-slate-400">
          <LoaderCircle className="mr-2 h-5 w-5 animate-spin text-cyan-300" />
          Loading staff evidence
        </div>
      ) : visibleStaff.length ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {visibleStaff.map((person) => (
            <StaffCard key={person.id} person={person} onOpen={setSelectedPerson} />
          ))}
        </div>
      ) : (
        <div className="rounded-lg border border-dashed border-slate-700 bg-slate-900/40 px-4 py-12 text-center">
          <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-slate-800">
            <UsersRound className="h-5 w-5 text-slate-400" />
          </div>
          <div className="text-lg font-bold text-white">{emptyTitle}</div>
          <p className="mt-1 text-sm text-slate-400">{emptyMessage}</p>
        </div>
      )}

      {selectedPerson ? (
        <StaffGalleryModal person={selectedPerson} onClose={() => setSelectedPerson(null)} />
      ) : null}
    </section>
  );
}
