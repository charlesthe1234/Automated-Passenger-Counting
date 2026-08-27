import { useCallback, useState } from "react";
import { endpoints, fetchJson } from "../../lib/api.js";
import usePoll from "../../lib/usePoll.js";

const ACTIVE_POLL_MS = 2000;

/**
 * One source of truth for which run the dashboard is looking at.
 *
 * Components must not guess the run from `metrics[0]` any more: live views
 * follow the active managed run, and selecting a historical run switches the
 * whole dashboard to that run's stored data.
 *
 * There must always be exactly one run in view once loading finishes. An empty
 * run ID is not a harmless default: the API omits blank query values, and the
 * backend reads a missing `run_id` as "every run", which silently sums totals
 * across all runs. So when nothing is live the view falls back to the most
 * recent run — the same rule report export already uses.
 */
export default function useRunState({ isAuthenticated }) {
  const [activeRun, setActiveRun] = useState(null);
  const [externalRun, setExternalRun] = useState(null);
  const [latestRun, setLatestRun] = useState(null);
  const [selectedHistoricalRun, setSelectedHistoricalRun] = useState(null);
  const [ready, setReady] = useState(false);

  const refresh = useCallback(async (signal) => {
    try {
      const [active, external] = await Promise.all([
        fetchJson(endpoints.runActive, { signal }),
        fetchJson(endpoints.runExternalActive, { signal }),
      ]);
      setActiveRun(active || null);
      setExternalRun(external || null);

      // Only needed when nothing is live, so the common case stays at two
      // requests per cycle.
      if (!active && !external) {
        const history = await fetchJson(endpoints.runs({ limit: 1 }), { signal });
        setLatestRun(history?.items?.[0] || null);
      } else {
        setLatestRun(null);
      }
    } catch (error) {
      if (error?.name === "AbortError") return;
      setActiveRun(null);
      setExternalRun(null);
      setLatestRun(null);
    } finally {
      setReady(true);
    }
  }, []);

  usePoll(refresh, {
    intervalMs: ACTIVE_POLL_MS,
    enabled: isAuthenticated,
    deps: [isAuthenticated, refresh],
  });

  const liveRun = activeRun || externalRun;
  const viewedRun = selectedHistoricalRun || liveRun || latestRun || null;
  const viewedRunId = viewedRun?.run_id || "";

  // "Historical" means anything that is not the currently live run, including
  // the most-recent-run fallback. Live tactical dots must never be shown under
  // a run that is not actually producing them.
  const isHistorical = Boolean(viewedRun) && viewedRun.run_id !== liveRun?.run_id;

  // Nothing has ever run on this system; there is genuinely no run to scope to.
  const hasNoRuns = ready && !viewedRun;

  return {
    activeRun,
    externalRun,
    latestRun,
    selectedHistoricalRun,
    setSelectedHistoricalRun,
    viewedRun,
    viewedRunId,
    isHistorical,
    hasNoRuns,
    ready,
    refresh,
  };
}
