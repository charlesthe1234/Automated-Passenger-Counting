import { useCallback, useEffect, useState } from "react";
import { useAuth } from "./auth/AuthProvider.jsx";
import LoginView from "./auth/LoginView.jsx";
import AdminSettingsView from "./features/admin/AdminSettingsView.jsx";
import ActiveRunBanner from "./features/runs/ActiveRunBanner.jsx";
import RunHistoryView from "./features/runs/RunHistoryView.jsx";
import StartRunDialog from "./features/runs/StartRunDialog.jsx";
import useRunState from "./features/runs/useRunState.js";
import StaffReviewView from "./features/staff-review/StaffReviewView.jsx";
import AssistanceView from "./components/AssistanceView.jsx";
import DashboardLayout from "./components/DashboardLayout.jsx";
import MetricTrendSparkline from "./components/MetricTrendSparkline.jsx";
import OperationsSidebarTabs from "./components/OperationsSidebarTabs.jsx";
import OperationsStatusPills from "./components/OperationsStatusPills.jsx";
import TacticalMap from "./components/TacticalMap.jsx";
import VideoPlayer from "./components/VideoPlayer.jsx";
import ZoneCapacityBars from "./components/ZoneCapacityBars.jsx";
import { endpoints, fetchJson, withQuery } from "./lib/api.js";
import usePoll from "./lib/usePoll.js";

const POLL_MS = 3000;
const TACTICAL_POLL_MS = 250;
const CV_POLL_MS = 2000;

export default function App() {
  const { isAuthenticated, isAdmin, loading: authLoading } = useAuth();
  const [activeTab, setActiveTab] = useState("assistance");
  const [cameras, setCameras] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState(null);
  const [metrics, setMetrics] = useState([]);
  const [metricTrend, setMetricTrend] = useState([]);
  const [tacticalState, setTacticalState] = useState(null);
  const [zoneStatus, setZoneStatus] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [apiOnline, setApiOnline] = useState(false);
  const [cvStatus, setCvStatus] = useState(null);
  const [startDialogOpen, setStartDialogOpen] = useState(false);
  const [runBusy, setRunBusy] = useState(false);
  const [runError, setRunError] = useState("");

  const runState = useRunState({ isAuthenticated });
  const {
    activeRun,
    externalRun,
    viewedRunId,
    isHistorical,
    hasNoRuns,
    ready: runReady,
    refresh: refreshRuns,
  } = runState;

  // Split by need: the header shows camera status on every tab, but zones,
  // trends and alerts are Operations-only and must not be fetched elsewhere.
  const onOperations = activeTab === "operations";

  usePoll(
    async (signal) => {
      try {
        const nextCameras = await fetchJson(endpoints.cameras, { signal });
        const normalizedCameras = Array.isArray(nextCameras) ? nextCameras : [];
        setCameras(normalizedCameras);
        setSelectedCameraId((current) => {
          if (current && normalizedCameras.some((camera) => camera.camera_id === current)) {
            return current;
          }
          return normalizedCameras[0]?.camera_id || current;
        });
        setApiOnline(true);
      } catch (error) {
        if (error?.name === "AbortError") return;
        setApiOnline(false);
        setCameras((current) =>
          current.map((camera) => ({
            ...camera,
            camera_connected: false,
            last_error: error instanceof Error ? error.message : "Backend unavailable",
          })),
        );
      }
    },
    { intervalMs: POLL_MS, enabled: isAuthenticated, deps: [isAuthenticated] },
  );

  // Operations-only operational data, scoped to the run being viewed.
  usePoll(
    async (signal) => {
      const runParams = viewedRunId ? { run_id: viewedRunId } : {};
      try {
        const [nextMetrics, nextMetricTrend, nextZoneStatus, nextAlerts] = await Promise.all([
          fetchJson(withQuery(endpoints.metrics, runParams), { signal }),
          fetchJson(withQuery(endpoints.metricTrends, runParams), { signal }),
          fetchJson(withQuery(endpoints.zoneStatus, runParams), { signal }),
          fetchJson(withQuery(endpoints.alerts, runParams), { signal }),
        ]);
        setMetrics(Array.isArray(nextMetrics) ? nextMetrics : []);
        setMetricTrend(Array.isArray(nextMetricTrend) ? nextMetricTrend : []);
        setZoneStatus(Array.isArray(nextZoneStatus) ? nextZoneStatus : []);
        setAlerts(Array.isArray(nextAlerts) ? nextAlerts : []);
      } catch (error) {
        if (error?.name === "AbortError") return;
        setMetricTrend([]);
        setZoneStatus([]);
      }
    },
    {
      intervalMs: POLL_MS,
      enabled: isAuthenticated && onOperations && runReady && Boolean(viewedRunId),
      deps: [isAuthenticated, onOperations, runReady, viewedRunId],
    },
  );

  usePoll(
    async (signal) => {
      try {
        setCvStatus(await fetchJson(endpoints.cvStatus, { signal }));
      } catch (error) {
        if (error?.name !== "AbortError") setCvStatus(null);
      }
    },
    { intervalMs: CV_POLL_MS, enabled: isAuthenticated, deps: [isAuthenticated] },
  );

  const selectedCamera =
    cameras.find((camera) => camera.camera_id === selectedCameraId) || cameras[0] || null;

  // Historical tactical replay does not exist, so live dots must never be
  // presented as belonging to a past run.
  const tacticalEnabled =
    isAuthenticated && onOperations && runReady && !isHistorical && Boolean(viewedRunId);

  useEffect(() => {
    if (!tacticalEnabled) setTacticalState(null);
  }, [tacticalEnabled, viewedRunId]);

  useEffect(() => {
    // Never let one run's numbers linger while another run is being loaded.
    setMetrics([]);
    setMetricTrend([]);
    setZoneStatus([]);
    setAlerts([]);
  }, [viewedRunId]);

  usePoll(
    async (signal) => {
      try {
        setTacticalState(await fetchJson(endpoints.tacticalLatestGlobal(viewedRunId), { signal }));
      } catch (error) {
        if (error?.name === "AbortError") return;
        setTacticalState((current) => ({
          ...(current || {}),
          camera_id: current?.camera_id || "fused",
          has_data: Boolean(current?.has_data),
          stale: true,
        }));
      }
    },
    {
      intervalMs: TACTICAL_POLL_MS,
      enabled: tacticalEnabled,
      deps: [tacticalEnabled, viewedRunId],
    },
  );

  const handleStarted = useCallback(async () => {
    setStartDialogOpen(false);
    setRunError("");
    await refreshRuns();
  }, [refreshRuns]);

  const handleEndRun = useCallback(async () => {
    if (!activeRun) return;
    setRunBusy(true);
    setRunError("");
    try {
      await fetchJson(endpoints.runEnd(activeRun.run_id), { method: "POST" });
      await refreshRuns();
    } catch (error) {
      setRunError(error instanceof Error ? error.message : "Could not end the run.");
    } finally {
      setRunBusy(false);
    }
  }, [activeRun, refreshRuns]);

  if (authLoading) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-slate-950 text-sm text-slate-400">
        Checking your session…
      </main>
    );
  }

  if (!isAuthenticated) {
    return <LoginView />;
  }

  const showRunBanner = ["operations", "assistance", "staff"].includes(activeTab);

  return (
    <DashboardLayout
      activeTab={activeTab}
      onTabChange={setActiveTab}
      cameras={cameras}
      status={selectedCamera}
      sidebar={
        activeTab === "operations" ? (
          <OperationsSidebarTabs metrics={metrics} alerts={alerts} />
        ) : null
      }
    >
      {showRunBanner ? (
        <ActiveRunBanner
          activeRun={activeRun}
          externalRun={externalRun}
          cvStatus={cvStatus}
          isHistorical={isHistorical}
          onReturnToCurrentRun={() => runState.setSelectedHistoricalRun(null)}
          onStartRun={() => setStartDialogOpen(true)}
          onEndRun={handleEndRun}
          busy={runBusy}
          error={runError}
        />
      ) : null}

      {activeTab === "settings" ? (
        <AdminSettingsView />
      ) : activeTab === "runs" ? (
        <RunHistoryView
          selectedHistoricalRun={runState.selectedHistoricalRun}
          onSelectHistoricalRun={(run) => {
            runState.setSelectedHistoricalRun(run);
            if (run) setActiveTab("operations");
          }}
          onChanged={refreshRuns}
        />
      ) : activeTab === "operations" ? (
        <section className="grid gap-4">
          <OperationsStatusPills apiOnline={apiOnline} cameras={cameras} status={selectedCamera} />
          {isHistorical ? (
            <div className="rounded-lg border border-cyan-400/30 bg-cyan-400/10 px-4 py-3 text-sm text-cyan-100">
              Viewing historical run{" "}
              <span className="font-mono font-bold">{viewedRunId}</span>. Historical tactical
              replay is unavailable, so the floor map is hidden. The live camera stream below is
              still live, not historical.
            </div>
          ) : null}
          <div className="grid gap-4 2xl:grid-cols-[minmax(420px,0.95fr)_minmax(0,1.05fr)]">
            {isHistorical ? (
              <div className="flex min-h-[280px] items-center justify-center rounded-xl border border-slate-800 bg-slate-900/70 p-6 text-center text-sm text-slate-400">
                Historical tactical replay is unavailable.
              </div>
            ) : (
              <TacticalMap state={tacticalState} cameraId="fused" apiOnline={apiOnline} />
            )}
            <div className="grid content-start gap-4">
              <ZoneCapacityBars zones={zoneStatus} />
              <MetricTrendSparkline points={metricTrend} />
            </div>
          </div>
          <VideoPlayer
            apiOnline={apiOnline}
            cameras={cameras}
            camera={selectedCamera}
            selectedCameraId={selectedCamera?.camera_id || selectedCameraId}
            onSelectCamera={setSelectedCameraId}
          />
        </section>
      ) : activeTab === "staff" ? (
        <StaffReviewView
          runId={viewedRunId}
          isHistorical={isHistorical}
          runReady={runReady}
          hasNoRuns={hasNoRuns}
        />
      ) : (
        <AssistanceView
          cameras={cameras}
          runId={viewedRunId}
          isHistorical={isHistorical}
          runReady={runReady}
          hasNoRuns={hasNoRuns}
        />
      )}

      {startDialogOpen ? (
        <StartRunDialog onClose={() => setStartDialogOpen(false)} onStarted={handleStarted} />
      ) : null}
    </DashboardLayout>
  );
}
