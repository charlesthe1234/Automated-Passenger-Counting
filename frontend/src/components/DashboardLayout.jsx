import { History, LogOut, MonitorDot, Search, Settings, ShieldCheck, Video } from "lucide-react";
import { useAuth } from "../auth/AuthProvider.jsx";
import ThemeDropdown from "./ThemeDropdown.jsx";

const tabs = [
  { id: "operations", label: "Operations", icon: Video },
  { id: "assistance", label: "Passenger Assistance", icon: Search },
  { id: "staff", label: "Staff Review", icon: ShieldCheck },
];

export default function DashboardLayout({ children, sidebar, status, cameras = [], activeTab, onTabChange }) {
  const { user, isAdmin, logout } = useAuth();
  const onlineCameraCount = cameras.filter((camera) => camera.camera_connected).length;
  const cameraCount = cameras.length;
  const connected = cameraCount ? onlineCameraCount > 0 : Boolean(status?.camera_connected);

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <div className="mx-auto flex min-h-screen max-w-[1500px] flex-col px-5 py-5">
        <header className="mb-5 flex flex-col gap-4 border-b border-slate-800 pb-5 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-cyan-400/25 bg-cyan-400/10 px-3 py-1 text-xs font-bold uppercase tracking-wide text-cyan-200">
              <MonitorDot className="h-3.5 w-3.5" />
              Passenger Monitoring V1.5
            </div>
            <h1 className="text-3xl font-black tracking-tight text-white lg:text-4xl">
              CAG Live Operations Dashboard
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-slate-400">
              FastAPI streaming engine with SQLite-backed metrics, alerts, and assistance logs.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div
              className={`rounded-lg border px-4 py-3 text-sm font-bold ${
                connected
                  ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-100"
                  : "border-amber-500/30 bg-amber-500/10 text-amber-100"
              }`}
            >
              {cameraCount ? `${onlineCameraCount}/${cameraCount} Cameras Online` : "Cameras Loading"}
            </div>
            <div className="rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-right">
              <p className="text-sm font-bold leading-tight text-white">{user?.display_name}</p>
              <p className="text-[11px] uppercase tracking-wide text-slate-400">{user?.role}</p>
            </div>
            <button
              type="button"
              onClick={logout}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2.5 text-sm font-bold text-slate-300 transition hover:border-slate-500 hover:text-white"
            >
              <LogOut className="h-4 w-4" />
              Sign Out
            </button>
            <ThemeDropdown />
          </div>
        </header>

        <nav className="mb-5 flex flex-wrap gap-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const selected = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => onTabChange(tab.id)}
                className={`inline-flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-bold transition ${
                  selected
                    ? "border-cyan-300 bg-cyan-300/15 text-cyan-100"
                    : "border-slate-800 bg-slate-900/70 text-slate-400 hover:border-slate-600 hover:text-white"
                }`}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            );
          })}
          {/* Runs and Settings are administrator areas. Hiding them is a
              usability choice; FastAPI remains the authority. */}
          {isAdmin ? (
            <button
              type="button"
              onClick={() => onTabChange("runs")}
              className={`inline-flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-bold transition ${
                activeTab === "runs"
                  ? "border-cyan-300 bg-cyan-300/15 text-cyan-100"
                  : "border-slate-800 bg-slate-900/70 text-slate-400 hover:border-slate-600 hover:text-white"
              }`}
            >
              <History className="h-4 w-4" />
              Runs
            </button>
          ) : null}
          {isAdmin ? (
            <button
              type="button"
              onClick={() => onTabChange("settings")}
              aria-label="Admin settings"
              className={`ml-auto inline-flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-bold transition ${
                activeTab === "settings"
                  ? "border-cyan-300 bg-cyan-300/15 text-cyan-100"
                  : "border-slate-800 bg-slate-900/70 text-slate-400 hover:border-slate-600 hover:text-white"
              }`}
            >
              <Settings className="h-4 w-4" />
              Settings
            </button>
          ) : null}
        </nav>

        <div className={sidebar ? "grid flex-1 gap-5 xl:grid-cols-[minmax(0,1fr)_360px]" : "flex-1"}>
          <section className="min-w-0">{children}</section>
          {sidebar ? <aside className="grid content-start gap-4">{sidebar}</aside> : null}
        </div>
      </div>
    </main>
  );
}
