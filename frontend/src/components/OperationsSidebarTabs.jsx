import { AlertTriangle, BarChart3 } from "lucide-react";
import { useState } from "react";
import AlertsPanel from "./AlertsPanel.jsx";
import MetricsPanel from "./MetricsPanel.jsx";

const tabs = [
  { id: "metrics", label: "Latest Metrics", icon: BarChart3 },
  { id: "alerts", label: "Latest Alerts", icon: AlertTriangle },
];

export default function OperationsSidebarTabs({ metrics, alerts }) {
  const [activeTab, setActiveTab] = useState("metrics");

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/70 p-3">
      <div className="mb-3 grid grid-cols-2 gap-2 rounded-lg bg-slate-950 p-1">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const selected = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`inline-flex items-center justify-center gap-2 rounded-md px-3 py-2 text-xs font-bold transition ${
                selected
                  ? "bg-cyan-300/15 text-cyan-100 ring-1 ring-cyan-300/60"
                  : "text-slate-400 hover:bg-slate-800 hover:text-white"
              }`}
            >
              <Icon className="h-4 w-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      <div className="max-h-[calc(100vh-260px)] overflow-y-auto pr-1">
        {activeTab === "metrics" ? <MetricsPanel metrics={metrics} compact /> : <AlertsPanel alerts={alerts} compact />}
      </div>
    </section>
  );
}
