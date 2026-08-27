import { Activity, Camera, MonitorCheck, Server, Video } from "lucide-react";

function pillTone(tone) {
  const tones = {
    good: "border-emerald-500/30 bg-emerald-500/10 text-emerald-100",
    warning: "border-amber-500/30 bg-amber-500/10 text-amber-100",
    critical: "border-red-500/30 bg-red-500/10 text-red-100",
    neutral: "border-slate-700 bg-slate-900/70 text-slate-200",
  };
  return tones[tone] || tones.neutral;
}

function StatusPill({ icon: Icon, label, value, tone = "neutral" }) {
  return (
    <div className={`inline-flex min-h-10 items-center gap-2 rounded-full border px-3 py-2 text-sm font-bold ${pillTone(tone)}`}>
      <Icon className="h-4 w-4" />
      <span className="text-xs uppercase tracking-wide opacity-75">{label}</span>
      <span>{value}</span>
    </div>
  );
}

export default function OperationsStatusPills({ apiOnline, cameras = [], status }) {
  const selectedConnected = Boolean(status?.camera_connected);
  const onlineCameraCount = cameras.filter((camera) => camera.camera_connected).length;
  const cameraCount = cameras.length;
  const resolution = status?.resolution;
  const resolutionText =
    resolution?.width && resolution?.height ? `${resolution.width} x ${resolution.height}` : "Unknown";

  return (
    <section className="flex flex-wrap items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/70 p-3">
      <div className="mr-1 inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-slate-400">
        <Activity className="h-4 w-4 text-cyan-300" />
        System
      </div>
      <StatusPill icon={Server} label="Backend" value={apiOnline ? "Online" : "Offline"} tone={apiOnline ? "good" : "critical"} />
      <StatusPill
        icon={Camera}
        label="Selected"
        value={selectedConnected ? "Connected" : "Disconnected"}
        tone={selectedConnected ? "good" : "warning"}
      />
      <StatusPill
        icon={MonitorCheck}
        label="Cameras"
        value={cameraCount ? `${onlineCameraCount}/${cameraCount}` : "None"}
        tone={onlineCameraCount > 0 ? "good" : "warning"}
      />
      <StatusPill icon={Video} label="Resolution" value={resolutionText} tone="neutral" />
    </section>
  );
}
