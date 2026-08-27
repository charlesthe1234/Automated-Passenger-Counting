import { MapPinned, RadioTower, UsersRound } from "lucide-react";

const VIEW_SIZE = 100;
const TENT_INSET = 8;
const TENT_SIZE = VIEW_SIZE - TENT_INSET * 2;
const TENT_END = TENT_INSET + TENT_SIZE;
const ROLE_COLORS = {
  evacuee: "#ef4444",
  cag: "#facc15",
  scdf: "#f97316",
};
const ANALYZING_COLOR = "#94a3b8";

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function formatAge(seconds) {
  if (seconds === null || seconds === undefined) return "waiting";
  if (seconds < 1) return "live";
  return `${seconds.toFixed(1)}s ago`;
}

function formatMeters(cm) {
  const meters = cm / 100;
  return Number.isInteger(meters) ? `${meters}m` : `${meters.toFixed(1)}m`;
}

function classifyPoint(point, mapSize, outsideContext) {
  if (point?.area === "inside" || point?.area === "outside_visible") {
    return point.area;
  }

  const x = Number(point?.x);
  const y = Number(point?.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  if (x >= 0 && x <= mapSize && y >= 0 && y <= mapSize) return "inside";
  if (
    x >= -outsideContext &&
    x <= mapSize + outsideContext &&
    y >= -outsideContext &&
    y <= mapSize + outsideContext
  ) {
    return "outside_visible";
  }
  return null;
}

function mapAxis(value, mapSize, outsideContext) {
  if (value < 0) {
    if (outsideContext <= 0) return TENT_INSET;
    return clamp(((value + outsideContext) / outsideContext) * TENT_INSET, 0, TENT_INSET);
  }

  if (value > mapSize) {
    if (outsideContext <= 0) return TENT_END;
    return clamp(TENT_END + ((value - mapSize) / outsideContext) * TENT_INSET, TENT_END, VIEW_SIZE);
  }

  return TENT_INSET + (value / mapSize) * TENT_SIZE;
}

function mapPoint(point, mapSize, outsideContext) {
  const x = Number(point?.x);
  const y = Number(point?.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;

  const area = classifyPoint(point, mapSize, outsideContext);
  if (!area) return null;
  const role = ROLE_COLORS[point?.role] ? point.role : null;
  // A role colour is earned only by an explicitly confirmed identity. Treat
  // missing state as unresolved too, so a partial/stale payload can never show
  // the evacuee red while the person is still being analyzed.
  const analyzing = point?.identity_state !== "confirmed" || !role;

  return {
    area,
    x: mapAxis(x, mapSize, outsideContext),
    y: mapAxis(y, mapSize, outsideContext),
    role,
    analyzing,
    color: analyzing || !role ? ANALYZING_COLOR : ROLE_COLORS[role],
    personId: point?.person_id || point?.master_id || null,
  };
}

function countByArea(points, area) {
  return points.filter((point) => point.area === area).length;
}

function safeCount(value, fallback) {
  const count = Number(value);
  return Number.isFinite(count) ? count : fallback;
}

export default function TacticalMap({ state, cameraId, apiOnline }) {
  const hasData = Boolean(state?.has_data);
  const stale = !apiOnline || Boolean(state?.stale);
  const mapSize = Number(state?.map_size_cm) > 0 ? Number(state.map_size_cm) : 300;
  const outsideContext = Number(state?.outside_context_cm) >= 0 ? Number(state.outside_context_cm) : 700;
  const positions = Array.isArray(state?.positions_cm) ? state.positions_cm : [];
  const plottedPositions = positions
    .map((point) => mapPoint(point, mapSize, outsideContext))
    .filter(Boolean);
  const insideCount = safeCount(state?.inside_count, countByArea(plottedPositions, "inside"));
  const outsideVisibleCount = safeCount(
    state?.outside_visible_count,
    countByArea(plottedPositions, "outside_visible")
  );
  const totalVisibleCount = safeCount(state?.total_visible_count, plottedPositions.length);
  const analyzingCount = safeCount(
    state?.analyzing_count,
    plottedPositions.filter((point) => point.analyzing).length,
  );
  const sourceId = state?.camera_id || cameraId || "fused";
  const sourceLabel = sourceId === "fused" ? "fused map" : sourceId;
  const statusLabel = !apiOnline ? "Backend offline" : !hasData ? "Waiting for tactical data" : stale ? "Stale" : "Live";
  const gridMarks = [0.25, 0.5, 0.75].map((ratio) => TENT_INSET + TENT_SIZE * ratio);
  const tentLabel = `${formatMeters(mapSize)} x ${formatMeters(mapSize)} monitored tent`;

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/70 p-3">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-slate-300">
            <MapPinned className="h-4 w-4 text-cyan-300" />
            Tactical Floor Map
          </div>
          <p className="mt-1 text-xs text-slate-500">Live fused X/Y foot-position dots from CV homography</p>
          <div className="mt-2 flex flex-wrap gap-3 text-xs font-semibold text-slate-400">
            <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-red-500" />Evacuee</span>
            <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-yellow-400" />CAG</span>
            <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-orange-500" />SCDF</span>
            <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-slate-400" />Analyzing</span>
            <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full border border-slate-300 bg-transparent" />Hollow = outside</span>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-full border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs font-bold text-slate-300">
            Source: <span className="text-white">{sourceLabel}</span>
          </span>
          <div
            className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-bold ${
              !hasData || stale
                ? "border-amber-500/30 bg-amber-500/10 text-amber-100"
                : "border-emerald-500/30 bg-emerald-500/10 text-emerald-100"
            }`}
          >
            <RadioTower className="h-3.5 w-3.5" />
            {statusLabel} · {formatAge(state?.age_seconds)}
          </div>
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_150px]">
        <div className="relative overflow-hidden rounded-lg border border-slate-800 bg-slate-950 p-3">
          <svg
            viewBox={`0 0 ${VIEW_SIZE} ${VIEW_SIZE}`}
            role="img"
            aria-label={`Global tactical map from ${sourceLabel}`}
            className="aspect-square w-full"
          >
            <rect x="0" y="0" width={VIEW_SIZE} height={VIEW_SIZE} rx="4" fill="#082f49" />
            <rect
              x="1.5"
              y="1.5"
              width={VIEW_SIZE - 3}
              height={VIEW_SIZE - 3}
              rx="4"
              fill="none"
              stroke="#0e7490"
              strokeWidth="0.8"
              strokeDasharray="2 2"
            />
            <text x="50" y="5" textAnchor="middle" fill="#67e8f9" fontSize="3" fontWeight="700">
              Outside visible area
            </text>

            <rect
              x={TENT_INSET}
              y={TENT_INSET}
              width={TENT_SIZE}
              height={TENT_SIZE}
              rx="3.5"
              fill="#eef2ff"
              stroke="#334155"
              strokeWidth="1.2"
            />
            {gridMarks.map((mark) => (
              <g key={mark}>
                <line x1={mark} y1={TENT_INSET} x2={mark} y2={TENT_END} stroke="#cbd5e1" strokeWidth="0.45" />
                <line x1={TENT_INSET} y1={mark} x2={TENT_END} y2={mark} stroke="#cbd5e1" strokeWidth="0.45" />
              </g>
            ))}
            <text x="11" y="15" fill="#475569" fontSize="3.4" fontWeight="800">
              {tentLabel}
            </text>

            {plottedPositions
              .filter((point) => point.area === "outside_visible")
              .map((point, index) => (
                <circle
                  key={`outside-${point.personId || index}-${point.x}-${point.y}`}
                  cx={point.x}
                  cy={point.y}
                  r="2.1"
                  fill="transparent"
                  stroke={point.color}
                  strokeWidth="0.9"
                />
              ))}

            {plottedPositions
              .filter((point) => point.area === "inside")
              .map((point, index) => (
                <circle
                  key={`inside-${point.personId || index}-${point.x}-${point.y}`}
                  cx={point.x}
                  cy={point.y}
                  r="2.1"
                  fill={point.color}
                />
              ))}
          </svg>

          {!hasData ? (
            <div className="absolute inset-3 flex items-center justify-center rounded-lg bg-slate-950/75 px-4 text-center backdrop-blur-sm">
              <p className="max-w-xs text-sm font-semibold text-slate-300">
                Waiting for `/api/tactical` updates from the CV pipeline.
              </p>
            </div>
          ) : null}
        </div>

        <div className="grid grid-cols-2 content-start gap-2 lg:grid-cols-1">
          <div className="rounded-lg border border-slate-800 bg-slate-950 p-2.5">
            <div className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-slate-500">
              <UsersRound className="h-3.5 w-3.5" />
              Visible people
            </div>
            <div className="mt-1 text-2xl font-black text-cyan-300">{totalVisibleCount}</div>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-950 p-2.5">
            <div className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-slate-500">
              <UsersRound className="h-3.5 w-3.5" />
              Evacuees inside
            </div>
            <div className="mt-1 text-2xl font-black text-red-400">{insideCount}</div>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-950 p-2.5">
            <div className="text-xs font-bold uppercase tracking-wide text-slate-500">Evacuees outside</div>
            <div className="mt-1 text-2xl font-black text-red-400">{outsideVisibleCount}</div>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-950 p-2.5">
            <div className="text-xs font-bold uppercase tracking-wide text-slate-500">Analyzing</div>
            <div className="mt-1 text-2xl font-black text-slate-300">{analyzingCount}</div>
          </div>
        </div>
      </div>
    </section>
  );
}
