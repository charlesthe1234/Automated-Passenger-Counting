import { Camera, Clock, Images, ImageOff, UserRound } from "lucide-react";
import { resolveApiUrl } from "../../lib/api.js";

const ROLE_STYLES = {
  cag: "border-yellow-300/40 bg-yellow-400/15 text-yellow-100",
  scdf: "border-orange-400/40 bg-orange-500/15 text-orange-100",
};

function formatTime(value) {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    month: "short",
    day: "numeric",
  });
}

function formatConfidence(value) {
  if (value === null || value === undefined || value === "") return "Confidence unavailable";
  const confidence = Number(value);
  return Number.isFinite(confidence)
    ? `${Math.round(confidence * 100)}% model confidence`
    : "Confidence unavailable";
}

export default function StaffCard({ person, onOpen }) {
  const role = String(person.role || "staff").toLowerCase();
  const primaryView = person.primary_view;
  const galleryFilled = Number(person.gallery_filled || 0);
  const galleryTotal = Number(person.gallery_total || 5);
  const inside = String(person.current_status || "").toLowerCase() === "inside";

  return (
    <article className="overflow-hidden rounded-lg border border-slate-800 bg-slate-900/70">
      <button
        type="button"
        onClick={() => onOpen(person)}
        className="group relative block aspect-[4/5] w-full overflow-hidden bg-slate-950 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300"
        aria-label={`Open evidence gallery for Master ID ${person.master_identity_id}`}
      >
        {primaryView ? (
          <img
            src={resolveApiUrl(primaryView.image_url)}
            alt={`Evidence for Master ID ${person.master_identity_id}`}
            className="h-full w-full object-cover transition duration-200 group-hover:scale-[1.02]"
          />
        ) : (
          <span className="flex h-full flex-col items-center justify-center gap-3 text-slate-500">
            <ImageOff className="h-9 w-9" />
            <span className="text-sm font-bold">Waiting for first view</span>
          </span>
        )}
        <span className="absolute inset-x-3 bottom-3 flex items-center justify-between gap-2 rounded-md border border-white/15 bg-slate-950/85 px-3 py-2 text-xs font-bold text-white backdrop-blur-sm">
          <span className="inline-flex items-center gap-2">
            <Images className="h-4 w-4 text-cyan-300" />
            View evidence
          </span>
          <span>{galleryFilled}/{galleryTotal}</span>
        </span>
      </button>

      <div className="grid gap-3 p-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-lg font-black text-white">
              Master ID {person.master_identity_id}
            </div>
            <div className="mt-1 text-xs font-bold text-slate-500">
              {formatConfidence(person.role_confidence)}
            </div>
          </div>
          <div
            className={`shrink-0 rounded-full border px-3 py-1 text-sm font-black uppercase ${
              ROLE_STYLES[role] || "border-slate-600 bg-slate-700/40 text-slate-200"
            }`}
          >
            Predicted {role}
          </div>
        </div>

        <div className="grid gap-2 rounded-md border border-slate-800 bg-slate-950/60 p-3 text-sm text-slate-300">
          <div className="flex items-center gap-2">
            <span className={`h-2.5 w-2.5 rounded-full ${inside ? "bg-emerald-400" : "bg-slate-500"}`} />
            <span className="font-bold capitalize">{person.current_status || "Unknown status"}</span>
          </div>
          <div className="flex items-center gap-2">
            <Camera className="h-4 w-4 text-slate-500" />
            <span>{person.last_camera_id || "Camera not provided"}</span>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-slate-500" />
            <span>{formatTime(person.last_seen_at)}</span>
          </div>
          <div className="flex items-center gap-2">
            <UserRound className="h-4 w-4 text-slate-500" />
            <span>Model-classified staff record</span>
          </div>
        </div>
      </div>
    </article>
  );
}
