import { Camera, Clock, Images, ImageOff, UserRound } from "lucide-react";
import { resolveApiUrl } from "../../lib/api.js";
import { formatAgeGroup } from "./ageGroups.js";

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

function viewLabel(value) {
  return String(value || "photo").replaceAll("_", " ");
}

export default function EvacueeCard({ evacuee, onOpen }) {
  const primaryView = evacuee.primary_view;
  const galleryFilled = Number(evacuee.gallery_filled || 0);
  const galleryTotal = Number(evacuee.gallery_total || 5);

  return (
    <article className="overflow-hidden rounded-lg border border-slate-800 bg-slate-900/70">
      <button
        type="button"
        onClick={() => onOpen(evacuee)}
        className="group relative block aspect-[4/5] w-full overflow-hidden bg-slate-950 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300"
        aria-label={`Open photo gallery for evacuee ${evacuee.master_identity_id}`}
      >
        {primaryView ? (
          <img
            src={resolveApiUrl(primaryView.image_url)}
            alt={`${viewLabel(primaryView.view_type)} view of evacuee ${evacuee.master_identity_id}`}
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
            View gallery
          </span>
          <span>{galleryFilled}/{galleryTotal}</span>
        </span>
      </button>

      <div className="grid gap-3 p-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-lg font-black text-white">Evacuee {evacuee.master_identity_id}</div>
            <div className="mt-1 text-xs font-bold uppercase tracking-wide text-slate-500">
              {primaryView ? `${viewLabel(primaryView.view_type)} thumbnail` : "No thumbnail"}
            </div>
          </div>
          <div className="rounded-full border border-cyan-300/30 bg-cyan-300/10 px-3 py-1 text-sm font-bold capitalize text-cyan-100">
            {evacuee.gender || "unknown"}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 rounded-md border border-slate-800 bg-slate-950/60 p-3">
          <div>
            <div className="text-base font-black text-white">{formatAgeGroup(evacuee.age)}</div>
            <div className="text-xs font-bold uppercase tracking-wide text-slate-500">estimated age group</div>
          </div>
          <div>
            <div className="text-sm font-black capitalize text-white">{evacuee.current_status || "unknown"}</div>
            <div className="text-xs font-bold uppercase tracking-wide text-slate-500">latest status</div>
          </div>
        </div>

        <div className="grid gap-2 text-sm text-slate-300">
          <div className="flex items-center gap-2">
            <Camera className="h-4 w-4 text-slate-500" />
            <span>{evacuee.last_camera_id || "Camera not provided"}</span>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-slate-500" />
            <span>{formatTime(evacuee.last_seen_at)}</span>
          </div>
          <div className="flex items-center gap-2">
            <UserRound className="h-4 w-4 text-slate-500" />
            <span>Master ID {evacuee.master_identity_id}</span>
          </div>
        </div>
      </div>
    </article>
  );
}
