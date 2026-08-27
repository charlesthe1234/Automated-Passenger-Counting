import { Camera, Clock, ImageOff, Images, ShieldCheck, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { resolveApiUrl } from "../../lib/api.js";
import { formatAgeGroup } from "./ageGroups.js";

const VIEW_SLOTS = [
  { id: "front", label: "Front" },
  { id: "left_side", label: "Left" },
  { id: "right_side", label: "Right" },
  { id: "back", label: "Back" },
  { id: "baseline", label: "Baseline" },
];

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
    year: "numeric",
  });
}

export default function EvacueeGalleryModal({ evacuee, onClose }) {
  const viewsByType = useMemo(
    () => Object.fromEntries((evacuee?.views || []).map((view) => [view.view_type, view])),
    [evacuee],
  );
  const initialView = evacuee?.primary_view?.view_type || evacuee?.views?.[0]?.view_type || "front";
  const [selectedType, setSelectedType] = useState(initialView);
  const selectedView = viewsByType[selectedType] || null;

  useEffect(() => {
    setSelectedType(initialView);
  }, [initialView, evacuee?.id]);

  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKeyDown);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [onClose]);

  if (!evacuee) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/85 p-3 backdrop-blur-sm sm:p-6"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="evacuee-gallery-title"
        className="max-h-[94vh] w-full max-w-5xl overflow-y-auto rounded-lg border border-slate-700 bg-slate-900 shadow-2xl"
      >
        <header className="sticky top-0 z-10 flex items-start justify-between gap-4 border-b border-slate-800 bg-slate-900/95 px-4 py-4 backdrop-blur sm:px-6">
          <div>
            <div className="mb-1 inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-cyan-200">
              <Images className="h-4 w-4" />
              Five-view evidence gallery
            </div>
            <h2 id="evacuee-gallery-title" className="text-2xl font-black text-white">
              Evacuee {evacuee.master_identity_id}
            </h2>
            <p className="mt-1 text-sm text-slate-400">
              {evacuee.gallery_filled}/{evacuee.gallery_total} views captured
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md border border-slate-700 bg-slate-950 text-slate-300 transition hover:border-slate-500 hover:text-white"
            aria-label="Close photo gallery"
            title="Close gallery"
          >
            <X className="h-5 w-5" />
          </button>
        </header>

        <div className="grid gap-5 p-4 sm:p-6 lg:grid-cols-[minmax(0,1fr)_260px]">
          <div className="grid gap-4">
            <div className="aspect-[4/3] overflow-hidden rounded-lg border border-slate-800 bg-slate-950">
              {selectedView ? (
                <img
                  src={resolveApiUrl(selectedView.image_url)}
                  alt={`${selectedType.replaceAll("_", " ")} view of evacuee ${evacuee.master_identity_id}`}
                  className="h-full w-full object-contain"
                />
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-3 px-4 text-center text-slate-500">
                  <ImageOff className="h-12 w-12" />
                  <div>
                    <div className="font-bold text-slate-300">This view has not been captured</div>
                    <div className="mt-1 text-sm">The gallery updates automatically as the person changes direction.</div>
                  </div>
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
              {VIEW_SLOTS.map((slot) => {
                const view = viewsByType[slot.id];
                const active = selectedType === slot.id;
                return (
                  <button
                    key={slot.id}
                    type="button"
                    onClick={() => setSelectedType(slot.id)}
                    className={`overflow-hidden rounded-md border text-left transition ${
                      active
                        ? "border-cyan-300 bg-cyan-300/10"
                        : "border-slate-800 bg-slate-950 hover:border-slate-600"
                    }`}
                    aria-pressed={active}
                  >
                    <span className="block aspect-[4/3] bg-slate-950">
                      {view ? (
                        <img src={resolveApiUrl(view.image_url)} alt="" className="h-full w-full object-cover" />
                      ) : (
                        <span className="flex h-full items-center justify-center text-slate-600">
                          <ImageOff className="h-5 w-5" />
                        </span>
                      )}
                    </span>
                    <span className="flex items-center justify-between gap-1 px-2 py-2 text-xs font-bold text-slate-300">
                      {slot.label}
                      <span className={view ? "text-emerald-300" : "text-slate-600"}>{view ? "Ready" : "Missing"}</span>
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          <aside className="grid content-start gap-3">
            <div className="rounded-lg border border-slate-800 bg-slate-950 p-4">
              <div>
                <div className="text-xs font-bold uppercase tracking-wide text-slate-500">
                  Model-estimated gender
                </div>
                <div className="mt-2 text-2xl font-black capitalize text-cyan-200">
                  {evacuee.gender || "unknown"}
                </div>
              </div>
              <div className="mt-4 border-t border-slate-800 pt-4">
                <div className="text-xs font-bold uppercase tracking-wide text-slate-500">
                  Model-estimated age group
                </div>
                <div className="mt-2 text-2xl font-black text-white">{formatAgeGroup(evacuee.age)}</div>
              </div>
            </div>
            <div className="grid gap-3 rounded-lg border border-slate-800 bg-slate-950 p-4 text-sm text-slate-300">
              <div className="flex items-start gap-2">
                <Camera className="mt-0.5 h-4 w-4 text-slate-500" />
                <div>
                  <div className="font-bold text-white">{selectedView?.camera_id || evacuee.last_camera_id || "Unknown"}</div>
                  <div className="text-xs text-slate-500">capture camera</div>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <Clock className="mt-0.5 h-4 w-4 text-slate-500" />
                <div>
                  <div className="font-bold text-white">{formatTime(selectedView?.captured_at || evacuee.last_seen_at)}</div>
                  <div className="text-xs text-slate-500">capture time</div>
                </div>
              </div>
            </div>
            <div className="rounded-lg border border-amber-400/30 bg-amber-500/10 p-4 text-sm leading-6 text-amber-100">
              <div className="mb-1 flex items-center gap-2 font-bold">
                <ShieldCheck className="h-4 w-4" />
                Manual verification required
              </div>
              Images and demographics are model-generated assistance evidence. They do not establish identity or perform face recognition.
            </div>
          </aside>
        </div>
      </section>
    </div>
  );
}
