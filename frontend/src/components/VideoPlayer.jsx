import { RadioTower, VideoOff } from "lucide-react";
import { useEffect, useRef } from "react";
import { endpoints } from "../lib/api.js";
import usePageVisible from "../lib/usePageVisible.js";

export default function VideoPlayer({ camera, cameras, selectedCameraId, onSelectCamera, apiOnline }) {
  const connected = Boolean(camera?.camera_connected);
  const showOffline = !apiOnline || !connected;
  const cameraId = camera?.camera_id || selectedCameraId;
  const streamUrl = cameraId ? endpoints.cameraStream(cameraId) : endpoints.stream;
  const offlineText = !apiOnline
    ? "Backend service unavailable"
    : camera?.last_error || "Camera disconnected. Waiting for reconnect.";

  const imageRef = useRef(null);
  const visible = usePageVisible();
  // An MJPEG response never ends, so the browser holds one of its ~6
  // connections to this origin for as long as the element points at it.
  // Removing the element is not enough: the src must be cleared so the browser
  // aborts the request. That also lets FastAPI notice the disconnect and stop
  // the frame generator instead of streaming to nobody.
  const streaming = visible && Boolean(streamUrl);

  useEffect(() => {
    const image = imageRef.current;
    if (!image) return undefined;

    if (streaming) {
      image.src = streamUrl;
    } else {
      image.removeAttribute("src");
    }

    return () => {
      image.removeAttribute("src");
    };
  }, [streamUrl, streaming]);

  return (
    <section className="relative overflow-hidden rounded-lg border border-slate-800 bg-slate-950 shadow-2xl">
      <div className="flex flex-col gap-3 border-b border-slate-800 px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h2 className="text-lg font-bold text-white">Live Camera Stream</h2>
          <p className="text-sm text-slate-400">
            {cameraId ? `${cameraId} native MJPEG feed from FastAPI` : "Waiting for configured cameras"}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {cameras.length > 1
            ? cameras.map((item) => {
                const active = item.camera_id === cameraId;
                const online = Boolean(item.camera_connected);
                return (
                  <button
                    key={item.camera_id}
                    type="button"
                    onClick={() => onSelectCamera(item.camera_id)}
                    className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-bold transition ${
                      active
                        ? "border-cyan-300 bg-cyan-300/15 text-cyan-100"
                        : "border-slate-700 bg-slate-900 text-slate-300 hover:border-slate-500 hover:text-white"
                    }`}
                  >
                    <span className={`h-2 w-2 rounded-full ${online ? "bg-emerald-300" : "bg-amber-300"}`} />
                    {item.camera_id}
                  </button>
                );
              })
            : null}
          <div
            className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-bold ${
              connected ? "bg-emerald-500/15 text-emerald-200" : "bg-amber-500/15 text-amber-200"
            }`}
          >
            <RadioTower className="h-3.5 w-3.5" />
            {connected ? "Live" : "Reconnecting"}
          </div>
        </div>
      </div>

      <div className="relative aspect-video bg-slate-950">
        {/* src is assigned by the effect above so it can be cleared reliably. */}
        <img
          ref={imageRef}
          alt={cameraId ? `${cameraId} live passenger monitoring camera stream` : "Live passenger monitoring stream"}
          className="h-full w-full object-cover"
        />
        {!visible && !showOffline ? (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-950/72 px-6 text-center text-sm text-slate-300 backdrop-blur-sm">
            Stream paused while this tab is in the background.
          </div>
        ) : null}
        {showOffline ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/72 px-6 text-center backdrop-blur-sm">
            <VideoOff className="mb-3 h-10 w-10 text-amber-300" />
            <div className="text-lg font-bold text-white">Stream not connected</div>
            <p className="mt-1 max-w-md text-sm text-slate-300">{offlineText}</p>
          </div>
        ) : null}
      </div>
    </section>
  );
}
