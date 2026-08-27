import { useEffect, useState } from "react";

/**
 * Tracks whether this browser tab is in the foreground.
 *
 * Viewer devices are often left open on a background tab. Without this, every
 * one of them keeps polling and keeps an MJPEG connection open indefinitely.
 */
export default function usePageVisible() {
  const [visible, setVisible] = useState(
    () => typeof document === "undefined" || document.visibilityState !== "hidden",
  );

  useEffect(() => {
    function handleChange() {
      setVisible(document.visibilityState !== "hidden");
    }

    document.addEventListener("visibilitychange", handleChange);
    handleChange();
    return () => document.removeEventListener("visibilitychange", handleChange);
  }, []);

  return visible;
}
