import { useEffect, useRef } from "react";
import usePageVisible from "./usePageVisible.js";

/**
 * Run an async task on an interval, safely.
 *
 * `setInterval` fires on a fixed clock regardless of whether the previous
 * request finished, so a slow backend used to leave requests stacking up until
 * the browser's per-origin connection budget was exhausted and the UI appeared
 * to hang until a page refresh. This hook fixes that by:
 *
 *  - skipping a tick while the previous run is still in flight;
 *  - scheduling the next run only after the previous one settles;
 *  - aborting the in-flight request when the caller unmounts or its inputs
 *    change, so navigating away releases the connection immediately;
 *  - pausing entirely while the tab is in the background.
 *
 * `task` receives an AbortSignal and should pass it to fetch.
 */
export default function usePoll(task, { intervalMs, enabled = true, deps = [] }) {
  const taskRef = useRef(task);
  taskRef.current = task;

  const visible = usePageVisible();
  const active = enabled && visible;

  useEffect(() => {
    if (!active) return undefined;

    let cancelled = false;
    let timer = null;
    const controller = new AbortController();

    async function run() {
      if (cancelled) return;
      try {
        await taskRef.current(controller.signal);
      } catch (error) {
        // An abort is the expected outcome of navigating away.
        if (error?.name !== "AbortError" && !cancelled) {
          // Errors are surfaced by the task itself; swallow here so one bad
          // response never stops the schedule.
        }
      }
      if (cancelled) return;
      timer = window.setTimeout(run, intervalMs);
    }

    run();

    return () => {
      cancelled = true;
      controller.abort();
      if (timer !== null) window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, intervalMs, ...deps]);
}
