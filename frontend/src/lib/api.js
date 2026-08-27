export const API_URL = (import.meta.env.VITE_API_URL || "").trim().replace(/\/$/, "");

function apiUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return API_URL ? `${API_URL}${normalizedPath}` : normalizedPath;
}

export const endpoints = {
  health: apiUrl("/health"),
  status: apiUrl("/api/status"),
  cameras: apiUrl("/api/cameras"),
  stream: apiUrl("/api/stream"),
  cameraStatus: (cameraId) => apiUrl(`/api/cameras/${encodeURIComponent(cameraId)}/status`),
  cameraStream: (cameraId) => apiUrl(`/api/cameras/${encodeURIComponent(cameraId)}/stream`),
  metrics: apiUrl("/api/metrics"),
  metricTrends: apiUrl("/api/metrics/trends"),
  zoneStatus: apiUrl("/api/zones/status"),
  tactical: apiUrl("/api/tactical"),
  tacticalLatest: (cameraId, runId) =>
    withQuery(apiUrl("/api/tactical/latest"), { camera_id: cameraId, run_id: runId }),
  tacticalLatestGlobal: (runId) => withQuery(apiUrl("/api/tactical/latest"), { run_id: runId }),
  alerts: apiUrl("/api/alerts"),
  observations: apiUrl("/api/observations"),
  observationsSummary: apiUrl("/api/observations/summary"),
  evacuees: apiUrl("/api/evacuees"),
  evacueesSummary: apiUrl("/api/evacuees/summary"),
  staff: apiUrl("/api/staff"),
  cvStatus: apiUrl("/api/cv/status"),
  cvStart: apiUrl("/api/cv/session/start"),
  cvStop: apiUrl("/api/cv/session/stop"),
  shiftReportCsv: (runId) => withQuery(apiUrl("/api/reports/shift.csv"), { run_id: runId }),
  shiftReportXlsx: (runId) => withQuery(apiUrl("/api/reports/shift.xlsx"), { run_id: runId }),
  authLogin: apiUrl("/api/auth/login"),
  authLogout: apiUrl("/api/auth/logout"),
  authMe: apiUrl("/api/auth/me"),
  authCsrf: apiUrl("/api/auth/csrf"),
  runs: (params) => withQuery(apiUrl("/api/runs"), params || {}),
  runActive: apiUrl("/api/runs/active"),
  runExternalActive: apiUrl("/api/runs/external-active"),
  runStart: apiUrl("/api/runs/start"),
  runEnd: (runId) => apiUrl(`/api/runs/${encodeURIComponent(runId)}/end`),
  runDelete: (runId) => apiUrl(`/api/runs/${encodeURIComponent(runId)}`),
  adminUsers: apiUrl("/api/admin/users"),
  adminUser: (userId) => apiUrl(`/api/admin/users/${encodeURIComponent(userId)}`),
  adminUserPassword: (userId) =>
    apiUrl(`/api/admin/users/${encodeURIComponent(userId)}/reset-password`),
};

export function resolveApiUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  return apiUrl(path);
}

export function withQuery(url, params = {}) {
  const isAbsoluteUrl = /^https?:\/\//i.test(url);
  const nextUrl = new URL(url, isAbsoluteUrl ? undefined : window.location.origin);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      nextUrl.searchParams.set(key, value);
    }
  });
  return isAbsoluteUrl || API_URL ? nextUrl.toString() : `${nextUrl.pathname}${nextUrl.search}`;
}

/**
 * The CSRF request token lives in memory only, never in the session cookie or
 * browser storage. AuthProvider owns it and registers it here so that every
 * caller goes through one place.
 */
let csrfToken = "";
let onUnauthorized = null;

export function setCsrfToken(token) {
  csrfToken = token || "";
}

export function getCsrfToken() {
  return csrfToken;
}

export function setUnauthorizedHandler(handler) {
  onUnauthorized = handler;
}

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

// A request that never settles would otherwise hold one of the browser's ~6
// connections to this origin forever.
export const DEFAULT_TIMEOUT_MS = 15000;

/**
 * Combine a caller's AbortSignal with an internal timeout, so a request is
 * cancelled either when the caller navigates away or when it stalls.
 */
function withTimeout(signal, timeoutMs) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);

  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  return { signal: controller.signal, done: () => window.clearTimeout(timer) };
}

export async function fetchJson(url, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  const headers = {
    Accept: "application/json",
    ...(options.headers || {}),
  };

  // CSRF belongs only on cookie-authenticated state changes.
  if (!SAFE_METHODS.has(method) && csrfToken) {
    headers["X-CSRF-Token"] = csrfToken;
  }

  const { timeoutMs = DEFAULT_TIMEOUT_MS, signal: callerSignal, ...rest } = options;
  const { signal, done } = withTimeout(callerSignal, timeoutMs);

  let response;
  try {
    response = await fetch(url, {
      ...rest,
      method,
      headers,
      signal,
      // Same-origin in server mode, and an explicitly configured origin in Vite
      // development. Required for the session cookie to travel.
      credentials: "include",
    });
  } finally {
    done();
  }

  if (!response.ok) {
    let detail = "";
    try {
      const payload = await response.json();
      detail = typeof payload?.detail === "string" ? payload.detail : "";
    } catch {
      // The status code remains useful when an upstream proxy returns HTML.
    }
    if (response.status === 401 && typeof onUnauthorized === "function") {
      onUnauthorized();
    }
    throw new ApiError(detail || `Request failed with ${response.status}`, response.status);
  }

  if (response.status === 204) return null;
  return response.json();
}
