import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import {
  endpoints,
  fetchJson,
  setCsrfToken,
  setUnauthorizedHandler,
} from "../lib/api.js";

const AuthContext = createContext(null);

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === null) {
    throw new Error("useAuth must be used inside AuthProvider.");
  }
  return context;
}

export default function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const applySession = useCallback((payload) => {
    setCsrfToken(payload?.csrf_token || "");
    setUser(payload?.user || null);
  }, []);

  const clearSession = useCallback(() => {
    setCsrfToken("");
    setUser(null);
  }, []);

  // A 401 from any request means the session ended; drop straight back to the
  // login screen rather than leaving stale sensitive data on screen.
  useEffect(() => {
    setUnauthorizedHandler(() => clearSession());
    return () => setUnauthorizedHandler(null);
  }, [clearSession]);

  // Restore an existing session on page load or refresh.
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const payload = await fetchJson(endpoints.authMe);
        if (mounted) applySession(payload);
      } catch {
        if (mounted) clearSession();
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => {
      mounted = false;
    };
  }, [applySession, clearSession]);

  const login = useCallback(
    async (username, password) => {
      const payload = await fetchJson(endpoints.authLogin, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      applySession(payload);
      return payload.user;
    },
    [applySession],
  );

  const logout = useCallback(async () => {
    try {
      await fetchJson(endpoints.authLogout, { method: "POST" });
    } catch {
      // An already-expired session still ends locally.
    } finally {
      clearSession();
    }
  }, [clearSession]);

  const value = useMemo(
    () => ({
      user,
      loading,
      login,
      logout,
      isAuthenticated: Boolean(user),
      isAdmin: user?.role === "admin",
    }),
    [user, loading, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
