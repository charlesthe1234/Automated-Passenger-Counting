import { LoaderCircle, LogIn, MonitorDot, ShieldAlert } from "lucide-react";
import { useState } from "react";
import { useAuth } from "./AuthProvider.jsx";

export default function LoginView() {
  const { login } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    if (submitting) return;
    setSubmitting(true);
    setError("");
    try {
      await login(username, password);
    } catch (nextError) {
      // The backend returns a deliberately generic message so the screen never
      // reveals whether an account exists.
      setError(nextError instanceof Error ? nextError.message : "Sign-in failed.");
      setPassword("");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-950 px-5 py-10 text-white">
      <div className="w-full max-w-md">
        <div className="mb-6 text-center">
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-cyan-400/25 bg-cyan-400/10 px-3 py-1 text-xs font-bold uppercase tracking-wide text-cyan-200">
            <MonitorDot className="h-3.5 w-3.5" />
            Passenger Monitoring V1.5
          </div>
          <h1 className="text-2xl font-black tracking-tight lg:text-3xl">
            CAG Live Operations Dashboard
          </h1>
          <p className="mt-2 text-sm text-slate-400">
            Sign in with your assigned operator account.
          </p>
        </div>

        <form
          onSubmit={handleSubmit}
          className="rounded-xl border border-slate-800 bg-slate-900/70 p-6 shadow-2xl"
        >
          <label htmlFor="username" className="mb-1.5 block text-sm font-bold text-slate-200">
            Username
          </label>
          <input
            id="username"
            name="username"
            type="text"
            autoComplete="username"
            autoFocus
            required
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            className="mb-4 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-300"
          />

          <label htmlFor="password" className="mb-1.5 block text-sm font-bold text-slate-200">
            Password
          </label>
          <input
            id="password"
            name="password"
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="mb-4 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-300"
          />

          {error ? (
            <p
              role="alert"
              className="mb-4 flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-100"
            >
              <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{error}</span>
            </p>
          ) : null}

          <button
            type="submit"
            disabled={submitting}
            className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-cyan-300 px-5 py-3 text-sm font-black text-slate-950 transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {submitting ? (
              <LoaderCircle className="h-4 w-4 animate-spin" />
            ) : (
              <LogIn className="h-4 w-4" />
            )}
            {submitting ? "Signing in…" : "Sign In"}
          </button>
        </form>

        <p className="mt-4 text-center text-xs text-slate-500">
          Passenger evidence is sensitive operational data. Sign out when you leave
          this workstation.
        </p>
      </div>
    </main>
  );
}
