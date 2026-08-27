import { KeyRound, LoaderCircle, ShieldCheck, UserPlus, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useAuth } from "../../auth/AuthProvider.jsx";
import { endpoints, fetchJson } from "../../lib/api.js";

function formatTime(value) {
  if (!value) return "Never";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function CreateUserForm({ onCreated }) {
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("staff");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      await fetchJson(endpoints.adminUsers, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username,
          display_name: displayName,
          password,
          role,
        }),
      });
      setUsername("");
      setDisplayName("");
      setPassword("");
      setRole("staff");
      onCreated();
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not create the account.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 shadow-2xl"
    >
      <h3 className="mb-3 flex items-center gap-2 text-sm font-black text-white">
        <UserPlus className="h-4 w-4 text-cyan-300" />
        Create Account
      </h3>
      <div className="grid gap-3 lg:grid-cols-4">
        <div>
          <label htmlFor="new-username" className="mb-1 block text-xs font-bold text-slate-300">
            Username
          </label>
          <input
            id="new-username"
            required
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-300"
          />
        </div>
        <div>
          <label htmlFor="new-display" className="mb-1 block text-xs font-bold text-slate-300">
            Display name
          </label>
          <input
            id="new-display"
            required
            value={displayName}
            onChange={(event) => setDisplayName(event.target.value)}
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-300"
          />
        </div>
        <div>
          <label htmlFor="new-password" className="mb-1 block text-xs font-bold text-slate-300">
            Initial password
          </label>
          <input
            id="new-password"
            type="password"
            autoComplete="new-password"
            required
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-300"
          />
        </div>
        <div>
          <label htmlFor="new-role" className="mb-1 block text-xs font-bold text-slate-300">
            Role
          </label>
          <select
            id="new-role"
            value={role}
            onChange={(event) => setRole(event.target.value)}
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-300"
          >
            <option value="staff">Staff</option>
            <option value="admin">Admin</option>
          </select>
        </div>
      </div>
      <p className="mt-2 text-xs text-slate-500">
        Passwords must be at least 10 characters. Give each person their own named
        account so operator actions stay attributable.
      </p>
      {error ? <p className="mt-2 text-xs text-red-300">{error}</p> : null}
      <button
        type="submit"
        disabled={submitting}
        className="mt-3 inline-flex items-center justify-center gap-2 rounded-lg bg-cyan-300 px-4 py-2 text-sm font-black text-slate-950 transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {submitting ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <UserPlus className="h-4 w-4" />}
        Create Account
      </button>
    </form>
  );
}

function ResetPasswordDialog({ user, onClose, onDone }) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      await fetchJson(endpoints.adminUserPassword(user.id), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      onDone();
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not reset the password.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 px-4">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-sm rounded-xl border border-slate-700 bg-slate-900 p-5 shadow-2xl"
      >
        <div className="mb-3 flex items-start justify-between gap-3">
          <h3 className="flex items-center gap-2 text-sm font-black text-white">
            <KeyRound className="h-4 w-4 text-cyan-300" />
            Reset password for {user.display_name}
          </h3>
          <button type="button" onClick={onClose} aria-label="Close" className="text-slate-400 hover:text-white">
            <X className="h-4 w-4" />
          </button>
        </div>
        <p className="mb-3 text-xs text-slate-400">
          This sets a complete replacement password and signs the account out
          everywhere. Tell the operator their new password in person.
        </p>
        <input
          type="password"
          autoComplete="new-password"
          required
          autoFocus
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          className="mb-3 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-300"
        />
        {error ? <p className="mb-3 text-xs text-red-300">{error}</p> : null}
        <div className="flex gap-2">
          <button
            type="submit"
            disabled={submitting}
            className="flex-1 rounded-lg bg-cyan-300 px-4 py-2 text-sm font-black text-slate-950 transition hover:bg-cyan-200 disabled:opacity-40"
          >
            Replace Password
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-slate-700 px-4 py-2 text-sm font-bold text-slate-300 hover:text-white"
          >
            Cancel
          </button>
        </div>
      </form>
    </div>
  );
}

export default function AdminSettingsView() {
  const { user: currentUser } = useAuth();
  const [users, setUsers] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [resetTarget, setResetTarget] = useState(null);

  const load = useCallback(async () => {
    try {
      const nextUsers = await fetchJson(endpoints.adminUsers);
      setUsers(Array.isArray(nextUsers) ? nextUsers : []);
      setError("");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not load accounts.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function updateUser(user, changes) {
    setError("");
    try {
      await fetchJson(endpoints.adminUser(user.id), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(changes),
      });
      await load();
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not update the account.");
    }
  }

  return (
    <section className="grid gap-4">
      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 shadow-2xl">
        <h2 className="flex items-center gap-2 text-base font-black text-white">
          <ShieldCheck className="h-4 w-4 text-cyan-300" />
          Account Management
        </h2>
        <p className="mt-1 text-sm text-slate-400">
          Accounts are disabled rather than deleted so historical actions stay
          attributable. Operators cannot change their own password yet — that is a
          later feature; use Reset Password until then.
        </p>
      </div>

      <CreateUserForm onCreated={load} />

      {error ? (
        <p role="alert" className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-100">
          {error}
        </p>
      ) : null}

      <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/70 shadow-2xl">
        <table className="w-full min-w-[720px] text-left text-sm">
          <thead className="border-b border-slate-800 text-xs uppercase tracking-wide text-slate-400">
            <tr>
              <th className="px-4 py-3">Username</th>
              <th className="px-4 py-3">Display name</th>
              <th className="px-4 py-3">Role</th>
              <th className="px-4 py-3">State</th>
              <th className="px-4 py-3">Created</th>
              <th className="px-4 py-3">Last login</th>
              <th className="px-4 py-3">Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={7} className="px-4 py-6 text-center text-slate-400">
                  Loading accounts…
                </td>
              </tr>
            ) : (
              users.map((user) => (
                <tr key={user.id} className="border-b border-slate-800/60 last:border-0">
                  <td className="px-4 py-3 font-bold text-white">{user.username}</td>
                  <td className="px-4 py-3 text-slate-300">{user.display_name}</td>
                  <td className="px-4 py-3">
                    <select
                      value={user.role}
                      aria-label={`Role for ${user.username}`}
                      onChange={(event) => updateUser(user, { role: event.target.value })}
                      className="rounded-lg border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-white outline-none focus:border-cyan-300"
                    >
                      <option value="staff">Staff</option>
                      <option value="admin">Admin</option>
                    </select>
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`rounded-full px-2.5 py-1 text-xs font-bold ${
                        user.is_active
                          ? "bg-emerald-500/15 text-emerald-100"
                          : "bg-slate-700/60 text-slate-300"
                      }`}
                    >
                      {user.is_active ? "Active" : "Disabled"}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-xs text-slate-400">{formatTime(user.created_at)}</td>
                  <td className="px-4 py-3 text-xs text-slate-400">{formatTime(user.last_login_at)}</td>
                  <td className="px-4 py-3">
                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={() => setResetTarget(user)}
                        className="rounded-lg border border-slate-700 px-2.5 py-1 text-xs font-bold text-slate-200 hover:border-cyan-300 hover:text-white"
                      >
                        Reset Password
                      </button>
                      <button
                        type="button"
                        onClick={() => updateUser(user, { is_active: !user.is_active })}
                        className="rounded-lg border border-slate-700 px-2.5 py-1 text-xs font-bold text-slate-200 hover:border-amber-300 hover:text-white"
                      >
                        {user.is_active ? "Disable" : "Enable"}
                      </button>
                    </div>
                    {user.id === currentUser?.id ? (
                      <p className="mt-1 text-[11px] text-slate-500">This is your account.</p>
                    ) : null}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {resetTarget ? (
        <ResetPasswordDialog
          user={resetTarget}
          onClose={() => setResetTarget(null)}
          onDone={() => {
            setResetTarget(null);
            load();
          }}
        />
      ) : null}
    </section>
  );
}
