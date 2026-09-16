import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { getProfile } from '../api/client';

// Replaces every scattered localStorage.getItem('user') read in the old
// vanilla app with one context. There is no password auth — "logged in"
// just means a profile name is remembered on this device.
const STORAGE_KEY = 'healthhub_user';
const UserContext = createContext(null);

function readStoredUser() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function UserProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // load on mount
  useEffect(() => {
    setUser(readStoredUser());
    setLoading(false);
  }, []);

  // login()/register() — store {name, email, ...} and make it the active profile
  const login = useCallback((profile) => {
    if (!profile || !profile.name) return;
    const value = { name: profile.name.trim(), email: profile.email || '' };
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(value)); } catch { /* ignore */ }
    setUser(value);
  }, []);

  const logout = useCallback(() => {
    try {
      Object.keys(localStorage)
        .filter((k) => k === STORAGE_KEY || k.startsWith('healthhub_'))
        .forEach((k) => localStorage.removeItem(k));
    } catch { /* ignore */ }
    try { sessionStorage.clear(); } catch { /* ignore */ }
    setUser(null);
  }, []);

  // re-sync the display name/email with the backend without ever inventing one
  const refresh = useCallback(async () => {
    if (!user?.name) return;
    try {
      const { data } = await getProfile(user.name);
      if (data && data.name) login({ name: data.name, email: data.email });
    } catch { /* offline — keep what's stored */ }
  }, [user?.name, login]);

  return (
    <UserContext.Provider value={{ user, loading, login, logout, refresh }}>
      {children}
    </UserContext.Provider>
  );
}

export function useUser() {
  const ctx = useContext(UserContext);
  if (!ctx) throw new Error('useUser must be used inside <UserProvider>');
  return ctx;
}
