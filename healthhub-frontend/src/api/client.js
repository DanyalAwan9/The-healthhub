import axios from 'axios';

// Same FastAPI backend as the vanilla frontend (backend/api.py) — unchanged.
// Override with VITE_API_URL in a .env file if the API runs elsewhere.
export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export const api = axios.create({ baseURL: API_URL, timeout: 15000 });

const withUser = (path, user) => (user ? `${path}?user=${encodeURIComponent(user)}` : path);

// ---- profile -----------------------------------------------------------
export const getProfile = (user) => api.get(withUser('/profile', user));
export const saveProfile = (data) => api.post('/profile', data);

// ---- dashboard -----------------------------------------------------------
export const getDashboard = (user) => api.get(withUser('/dashboard', user));

// ---- fitness ---------------------------------------------------------------
// dynamic=true (default) asks for a Gemini-personalised workout that actually
// uses gym_access (equipment-aware); the backend validates it and falls back
// to a static routine on any failure/timeout. Gemini can take up to ~25s on a
// cold (uncached) profile, so this call gets a longer timeout than the 15s
// default - identical profiles are served from a 7-day cache after the first call.
export const fitnessAssess = (data, dynamic = true) =>
  api.post(`/fitness/assess?dynamic=${dynamic ? 1 : 0}`, data, { timeout: 35000 });

// ---- nutrition -------------------------------------------------------------
export const getChatHistory = (user) => api.get(withUser('/nutrition/history', user));
// Gemini calls on the backend are capped at 25s (GEMINI_TIMEOUT) - give this a
// longer client-side timeout so a slow-but-successful reply isn't cut off early.
export const nutritionChat = (message, user) =>
  api.post('/nutrition/chat', { message, user }, { timeout: 30000 });
export const clearChat = (user) => api.post(withUser('/nutrition/clear', user), {});

// ---- skin --------------------------------------------------------------------
// `image` is a base64 data URL (FileReader.readAsDataURL) — the backend accepts
// either multipart or {image: "<base64>"} JSON; this uses the JSON form.
export const skinAnalyze = (image) => api.post('/skin/analyze', { image });

// ---- progress --------------------------------------------------------------
export const getProgress = (user) => api.get(withUser('/progress/entries', user));
export const addProgress = (data, user) => api.post('/progress/entries', { ...data, user });

// ---- health ------------------------------------------------------------------
export const getHealth = () => api.get('/health');
