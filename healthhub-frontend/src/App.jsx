import { useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import AuthGate from './components/AuthGate';
import { UserProvider, useUser } from './context/UserContext';
import { ToastProvider } from './context/ToastContext';

import Dashboard from './pages/Dashboard';
import FitnessAssessment from './pages/FitnessAssessment';
import NutritionChat from './pages/NutritionChat';
import SkinAnalysis from './pages/SkinAnalysis';
import ProgressTracking from './pages/ProgressTracking';
import Profile from './pages/Profile';

function AppShell() {
  const [drawerOpen, setDrawerOpen] = useState(false);

  return (
    <div className="app-shell">
      <Sidebar open={drawerOpen} onNavigate={() => setDrawerOpen(false)} />
      {drawerOpen && <div className="backdrop show" onClick={() => setDrawerOpen(false)} />}
      <div className="app-body">
        <Navbar onHamburger={() => setDrawerOpen((v) => !v)} />
        <main className="app-main">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/fitness" element={<FitnessAssessment />} />
            <Route path="/nutrition" element={<NutritionChat />} />
            <Route path="/skin" element={<SkinAnalysis />} />
            <Route path="/progress" element={<ProgressTracking />} />
            <Route path="/profile" element={<Profile />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

function Root() {
  const { user, loading } = useUser();
  if (loading) return null;              // avoid an auth-gate flash on first paint
  if (!user) return <AuthGate />;        // no profile in context -> create-profile form
  return <AppShell />;
}

export default function App() {
  return (
    <UserProvider>
      <ToastProvider>
        <BrowserRouter>
          <Root />
        </BrowserRouter>
      </ToastProvider>
    </UserProvider>
  );
}
