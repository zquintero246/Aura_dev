import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import HomePage from './pages/Home';
import Register from './pages/Authentication/Register';
import VerifyEmailPage from './pages/Authentication/VerifyEmailPage';
import Login from './pages/Authentication/Login';
import ChatHome from './pages/Chat/AppLayout';
import RequireVerified from './guards/RequireVerified';
import RedirectIfVerified from './guards/RedirectIfVerified';
import Profile from './pages/Account/Profile';
import ChangePassword from './pages/Account/ChangePassword';

const AppRoutes = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<RedirectIfVerified><HomePage /></RedirectIfVerified>} />
        <Route path="/register" element={<RedirectIfVerified><Register /></RedirectIfVerified>} />
        <Route path="/verify-email" element={<RedirectIfVerified><VerifyEmailPage /></RedirectIfVerified>} />
        <Route path="/login" element={<RedirectIfVerified><Login /></RedirectIfVerified>} />
        <Route path="/chat" element={<RequireVerified><ChatHome /></RequireVerified>} />
        {/* Whiteboard routes removed */}
        <Route path="/profile" element={<RequireVerified><Profile /></RequireVerified>} />
        <Route path="/change-password" element={<RequireVerified><ChangePassword /></RequireVerified>} />
      </Routes>
    </Router>
  );
};

export default AppRoutes;
