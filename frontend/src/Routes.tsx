import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import HomePage from './pages/Home';
import Register from './pages/Authentication/Register';
import VeriVerificationCode from './pages/Authentication/VerificationCode';
import Login from './pages/Authentication/Login';
import ChatHome from './pages/Chat/AppLayout';

const AppRoutes = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/register" element={<Register />} />
        <Route path="/verify-code" element={<VeriVerificationCode />} />
        <Route path="/login" element={<Login />} />
        <Route path="/chat" element={<ChatHome />} />
      </Routes>
    </Router>
  );
};

export default AppRoutes;
