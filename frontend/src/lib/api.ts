import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000',
  withCredentials: true, // importante para enviar cookies de sesión
  headers: { 'X-Requested-With': 'XMLHttpRequest' },
  xsrfCookieName: 'XSRF-TOKEN', // ayuda si el backend emite esta cookie
  xsrfHeaderName: 'X-XSRF-TOKEN',
});

export default api;
