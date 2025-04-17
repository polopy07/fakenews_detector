import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App'; // ✅ App 컴포넌트를 불러옴

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />  {/* ✅ App 컴포넌트를 렌더링 */}
  </React.StrictMode>
);

