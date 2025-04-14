import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App"; // ✅ App 컴포넌트 가져오기

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <App /> {/* ✅ App 컴포넌트 렌더링 */}
  </React.StrictMode>
);
