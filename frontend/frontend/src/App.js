import { useState } from "react";

function App() {
  const [news, setNews] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const analyzeNews = async () => {
    if (!news.trim()) {
      setErrorMsg("⚠️ 뉴스 기사를 입력해 주세요!");
      return;
    }

    setLoading(true);
    setErrorMsg("");
    setResult(null);

    try {
      const response = await fetch("http://localhost:5000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: news }),
      });
      const data = await response.json();
      setResult(data.result);
    } catch (error) {
      console.error("에러 발생:", error);
      setErrorMsg("서버와 연결할 수 없습니다. 다시 시도해 주세요.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "30px", textAlign: "center", fontFamily: "Arial" }}>
      <h1 style={{ marginBottom: "10px" }}>📰 가짜 뉴스 탐지 시스템</h1>
      <p style={{ color: "#666" }}>뉴스 기사를 입력하면 신뢰도를 분석해드립니다.</p>

      <textarea
        value={news}
        onChange={(e) => setNews(e.target.value)}
        placeholder="뉴스 기사를 입력하세요..."
        rows="6"
        style={{
          width: "80%",
          padding: "12px",
          fontSize: "16px",
          marginTop: "20px",
          border: "1px solid #ccc",
          borderRadius: "8px",
          resize: "none",
        }}
      ></textarea>

      <br />

      <button
        onClick={analyzeNews}
        style={{
          marginTop: "15px",
          padding: "12px 25px",
          fontSize: "16px",
          cursor: "pointer",
          backgroundColor: "#007bff",
          color: "white",
          border: "none",
          borderRadius: "8px",
          transition: "background-color 0.3s",
        }}
        onMouseOver={(e) => (e.target.style.backgroundColor = "#0056b3")}
        onMouseOut={(e) => (e.target.style.backgroundColor = "#007bff")}
      >
        분석하기
      </button>

      {/* 에러 메시지 표시 */}
      {errorMsg && (
        <div style={{ color: "red", marginTop: "15px", fontWeight: "bold" }}>
          {errorMsg}
        </div>
      )}

      {/* 로딩 중 표시 */}
      {loading && (
        <div style={{ marginTop: "20px", fontSize: "18px", color: "#555" }}>
          🔄 분석 중입니다...
        </div>
      )}

      {/* 결과 표시 */}
      {result && !loading && (
        <div
          style={{
            marginTop: "25px",
            padding: "20px",
            display: "inline-block",
            backgroundColor: result === "FAKE" ? "#ffe6e6" : "#e0ffe6",
            color: result === "FAKE" ? "#d9534f" : "#5cb85c",
            border: "2px solid",
            borderColor: result === "FAKE" ? "#d9534f" : "#5cb85c",
            borderRadius: "12px",
            fontSize: "20px",
            fontWeight: "bold",
            boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
            transition: "all 0.3s ease",
          }}
        >
          {result === "FAKE" ? "❌ 가짜 뉴스입니다!" : "✅ 진짜 뉴스입니다!"}
        </div>
      )}
    </div>
  );
}

export default App;
