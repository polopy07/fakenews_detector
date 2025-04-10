import { useState } from "react";

function App() {
  const [news, setNews] = useState("");
  const [result, setResult] = useState(null);

  const analyzeNews = async () => {
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
    }
  };

  return (
    <div style={{ padding: "20px", textAlign: "center" }}>
      <h1>가짜 뉴스 탐지 시스템</h1>
      <p>뉴스 기사를 입력하면 신뢰도를 분석해드립니다.</p>

      <textarea
        value={news}
        onChange={(e) => setNews(e.target.value)}
        placeholder="뉴스 기사를 입력하세요..."
        rows="5"
        style={{ width: "80%", padding: "10px", fontSize: "16px" }}
      ></textarea>

      <br />

      <button
        onClick={analyzeNews}
        style={{
          marginTop: "10px",
          padding: "10px 20px",
          fontSize: "16px",
          cursor: "pointer",
        }}
      >
        분석하기
      </button>

      {result && (
        <div
          style={{
            marginTop: "20px",
            fontSize: "18px",
            fontWeight: "bold",
            color: result === "FAKE" ? "red" : "green",
            backgroundColor: result === "FAKE" ? "#ffe6e6" : "#e0ffe6",
            padding: "10px",
            borderRadius: "10px",
            display: "inline-block",
          }}
        >
          분석 결과:{" "}
          {result === "FAKE" ? "❌ 가짜 뉴스입니다!" : "✅ 진짜 뉴스입니다!"}
        </div>
      )}
    </div>
  );
}

export default App;
