import { useState } from "react";

function App() {
  const [news, setNews] = useState(""); // 입력한 뉴스 저장
  const [result, setResult] = useState(null); // 분석 결과 저장

  // 🔹 서버로 뉴스 데이터 보내는 함수
  const analyzeNews = async () => {
    try {
      const response = await fetch("http://localhost:5000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: news }),
      });

      const data = await response.json();
      setResult(data.result); // 분석 결과 저장
    } catch (error) {
      console.error("에러 발생:", error);
    }
  };

  return (
    <div style={{ padding: "20px", textAlign: "center" }}>
      <h1>가짜 뉴스 탐지 시스템</h1>
      <p>뉴스 기사를 입력하면 신뢰도를 분석해드립니다.</p>

      {/* 🔹 뉴스 기사 입력창 */}
      <textarea
        value={news}
        onChange={(e) => setNews(e.target.value)}
        placeholder="뉴스 기사를 입력하세요..."
        rows="5"
        style={{ width: "80%", padding: "10px", fontSize: "16px" }}
      ></textarea>

      <br />

      {/* 🔹 분석 버튼 */}
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

      {/* 🔹 분석 결과 표시 */}
      {result && (
        <div style={{ marginTop: "20px", fontSize: "18px", fontWeight: "bold" }}>
          분석 결과: {result}
        </div>
      )}
    </div>
  );
}

export default App;
