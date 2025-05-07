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
    <div
      style={{
        minHeight: "100vh",
        background: "linear-gradient(to bottom, #f0f4ff, #ffffff)",
        fontFamily: "'Roboto', sans-serif",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* 상단 네비게이션 바 */}
      <header
        style={{
          backgroundColor: "#0056b3",
          padding: "20px 40px",
          color: "white",
          fontSize: "24px",
          fontWeight: "bold",
          boxShadow: "0 2px 10px rgba(0,0,0,0.2)",
          textAlign: "center",
        }}
      >
        🧠 AI 뉴스 진위 판별 시스템
      </header>

      {/* 메인 콘텐츠 */}
      <main
        style={{
          flex: 1,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          padding: "40px 20px",
        }}
      >
        <div
          style={{
            maxWidth: "700px",
            width: "100%",
            backgroundColor: "white",
            padding: "40px",
            borderRadius: "16px",
            boxShadow: "0 6px 20px rgba(0,0,0,0.1)",
            textAlign: "center",
          }}
        >
          <h1 style={{ fontSize: "36px", color: "#333", marginBottom: "20px" }}>
            📰 뉴스 신뢰도 분석
          </h1>
          <p style={{ color: "#666", fontSize: "16px", marginBottom: "30px" }}>
            뉴스 기사 내용을 입력하면 AI가 진짜인지 판별해 드립니다.
          </p>

          <textarea
            value={news}
            onChange={(e) => setNews(e.target.value)}
            placeholder="뉴스 기사를 입력하세요..."
            rows="8"
            style={{
              width: "100%",
              padding: "18px",
              fontSize: "16px",
              border: "1px solid #ccc",
              borderRadius: "10px",
              resize: "none",
              boxSizing: "border-box",
              marginBottom: "30px",
              fontFamily: "'Roboto', sans-serif",
              color: "#333",
              backgroundColor: "#f9f9f9",
            }}
          ></textarea>

          <button
            onClick={analyzeNews}
            style={{
              padding: "15px 30px",
              fontSize: "18px",
              cursor: "pointer",
              backgroundColor: "#007bff",
              color: "white",
              border: "none",
              borderRadius: "10px",
              transition: "all 0.3s ease",
              width: "100%",
              maxWidth: "200px",
              boxShadow: "0 4px 10px rgba(0, 123, 255, 0.2)",
            }}
            onMouseOver={(e) => {
              e.target.style.backgroundColor = "#0056b3";
              e.target.style.transform = "scale(1.05)";
            }}
            onMouseOut={(e) => {
              e.target.style.backgroundColor = "#007bff";
              e.target.style.transform = "scale(1)";
            }}
          >
            분석하기
          </button>

          {errorMsg && (
            <div
              style={{
                color: "#d9534f",
                marginTop: "20px",
                fontWeight: "bold",
                fontSize: "16px",
              }}
            >
              {errorMsg}
            </div>
          )}

          {loading && (
            <div
              style={{
                marginTop: "25px",
                fontSize: "18px",
                color: "#555",
              }}
            >
              🔄 분석 중입니다...
            </div>
          )}

          {result && !loading && (
            <div
              style={{
                marginTop: "30px",
                padding: "25px",
                borderRadius: "12px",
                fontSize: "22px",
                fontWeight: "bold",
                backgroundColor: result === "FAKE" ? "#f8d7da" : "#d4edda",
                color: result === "FAKE" ? "#721c24" : "#155724",
                border: "2px solid",
                borderColor: result === "FAKE" ? "#f5c6cb" : "#c3e6cb",
                boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                textAlign: "center",
              }}
            >
              {result === "FAKE"
                ? "❌ 가짜 뉴스입니다!"
                : "✅ 진짜 뉴스입니다!"}
              <p
                style={{
                  marginTop: "12px",
                  fontSize: "14px",
                  fontWeight: "normal",
                  color: result === "FAKE" ? "#a94442" : "#3c763d",
                }}
              >
                AI가 분석한 결과입니다. 단어 사용, 문장 구조, 패턴 등을 고려했습니다.
              </p>
            </div>
          )}
        </div>
      </main>

      {/* 하단 푸터 */}
      <footer
        style={{
          textAlign: "center",
          padding: "20px",
          color: "#888",
          fontSize: "14px",
          backgroundColor: "#f0f0f0",
        }}
      >
        © 2025 동아대학교 AI 기반 가짜 뉴스 탐지 프로젝트
      </footer>
    </div>
  );
}

export default App;
