import { useState } from "react";

function App() {
  const [news, setNews] = useState("");
  const [result, setResult] = useState(null);
  const [label, setLabel] = useState(null); // ✅ 추가
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [prob, setProb] = useState(null); // ✅ 확률 저장용

  // 🔧 입력 품질 검사 함수 (컴포넌트 외부에 정의)
function getInputQualityScore(text) {
  const cleaned = text.replace(/\s+/g, ""); // 공백 제거
  const words = text.split(/\s+/).filter((w) => w.length > 0);
  const uniqueWords = new Set(words);
  const uniqueChars = new Set(cleaned);

  const length = cleaned.length;
  const wordRatio = uniqueWords.size / words.length;
  const charRatio = uniqueChars.size / cleaned.length;

  const hasMeaninglessPattern = /(ㅋㅋ+|ㅎㅎ+|[a-zA-Z]{12,}|[!@#$%^&*()_+=\-{}[\]:;"'<>,.?/~`\\|]{4,}){2,}/
.test(text);

  // 기준치 조건
  const isTooShort = length < 50;
  const isTooLong = length > 3000;
  const isTooRepetitive =
  (words.length >= 50 && wordRatio < 0.5) || // 길이가 충분할 때만 비율 적용
  (cleaned.length >= 50 && charRatio < 0.2); // 너무 짧은 입력은 무시
  const isGibberish = hasMeaninglessPattern;

  const qualityIssues = [];

  if (isTooShort) qualityIssues.push("⚠️ 글자 수가 너무 적습니다.");
  if (isTooLong) qualityIssues.push("⚠️ 글자 수가 너무 많습니다.");
  if (isTooRepetitive) qualityIssues.push("⚠️ 반복 단어/문자가 과도하게 많습니다.");
  if (isGibberish) qualityIssues.push("⚠️ 의미 없는 패턴(특수문자)이 포함되어 있습니다.");


  return {
    isValid: qualityIssues.length === 0,
    issues: qualityIssues
  };
}

  const analyzeNews = async () => {
    if (!news.trim()) {
      setErrorMsg("⚠️ 뉴스 기사를 입력해 주세요!");
      return;
    }

    setLoading(true);
    setErrorMsg("");
    setResult(null);
    setLabel(null); // 초기화

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: news }),
      });
      const data = await response.json();
      if (data.error) {
      setErrorMsg("⚠️ 문장을 이해할 수 없어 분석할 수 없습니다.");
      setLoading(false);
      setProb(null); // ✅ 확률 초기화 추가
      return;
    }
      setProb(data.probabilities);  // ← 확률 받아서 상태 저장
      setResult(data.result);   // 설명 문구 (ex: "🔴 가짜 뉴스로 판단됨")
      setLabel(data.label);     
    } catch (error) {
      console.error("에러 발생:", error);
      setErrorMsg("서버와 연결할 수 없습니다. 다시 시도해 주세요.");
      setProb(null); // ✅ 확률 초기화 추가
    } finally {
      setLoading(false);
    }
  };

  const quality = errorMsg === "서버와 연결할 수 없습니다. 다시 시도해 주세요."
  ? { isValid: true, issues: [] }
  : getInputQualityScore(news);


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
            placeholder="뉴스 기사를 입력하세요(50 ~ 1500자). . ."
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
          <p 
          style={{ 
            fontSize: "13px", 
            color: "#888", 
            marginTop: "-20px", 
            marginBottom: "20px"
             }}>
          ({news.length} / 3000자)
          </p>

       <button
          onClick={analyzeNews}
          disabled={!quality.isValid} // ✅ 통합 검사 결과에 따라 비활성화
          style={{
            padding: "15px 30px",
            fontSize: "18px",
            cursor: !quality.isValid ? "not-allowed" : "pointer", // ✅ 커서
            backgroundColor: !quality.isValid ? "#ccc" : "#007bff", // ✅ 배경색
            color: "white",
            border: "none",
            borderRadius: "10px",
            transition: "all 0.3s ease",
            width: "100%",
            maxWidth: "200px",
            boxShadow: "0 4px 10px rgba(0, 123, 255, 0.2)",
          }}
          onMouseOver={(e) => {
            if (quality.isValid) {
              e.target.style.backgroundColor = "#0056b3";
              e.target.style.transform = "scale(1.05)";
            }
          }}
          onMouseOut={(e) => {
            if (quality.isValid) {
              e.target.style.backgroundColor = "#007bff";
              e.target.style.transform = "scale(1)";
            }
          }}
        >
          분석하기
        </button>

      {!errorMsg &&
        quality.issues
          .filter(msg => !msg.includes("글자 수")) // ✅ 글자 수 경고는 제외
          .map((msg, i) => (
            <p key={i} style={{ color: "#d9534f", fontSize: "14px", marginBottom: "4px" }}>
              {msg}
            </p>
      ))}

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

        {label !== null && label !== -1 && !loading && (
          <div
            style={{
              marginTop: "30px",
              padding: "25px",
              borderRadius: "12px",
              fontSize: "22px",
              fontWeight: "bold",
              backgroundColor: label === 1 ? "#f8d7da" : "#d4edda",
              color: label === 1 ? "#721c24" : "#155724",
              border: "2px solid",
              borderColor: label === 1 ? "#f5c6cb" : "#c3e6cb",
              boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
              textAlign: "center",
            }}
          >
            {label === 1 ? "❌ 가짜 뉴스입니다!" : "✅ 진짜 뉴스입니다!"}
            <p
              style={{
                marginTop: "12px",
                fontSize: "14px",
                fontWeight: "normal",
                color: label === 1 ? "#a94442" : "#3c763d",
              }}
            >
              {result}
            </p>
          </div>
        )}

        {/* 예측 불가 메시지 (label === -1) */}
        {label === -1 && !loading && (
          <div
            style={{
              marginTop: "30px",
              padding: "20px",
              borderRadius: "10px",
              fontSize: "16px",
              color: "#d9534f",
              backgroundColor: "#f9d6d5",
              border: "1px solid #f5c6cb",
              textAlign: "center",
            }}
          >
             {result}
          </div>
        )}

        {/* 예측 확률 (정상 label만 표시) */}
        {prob && label !== -1 && (
          <div
            style={{
              marginTop: "15px",
              fontSize: "14px",
              color: "#555",
              lineHeight: "1.6",
            }}
          >
            <strong>예측 확률:</strong><br />
            ✅ 진짜 뉴스일 확률: <strong>{(prob.real * 100).toFixed(2)}%</strong><br />
            ❌ 가짜 뉴스일 확률: <strong>{(prob.fake * 100).toFixed(2)}%</strong>
          </div>
        )}
        </div>
      </main>

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

