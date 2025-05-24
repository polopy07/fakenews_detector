import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

# 환경 변수에서 HF 토큰 불러오기
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://huggingface.co/olopy/fakenews"  # ← 모델 주소
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

app = FastAPI()

# CORS 설정 (React 프론트엔드용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포시 React 주소만 허용해도 됨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsItem(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Fake news API is running."}

@app.post("/predict")
def predict(news: NewsItem):
    payload = {"inputs": news.text}
    response = requests.post(API_URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        predictions = response.json()
        if isinstance(predictions, list):
            label = predictions[0]["label"]
            score = predictions[0]["score"]
            result = "❌ 가짜 뉴스입니다!" if "fake" in label.lower() else "✅ 진짜 뉴스입니다!"
            return {
                "label": 1 if "fake" in label.lower() else 0,
                "confidence": round(score, 4),
                "result": result,
                "probabilities": {
                    "real": round(score, 4) if "real" in label.lower() else round(1 - score, 4),
                    "fake": round(score, 4) if "fake" in label.lower() else round(1 - score, 4)
                }
            }
        else:
            return {"error": "모델 응답을 이해할 수 없습니다."}
    else:
        return {"error": f"Hugging Face 응답 오류: {response.status_code}"}