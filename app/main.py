from fastapi import FastAPI
from pydantic import BaseModel
from .model_loader import predict_fake_news  # ✅ 여기만 수정!
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

class NewsItem(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "KoELECTRA fake news API up!"}

@app.post("/predict")
def predict(news: NewsItem):
    result = predict_fake_news(news.text)
    return result

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중엔 * 허용, 배포 시엔 프론트 주소로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
