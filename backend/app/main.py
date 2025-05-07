from fastapi import FastAPI
from pydantic import BaseModel
from .model_loader import predict_fake_news  # ✅ 여기만 수정!

app = FastAPI()

class NewsItem(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "KoBERT fake news API up!"}

@app.post("/predict")
def predict(news: NewsItem):
    result = predict_fake_news(news.text)
    return result
