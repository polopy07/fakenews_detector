import json
import pandas as pd

# JSON 파일 로드
file_path = "processed_fake_news.json"
with open(file_path, "r", encoding="utf-8") as file:
    json_data = json.load(file)

# JSON을 DataFrame으로 변환
df = pd.DataFrame(json_data)

# 라벨이 1인 개수 확인
fake_news_count = (df["Label"] == 0).sum()
print(f"가짜 뉴스(라벨=0) 개수: {fake_news_count}")

fake_news_count = (df["Label"] == 1).sum()
print(f"가짜 뉴스(라벨=1) 개수: {fake_news_count}")