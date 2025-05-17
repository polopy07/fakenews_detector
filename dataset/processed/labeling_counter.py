import json
import pandas as pd

# 파일 이름
file_path = "fakenews_100.jsonl"

# JSONL 파일 열기
with open(file_path, "r", encoding="utf-8") as file:
    jsonl_data = [json.loads(line) for line in file]  # 한 줄씩 읽어서 리스트로 변환

# JSON을 DataFrame으로 변환
df = pd.DataFrame(jsonl_data)

# 라벨별 개수 확인
real_news_count = (df["label"] == 0).sum()
fake_news_count = (df["label"] == 1).sum()

print(f"진짜 뉴스(라벨=0) 개수: {real_news_count}")
print(f"가짜 뉴스(라벨=1) 개수: {fake_news_count}")