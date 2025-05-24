import json
import pandas as pd

# 파일 이름
file_path = "news_data_200.jsonl"

# JSONL 파일 열기
with open(file_path, "r", encoding="utf-8") as file:
    jsonl_data = [json.loads(line) for line in file]  # 한 줄씩 읽어서 리스트로 변환

# JSON을 DataFrame으로 변환
df = pd.DataFrame(jsonl_data)

# 라벨별 개수 확인
real_news_count = (df["label"] == 0).sum()
fake_news_count = (df["label"] == 1).sum()
# 라벨이 0, 1이 아닌 경우 (벡터 연산으로 처리)
strange_news = ((df["label"] != 0) & (df["label"] != 1)).sum()

# 출력
print(f"진짜 뉴스(라벨=0) 개수: {real_news_count}")
print(f"가짜 뉴스(라벨=1) 개수: {fake_news_count}")
print(f"라벨이 0, 1 제외한 이상 라벨 개수: {strange_news}")
print(f"전체 라벨 수 합계: {real_news_count + fake_news_count + strange_news}")
