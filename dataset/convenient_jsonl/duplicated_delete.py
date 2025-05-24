import pandas as pd

# CSV 또는 JSONL 파일 불러오기
# CSV 사용 시
#df = pd.read_csv("your_news_file.csv")

# JSONL 사용 시
df = pd.read_json("news_data_balanced_1to1.jsonl", lines=True)

# 🔸 본문 기준으로 중복 제거 (text가 완전히 같은 경우)
df_unique = df.drop_duplicates(subset=['text'])  # 또는 'text' 컬럼명 사용

# JSONL로 저장
df_unique.to_json("newsdata_1to1_noDuplicated.jsonl", orient='records', lines=True, force_ascii=False)

print(f"중복 제거 완료! 원래 기사 수: {len(df)}, 중복 제거 후: {len(df_unique)}")
