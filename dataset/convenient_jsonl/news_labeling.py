import json

# JSON 파일 불러오기
with open("morphs_news_data_newline.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 모든 뉴스에 라벨 1 추가
for news in data:
    news["label"] = 1

# 변경된 데이터를 다시 JSON으로 저장
with open("morphs_news_data_newline_labeled.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("모든 뉴스에 label=1 추가 완료! 저장됨: morphs_news_data_newline.json")