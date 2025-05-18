import json

# 입력 JSON 파일 열기
with open("0.json", "r", encoding="utf-8") as json_file:
    data = json.load(json_file)  # 리스트 로드

# JSONL 파일로 저장
with open("AIGen_fakenews_100_2.jsonl", "w", encoding="utf-8") as jsonl_file:
    for item in data:
        # 새 형식으로 재구성
        new_item = {
            "label": item["label"],
            "text": item["content"]  # content → text
        }
        jsonl_file.write(json.dumps(new_item, ensure_ascii=False) + "\n")