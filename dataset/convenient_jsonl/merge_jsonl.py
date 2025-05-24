import json

# 병합할 파일 목록
input_files = ["news_100_2.jsonl", "news_100.jsonl"]
output_file = "news_data_200.jsonl"

# 중복 제거용 딕셔너리 (text를 key로 사용)
unique_texts = {}

total_lines = 0
duplicate_count = 0

# 병합 및 중복 제거
for file_path in input_files:
    print(f"📁 {file_path} 처리 중...")
    try:
        with open(file_path, "r", encoding="utf-8") as infile:
            for line in infile:
                try:
                    data = json.loads(line.strip())
                    text = data.get("text")
                    if text and text not in unique_texts:
                        unique_texts[text] = data
                        total_lines += 1
                    else:
                        duplicate_count += 1
                except json.JSONDecodeError:
                    print(f"⚠️ JSON 형식 오류: {line}")
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")

# 파일로 저장
with open(output_file, "w", encoding="utf-8") as outfile:
    for item in unique_texts.values():
        outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 최종 뉴스 개수: {len(unique_texts)}개")
print(f"🗑️ 중복 제거된 뉴스 수: {duplicate_count}개")
