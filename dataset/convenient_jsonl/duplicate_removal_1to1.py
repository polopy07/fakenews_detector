import json

# 두 파일 경로
file1 = "news_balanced_2600.jsonl"
file2 = "news_data_balanced_1to1.jsonl"

# 1. 파일 불러오기
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

data1 = load_jsonl(file1)
data2 = load_jsonl(file2)

# 2. 중복 제거를 위한 set 만들기
texts1 = set(item["text"].strip() for item in data1)
unique_data2 = []

# 3. data2에서 중복되지 않은 항목만 선택
for item in data2:
    text = item["text"].strip()
    if text not in texts1:
        unique_data2.append(item)

print(f"원래 data2 개수: {len(data2)}")
print(f"중복 제거 후 data2 개수: {len(unique_data2)}")

# 4. 결과 저장 (선택 사항)
output_path = "news_balanced_50k.jsonl"
with open(output_path, 'w', encoding='utf-8') as f:
    for item in unique_data2:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"✅ 중복 제거된 데이터가 저장되었습니다: {output_path}")
