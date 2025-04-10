import pandas as pd
import json
import re
import multiprocessing as mp
from bareunpy import Tagger
from tqdm import tqdm  # 진행률 표시 라이브러리

API_KEY = "https://github.com/bareun-nlp/bareunpy"
file_path = '.json'

# JSON 파일 로드
with open(file_path, "r", encoding="utf-8") as file:
    json_data = json.load(file)

def flatten_json(json_obj, parent_key='', sep='_'):
    flattened_dict = {}
    for key, value in json_obj.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened_dict.update(flatten_json(value, new_key, sep))
        elif isinstance(value, list):
            if all(isinstance(i, dict) for i in value):
                for i, item in enumerate(value):
                    flattened_dict.update(flatten_json(item, f"{new_key}_{i}", sep))
            else:
                flattened_dict[new_key] = " ".join(map(str, value))
        else:
            flattened_dict[new_key] = value
    return flattened_dict

# JSON 데이터 평탄화 후 DataFrame 변환
data = json_data['data']
flattened_data = [flatten_json(doc) for doc in data]
df = pd.DataFrame(flattened_data)

# 중복 제거
df = df.drop_duplicates(subset=['doc_id', 'doc_title', 'doc_source', 
                                'doc_published', 'doc_class_code', 
                                'created', 'paragraphs_0_context'])

print(df.info())  # 데이터 확인

# 텍스트 전처리 함수
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"<[^>]*>", "", text)  
    text = re.sub(r"[^\w\sㄱ-ㅎ가-힣]", "", text)  
    text = re.sub(r"\d+", "", text)  
    text = text.lower()  
    return text.strip()

# 전처리 적용
df["doc_title"] = df["doc_title"].apply(clean_text)
df["paragraphs_0_context"] = df["paragraphs_0_context"].apply(clean_text)

# 병렬 실행을 위한 함수
def tokenize_and_stem(text):
    if pd.isna(text):
        return ""
    
    # 각 프로세스에서 `Tagger` 새로 생성
    local_tagger = Tagger(API_KEY, 'localhost')
    
    tokens = local_tagger.morphs(text)
    return " ".join(tokens)

# 멀티프로세싱 + 진행률 출력
def process_with_multiprocessing(df, column_name):
    num_workers = max(1, mp.cpu_count() // 2)  # CPU 절반만 사용
    with mp.get_context("spawn").Pool(num_workers) as pool:
        # tqdm 추가: 진행률 출력
        results = list(tqdm(pool.imap(tokenize_and_stem, df[column_name]), total=len(df), desc=f"Processing {column_name}"))
        df[column_name] = results  # 변환된 데이터 저장
    return df

if __name__ == "__main__":
    df = process_with_multiprocessing(df, "doc_title")
    df = process_with_multiprocessing(df, "paragraphs_0_context")

    print(df.head())  

    # JSON 저장
    save_path_json = "morphs_news_data.json"
    df.to_json(save_path_json, orient="records", force_ascii=False)

    print(f"파일이 저장됨: {save_path_json}")