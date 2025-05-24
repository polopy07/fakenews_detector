import pandas as pd

# 기존 JSON 파일 불러오기
file_path = "morphs_news_data.json"  # 기존에 저장한 파일
df = pd.read_json(file_path)

# JSON을 줄바꿈 있는 형식으로 다시 저장
save_path_json = "morphs_news_data_newline.json"
df.to_json(save_path_json, orient="records", force_ascii=False, indent=4)

print(f"새로운 JSON 파일 저장 완료: {save_path_json}")