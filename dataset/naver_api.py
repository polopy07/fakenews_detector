import requests
import json
import time  # 너무 빠른 요청 방지를 위해 sleep 사용

# 네이버 API 자격 증명
client_id = ''
client_secret = ''

# 검색어 설정
query = '인공지능'
display = 100  # 한 번에 불러올 수 있는 최대값
max_start = 1000  # 1000개까지만 지원
all_results = []

# 헤더 설정
headers = {
    'X-Naver-Client-Id': client_id,
    'X-Naver-Client-Secret': client_secret
}

# 100개 단위로 반복 요청
for start in range(1, max_start, display):
    url = f'https://openapi.naver.com/v1/search/news.json?query={query}&display={display}&start={start}&sort=date'
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        items = response.json().get('items', [])
        all_results.extend(items)
        print(f"{len(items)}개 기사 불러옴 (start={start})")
    else:
        print("요청 실패:", response.status_code)
        print(response.text)
        break

    time.sleep(0.5)  # 너무 빠르게 요청하지 않도록 잠깐 쉬기

# 전체 결과를 JSON 파일로 저장
with open('naver_news_all.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

print(f"\n총 {len(all_results)}개 기사를 저장했습니다.")
