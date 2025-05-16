import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score  # 정확도 계산을 위한 라이브러리

# 모델과 토크나이저 로드 
model_path = ""  
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
# JSONL 파일 경로
file_path = ""  # 예: 바탕화면에 있는 전체 뉴스 데이터

# 파일 읽기
with open(file_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# ✅ 기사 내용과 실제 라벨 추출
texts = [item["text"] for item in data]
labels = [item["label"] for item in data]

# ✅ 배치 단위로 분류 (한 번에 여러 개 처리)
batch_size = 8
predictions = []
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    # ✅ 결과 저장
    predictions.extend(preds.tolist())  # 예측 결과를 저장

# ✅ 각 기사별 예측 및 실제 라벨 출력
for i in range(len(texts)):
    print(f"기사 {i + 1}:")
    print(f"텍스트: {texts[i][:200]}...")  # 텍스트의 앞부분만 출력 (길면 모두 출력하지 않음)
    print(f"실제 라벨: {labels[i]}")
    print(f"예측 라벨: {predictions[i]}")
    print("-" * 50)

# ✅ 정확도 계산
accuracy = accuracy_score(labels, predictions)

# ✅ 결과 출력
print(f"모델 정확도: {accuracy * 100:.2f}%")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["진짜뉴스", "가짜뉴스"])
disp.plot()
