import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "../best_model"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # ✅ 한 번만 호출

label_map = {
    0: "📰 진짜 뉴스",
    1: "⚠️ 가짜 뉴스"
}

def predict_fake_news(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # ✅ 안전하게 이동

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = round(float(probs[0][pred]), 4)

    return {
        "label": pred,
        "confidence": confidence,
        "result": label_map[pred]
    }
