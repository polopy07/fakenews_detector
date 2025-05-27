# KoELECTRA 기반 가짜 뉴스 탐지기 (Fake News Detector)

본 프로젝트는 KoELECTRA 모델을 기반으로 한 **한국어 뉴스 진위 판별 시스템**입니다.  
문장 단위로 긴 기사를 나누어 예측한 뒤, **기사 단위로 종합 판단**하는 방식으로 구성되어 있어 긴 뉴스도 정확하게 처리할 수 있습니다.

## 📌 프로젝트 개요

- **모델**: KoELECTRA (monologg/koelectra-base-discriminator)
- **방식**:
  - 문장 단위로 입력을 분할 (청크 처리)
  - 각 청크별 예측 결과를 가중 평균하여 기사 단위 판단
  - 키워드 기반 점수를 앙상블하여 정밀도 향상
- **예외 처리**:
  - 반복 문자 / 무의미한 입력 감지
  - 너무 짧거나 이상한 입력은 예측 차단

## 기술 스택

- Python 3.10 이상
- FastAPI (백엔드)
- React (프론트엔드)
- Hugging Face Transformers
- Scikit-learn, torch, etc.

---

## 모델 다운로드 및 사용 안내 (Hugging Face)

이 프로젝트는 사전 학습된 KoELECTRA 모델을 기반으로 합니다.  
아래 Hugging Face 저장소에서 모델을 수동으로 다운로드한 뒤, 로컬 경로에 배치하여 사용하세요.

### 모델 다운로드 주소

https://huggingface.co/olopy/fakenews
