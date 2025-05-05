# Qwen3 on SageMaker

이 프로젝트는 Qwen3 모델을 QLoRA 방식으로 파인튜닝하고, 추론까지 실습할 수 있는 전체 워크플로우 예시를 제공합니다.

## 폴더 구조

- `1. Preparation-qwen3.ipynb` : 실습 환경 준비 및 필수 패키지 설치, 모델/토크나이저 다운로드
- `2. Fine-tuning-model-qwen3.ipynb` : QLoRA 기반 Qwen3 모델 파인튜닝 실습
- `3. Serving-model-qwen3.ipynb` : 파인튜닝된 모델의 추론 예시
- `src/`
  - `sm_lora_trainer.py` : QLoRA 기반 파인튜닝 스크립트 (분산 학습, 메모리 최적화 포함)
  - `requirements.txt` : 실험에 필요한 Python 패키지 목록

## 실행 환경

- Python 3.10 이상
- GPU 인스턴스 권장: `ml.g5.2xlarge` (학습 및 추론 시)
- 노트북 환경: Jupyter, SageMaker, Colab 등 어떤 환경이든 무방

## 주요 패키지

- transformers==4.51.3
- datasets==3.5.1
- peft==0.15.2
- trl==0.17.0
- bitsandbytes==0.45.5
- wandb==0.19.10
- tokenizers==0.21.1

## 사용 방법

1. `1. Preparation-qwen3.ipynb` 실행: 환경 및 패키지 설치, 모델/토크나이저 준비
2. `2. Fine-tuning-model-qwen3.ipynb` 실행: 데이터셋 준비 및 QLoRA 기반 Qwen3 파인튜닝
3. `3. Serving-model-qwen3.ipynb` 실행: 파인튜닝된 모델로 추론 테스트

## 참고 자료

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Hugging Face TRL](https://huggingface.co/docs/trl/index)
- [Hugging Face PEFT](https://huggingface.co/docs/peft/index) 