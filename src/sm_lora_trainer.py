import os
import torch
import torch.distributed as dist
import yaml
import logging
import argparse
import gc
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer

# 토크나이저 병렬 처리 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 랭크 정보 가져오기
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
global_rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))

# 로깅 설정
logging.basicConfig(level=logging.INFO if global_rank == 0 else logging.WARNING)
logger = logging.getLogger(__name__)

def main():
    # 인자 파싱
    parser = argparse.ArgumentParser(description="Qwen3 모델 학습")
    parser.add_argument("--config", type=str, default="/opt/ml/input/data/config/qwen3-4b.yaml")
    args = parser.parse_args()
    
    # 분산 학습 초기화
    is_distributed = world_size > 1
    is_main_process = global_rank == 0
    
    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    # 설정 파일 로드
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # 경로 설정
    model_path = config.get("model_name_or_path", "/opt/ml/input/data/model_weight")
    train_path = os.path.join(
        config.get("train_dataset_path", "/opt/ml/input/data/training"),
        config.get("data", {}).get("train_path", "train_dataset.json")
    )
    output_dir = config.get("output_dir", "/opt/ml/checkpoints")
    
    # 디렉토리 생성
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    
    # 시드 설정
    training_config = config.get("training", {})
    set_seed(training_config.get("seed", 42))
    
    # 정밀도 설정
    use_bf16 = config.get("model", {}).get("use_bf16", True)
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        trust_remote_code=True,
        device_map=None
    )
    model = model.to("cuda")
    model.config.use_cache = False
    
    # 데이터셋 로드 및 전처리
    raw_dataset = load_dataset("json", data_files=train_path, split="train")
    
    def preprocess_function(examples):
        text_field = "text" if "text" in examples else list(examples.keys())[0]
        texts = examples[text_field]
        if isinstance(texts, list):
            texts = [str(item) if not isinstance(item, str) else item for item in texts]
        
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=config.get("data", {}).get("max_seq_length", 2048),
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    train_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=1,
    )
    
    # LoRA 설정 - 더 명확하게 구성
    lora_config = config.get("lora", {})
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.get("lora_r", 64),
        lora_alpha=lora_config.get("lora_alpha", 16),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias="none",
        target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ]),
    )
    
    # LoRA 적용
    if is_main_process:
        logger.info(f"LoRA 설정: r={peft_config.r}, alpha={peft_config.lora_alpha}, dropout={peft_config.lora_dropout}")
        logger.info(f"LoRA 타겟 모듈: {peft_config.target_modules}")
    
    # 모델에 LoRA 적용
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # 학습 가능한 파라미터 정보 출력
    
    # 학습 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 2),
        learning_rate=training_config.get("learning_rate", 2e-4),
        num_train_epochs=training_config.get("num_train_epochs", 1),
        logging_steps=training_config.get("logging_steps", 10),
        warmup_steps=training_config.get("warmup_steps", 10),
        bf16=use_bf16,
        fp16=not use_bf16,
        save_strategy="steps",
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        seed=training_config.get("seed", 42),
        dataloader_num_workers=0,
        group_by_length=True,
        report_to="none",
        ddp_find_unused_parameters=False,
        label_names=["labels"],
    )
    
    try:
        # 메모리 정리
        gc.collect()
        torch.cuda.empty_cache()
        
        # 학습 실행
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            peft_config=None,  # 이미 적용된 모델 사용
        )
        
        # 학습 시작
        if is_main_process:
            logger.info("학습 시작...")
        
        trainer.train()
        
        # 모델 저장
        if is_main_process:
            logger.info("모델 저장 중...")
            # LoRA 어댑터만 저장
            model.save_pretrained(output_dir)
            # 토크나이저 저장
            tokenizer.save_pretrained(output_dir)
            logger.info(f"모델이 {output_dir}에 저장되었습니다.")
    
    finally:
        # 분산 학습 정리
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()