# !pip install trl datasets


import pandas as pd
import torch

train_df = pd.read_csv("/content/drive/MyDrive/gen_dataset_gen/csv_datasets/train_task.csv")
train_df

import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# 환경 변수 설정
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# 모델명 설정
model_name = "GenerTeam/GENERator-eukaryote-3b-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenizer 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Llama 기반 모델에서 pad_token 설정 필수

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

#  LoRA 설정 (target_modules 변경 가능)
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # 변경 가능
    bias="none",
    task_type="CAUSAL_LM",
)

# PEFT 적용
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

#  CSV 데이터 로드
train_df = pd.read_csv("/content/drive/MyDrive/gen_dataset_gen/csv_datasets/train_task.csv")

#  Hugging Face Dataset 변환
dataset = Dataset.from_pandas(train_df)

#  전처리 함수 (입력은 512bp, 출력은 3000bp까지 잘림)
def preprocess_function(examples):
    inputs = tokenizer(
        examples["input_sequence"], truncation=True, padding="max_length", max_length=512
    )
    labels = tokenizer(
        examples["output_sequence"], truncation=True, padding="max_length", max_length=3000
    )

    # Shifted Labels 적용
    labels["input_ids"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in label]
        for label in labels["input_ids"]
    ]
    inputs["labels"] = labels["input_ids"]
    return inputs

#  데이터셋 토큰화
tokenized_dataset = dataset.map(preprocess_function, batched=True)

#  TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./fine_tuned_generator",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    logging_dir="./logs",
    logging_steps=100,
    num_train_epochs=3,
    save_total_limit=2,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    report_to="none"
)

# SFTTrainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    args=training_args,
)

#  학습 시작
print(" Training 시작!")
trainer.train()

#  모델 저장
model.save_pretrained("./fine_tuned_generator")
tokenizer.save_pretrained("./fine_tuned_generator")





# 1. Hugging Face Hub에 모델 업로드하기
from huggingface_hub import login, HfApi
import os

# Hugging Face Hub에 로그인 (토큰이 필요합니다)
# 토큰은 https://huggingface.co/settings/tokens 에서 생성할 수 있습니다
login(token="hf_")  # 실제 토큰으로 대체하세요

# 모델과 토크나이저 업로드
model_name = "./fine_tuned_generator"
repo_name = "hyoo14/GENERator-eukaryote-3b-base-dna_gen"  # 원하는 리포지토리 이름으로 변경하세요

# 허깅페이스 API 초기화
api = HfApi()

# 리포지토리 생성 (없는 경우)
api.create_repo(
    repo_id=repo_name,
    private=False,  # 원하는 경우 private으로 설정
    exist_ok=True
)

# 모델 업로드
api.upload_folder(
    folder_path=model_name,  # 로컬 모델 폴더 경로
    repo_id=repo_name,
    repo_type="model"
)

print(f"모델이 {repo_name}에 업로드되었습니다.")



repo_name = "hyoo14/GENERator-eukaryote-3b-base-dna_gen"

# 2. 나중에 모델 다운로드하기 (메모리 효율적인 방법)
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 중인 디바이스: {device}")


# 토크나이저 로드 (원본 모델에서)
tokenizer = AutoTokenizer.from_pretrained(
    "GenerTeam/GENERator-eukaryote-3b-base",
    trust_remote_code=True
)

# 메모리 효율적인 방법으로 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    repo_name,  # 업로드한 모델 리포지토리 이름
    device_map="auto",  # GPU 메모리를 효율적으로 사용
    load_in_8bit=False,  # 8비트 양자화로 메모리 사용량 줄임
    trust_remote_code=True
).to(device)

# 추론 함수 정의
def generate_sequence(input_seq, max_length=3000):
    inputs = tokenizer(input_seq, return_tensors="pt").to(device)

    # 모델 추론 설정
    with torch.no_grad():
        # outputs = model.generate(
        #     inputs.input_ids,
        #     max_length=max_length,
        #     num_beams=4,
        #     top_k=50,
        #     top_p=0.95,
        #     do_sample=True,
        #     temperature=0.7,
        #     repetition_penalty=1.1,
        #     pad_token_id=tokenizer.eos_token_id
        # )
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            do_sample=False,         # 샘플링 비활성화
            num_beams=1,             # 빔 서치 비활성화 (빔 크기 1)
            temperature=1.0,         # 기본 온도
            # top_k, top_p 파라미터 제거 (사용하지 않음)
            # repetition_penalty는 선택적으로 유지할 수 있음
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# 테스트 데이터 중 첫 2개 샘플에 대해 추론 실행 및 결과 비교
for i in range(min(2, len(test_df))):
    print(f"\n============= 테스트 샘플 {i+1} =============")
    input_seq = test_df.iloc[i]["input_sequence"]
    target_seq = test_df.iloc[i]["output_sequence"]

    # 입력 시퀀스 출력 (처음 50자만)
    print(f"입력 시퀀스: {input_seq[:50]}...")

    # 추론 실행
    print("추론 중...")
    generated_seq = generate_sequence(input_seq)

    # 결과 비교
    print(f"생성된 시퀀스 (처음 100자): {generated_seq[:100]}...")
    print(f"정답 시퀀스 (처음 100자): {target_seq[:100]}...")

    # 간단한 유사도 측정 (일치하는 문자 비율)
    min_len = min(len(generated_seq), len(target_seq))
    matches = sum(1 for a, b in zip(generated_seq[:min_len], target_seq[:min_len]) if a == b)
    similarity = matches / min_len * 100
    print(f"첫 {min_len}자에 대한 유사도: {similarity:.2f}%")


