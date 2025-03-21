# Gemma3ForCausalLM(
#   (model): Gemma3TextModel(
#     (embed_tokens): Gemma3TextScaledWordEmbedding(262144, 1152, padding_idx=0)
#     (layers): ModuleList(
#       (0-25): 26 x Gemma3DecoderLayer(
#         (self_attn): Gemma3Attention(
#           (q_proj): Linear(in_features=1152, out_features=1024, bias=False)
#           (k_proj): Linear(in_features=1152, out_features=256, bias=False)
#           (v_proj): Linear(in_features=1152, out_features=256, bias=False)
#           (o_proj): Linear(in_features=1024, out_features=1152, bias=False)
#           (q_norm): Gemma3RMSNorm((256,), eps=1e-06)
#           (k_norm): Gemma3RMSNorm((256,), eps=1e-06)
#         )
#         (mlp): Gemma3MLP(
#           (gate_proj): Linear(in_features=1152, out_features=6912, bias=False)
#           (up_proj): Linear(in_features=1152, out_features=6912, bias=False)
#           (down_proj): Linear(in_features=6912, out_features=1152, bias=False)
#           (act_fn): PytorchGELUTanh()
#         )
#         (input_layernorm): Gemma3RMSNorm((1152,), eps=1e-06)
#         (post_attention_layernorm): Gemma3RMSNorm((1152,), eps=1e-06)
#         (pre_feedforward_layernorm): Gemma3RMSNorm((1152,), eps=1e-06)
#         (post_feedforward_layernorm): Gemma3RMSNorm((1152,), eps=1e-06)
#       )
#     )
#     (norm): Gemma3RMSNorm((1152,), eps=1e-06)
#     (rotary_emb): Gemma3RotaryEmbedding()
#     (rotary_emb_local): Gemma3RotaryEmbedding()
#   )
#   (lm_head): Linear(in_features=1152, out_features=262144, bias=False)
# )




# !pip install git+https://github.com/huggingface/transformers.git --upgrade


# !pip install -q peft accelerate trl datasets scipy bitsandbytes


import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import pandas as pd
from transformers import Gemma3ForCausalLM  # Llama 대신 Gemma 사용

# Hugging Face에서 Gemma 3.1B 모델 불러오기
model_name = "google/gemma-3-1b-pt"

# 모델 및 토크나이저 로드
max_seq_length = 4096  # 필요에 따라 변경 가능
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token  # PAD 토큰 설정
tokenizer.padding_side = "right"
tokenizer.model_max_length = max_seq_length

model = Gemma3ForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  #  float32 사용#torch_dtype=torch.bfloat16,  # bfloat16 사용
    device_map="auto",
    token=HF_TOKEN
)


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "o_proj"] #"v_proj"]  #  target_modules 명시적으로 설정
)

# LoRA 적용
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()



# CSV 데이터 로드 및 가공
train_df = pd.read_csv("/content/drive/MyDrive/gen_dataset_gen/csv_datasets/train_task.csv")
test_df = pd.read_csv("/content/drive/MyDrive/gen_dataset_gen/csv_datasets/test_task.csv")

train_df["text"] = "Input: " + train_df["input_sequence"] + " Output: " + train_df["output_sequence"].apply(lambda x: x[:3000])
test_df["text"] = "Input: " + test_df["input_sequence"] + " Output: "

train_dataset = Dataset.from_pandas(train_df[["text"]])
test_dataset = Dataset.from_pandas(test_df[["text"]])

# 데이터 토큰화 함수
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_seq_length)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)





# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./gemma3_output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    weight_decay=0.01,
    num_train_epochs=3,
    logging_dir="./logs",
    push_to_hub=False,
    save_total_limit=2,
    fp16=False,  #  float16 대신 float32 사용
    bf16=False,  #  bfloat16 비활성화
    optim="paged_adamw_32bit",
    report_to="none"
)


# SFTTrainer 설정 및 학습 시작
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    args=training_args
)

print("Training 시작!")
trainer.train()



from huggingface_hub import notebook_login

import os
os.environ["HF_TOKEN"] = HF_TOKEN#"hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

from huggingface_hub import login
login(os.getenv("HF_TOKEN"))  # 이 방법이 공식적으로 지원됨!


from huggingface_hub import HfApi

api = HfApi()
user_info = api.whoami()
print(f"Hugging Face 로그인 성공! 계정: {user_info['name']}")


# 이미 저장소에 저장된 경우
from huggingface_hub import HfApi

repo_name = "gemma-3-1b-pt-dna_gen"  # 원하는 저장소 이름
api = HfApi()
api.create_repo(repo_name, private=False)  # private=True 하면 비공개 레포!


from transformers import AutoModelForCausalLM, AutoTokenizer

repo_name = "hyoo14/gemma-3-1b-pt-dna_gen"  # 내 원본 모델 이름

#  모델 저장 (Colab 기준)
save_directory = "/content/gemma3_finetuned"

# 모델 저장
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# Hugging Face 업로드
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)



from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Gemma3ForCausalLM

repo_name = "hyoo14/gemma-3-1b-pt-dna_gen"  # 업로드한 Hugging Face 모델

model_hf = Gemma3ForCausalLM.from_pretrained(repo_name)
tokenizer_hf = AutoTokenizer.from_pretrained(repo_name)


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_hf.to(device)


import torch

# 테스트 데이터에서 2개 샘플 선택
sample_inputs = test_df["input_sequence"][:2].tolist()
sample_outputs = test_df["output_sequence"][:2].tolist()  # 정답 비교용

# 프롬프트 형식 맞추기 (모델 입력 형식)
formatted_inputs = ["Input: " + inp + " Output: " for inp in sample_inputs]


# 토크나이징
inputs = tokenizer_hf(formatted_inputs, return_tensors="pt", padding=True, truncation=True, max_length=4096).to("cuda")

# Greedy Decoding 방식으로 생성
with torch.no_grad():
    outputs = model_hf.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=3000,  # 출력 최대 길이
        #temperature=0.0,  # greedy decoding (무작위성 없음)
        do_sample=False  # 샘플링 비활성화 (항상 같은 출력)
    )

# 디코딩
generated_texts = tokenizer_hf.batch_decode(outputs, skip_special_tokens=True)

# 출력 결과 비교
for i in range(2):
    print(f"\n **Example {i+1}**")
    print(f"**Input:** {sample_inputs[i]}")
    print(f" **Generated Output:** {generated_texts[i]}")
    print(f" **Expected Output:** {sample_outputs[i]}")


# !pip install Levenshtein

model_and_task = "gemma_ft"




import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein

# 유사도 및 GC Content 계산 함수
def jaccard_similarity(seq1, seq2):
    set1, set2 = set(seq1), set(seq2)
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

def cosine_similarity_custom(seq1, seq2):
    vectorizer = CountVectorizer(analyzer='char').fit([seq1, seq2])
    vectors = vectorizer.transform([seq1, seq2]).toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

def levenshtein_similarity(seq1, seq2):
    return 1 - (Levenshtein.distance(seq1, seq2) / max(len(seq1), len(seq2)))

def gc_content(sequence):
    return (sequence.count("G") + sequence.count("C")) / len(sequence) if sequence else 0

# 결과 저장을 위한 리스트
results = []

#  모든 test_df 데이터에 대해 실행
batch_size = 2  # 한 번에 처리할 개수 (조정 가능)
for i in range(0, len(test_df), batch_size):
#for i in range(0, 2, batch_size):
    batch = test_df.iloc[i : i + batch_size]  # batch 단위로 데이터 처리

    sample_inputs = batch["input_sequence"].tolist()
    sample_outputs = batch["output_sequence"].tolist()  # 정답 비교용

    # 프롬프트 형식 맞추기
    formatted_inputs = ["Input: " + inp + " Output: " for inp in sample_inputs]

    #  입력 시퀀스 토큰화 및 GPU로 이동
    inputs = tokenizer_hf(
        formatted_inputs, return_tensors="pt", padding=True, truncation=True, max_length=4096
    ).to("cuda")

    #  DNA 시퀀스 생성
    with torch.no_grad():
        outputs = model_hf.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=3000,  # 출력 최대 길이
            do_sample=False  # Greedy decoding (샘플링 비활성화)
        )

    #  생성된 시퀀스를 디코딩
    generated_texts = tokenizer_hf.batch_decode(outputs, skip_special_tokens=True)

    #  유사도 및 GC Content 계산
    for j in range(len(sample_inputs)):
        generated_seq = generated_texts[j][:3000]  # 3000bp 제한
        expected_seq = sample_outputs[j][:3000]

        jaccard = jaccard_similarity(generated_seq, expected_seq)
        cosine = cosine_similarity_custom(generated_seq, expected_seq)
        levenshtein = levenshtein_similarity(generated_seq, expected_seq)
        gc_generated = gc_content(generated_seq)
        gc_expected = gc_content(expected_seq)

        #  결과 저장
        results.append([
            sample_inputs[j], generated_seq, expected_seq, jaccard, cosine, levenshtein, gc_generated, gc_expected
        ])

    # 진행 상태 출력
    print(f"Processed {min(i + batch_size, len(test_df))}/{len(test_df)} sequences")

#  DataFrame 생성 및 CSV 저장
df_result = pd.DataFrame(results, columns=[
    "input_sequence", "generated_sequence", "expected_output",
    "jaccard_similarity", "cosine_similarity", "levenshtein_similarity",
    "gc_generated", "gc_expected"
])

# Google Drive에 저장
csv_path = f"/content/drive/MyDrive/gen_dataset_gen/results/{model_and_task}_generated_sequences_greedy_all.csv"
df_result.to_csv(csv_path, index=False)

#  평균 유사도 출력
mean_jaccard = df_result["jaccard_similarity"].mean()
mean_cosine = df_result["cosine_similarity"].mean()
mean_levenshtein = df_result["levenshtein_similarity"].mean()
print(f"\n Jaccard Similarity Mean: {mean_jaccard:.4f}")
print(f" Cosine Similarity Mean: {mean_cosine:.4f}")
print(f" Levenshtein Similarity Mean: {mean_levenshtein:.4f}")

#   GC content 상관관계 분석
correlation = df_result[["gc_generated", "gc_expected"]].corr()
print("\n GC Content Correlation:")
print(correlation)

# 결과 저장 완료 메시지
print(f"\n Results saved to: {csv_path}")

