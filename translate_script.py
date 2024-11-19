import pandas as pd
import torch
import time
import os
from transformers import MarianTokenizer, MarianMTModel

# GPU 사용 여부 체크
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델과 토크나이저 로드
model_name = 'Helsinki-NLP/opus-mt-ko-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

# 번역 함수 정의
def translate_text(text):
    if pd.isna(text):
        return ""
    text = str(text)[:500]  # 500자로 제한
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error translating {text}: {e}")
        return text

# 데이터프레임 읽기
df = pd.read_csv('/mount/nas/disk02/Data/Health/Mental_Health/BERT/real/validation.csv')
df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"])

# 번역할 열 정의
columns_to_translate = ['sentenceInfo', 'newsSubcategory', 'newsTitle', 'newsSubTitle', 'newsContent']

# 데이터 청크 분할 (청크당 1000행)
chunk_size = 1000
chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)] 
output_path = '/mount/nas/disk02/Data/Health/Mental_Health/BERT/real'
partial_save_path = f"{output_path}/translated_data_partial_v.csv"

resume_index = 0

# 중간 저장 함수
def save_partial_results(path, index):
    partial_save_path = f"{path}/translated_data_partial_v.csv"
    pd.concat(chunks[:index + 1]).to_csv(partial_save_path, index=False)
    print(f"Partial results saved at {partial_save_path} (Chunk index: {index})")

# 번역된 샘플 출력 함수
def print_sample_translation(chunk):
    sample = chunk.head(3)
    for column in columns_to_translate:
        if column in sample.columns:
            print(f"\nSample translations from column '{column}':")
            print(sample[column].tolist())

# 번역 작업 시작 (resume_index 이후부터)
total_translated = 0

for chunk_index, chunk in enumerate(chunks):
    # 이미 번역된 청크는 건너뜀
    if chunk_index <= resume_index:
        continue

    print(f"\n[INFO] Translating chunk {chunk_index + 1}/{len(chunks)}")
    chunk_translated_count = 0

    for column in columns_to_translate:
        if column in chunk.columns:
            print(f"  - Translating column '{column}'")
            # 번역 작업 진행
            chunk[column] = chunk[column].apply(translate_text)
            # 현재 열에서 번역된 행 개수 출력
            translated_count = chunk[column].notna().sum()
            chunk_translated_count += translated_count
            print(f"    > Translated {translated_count} rows in column '{column}'")

    # 번역된 샘플 출력
    print_sample_translation(chunk)

    # 현재 청크에서 번역된 행 개수 출력
    total_translated += chunk_translated_count
    print(f"[INFO] Total translated texts so far: {total_translated} (Current chunk translated: {chunk_translated_count})")

    # 일정 주기마다 중간 저장
    if (chunk_index + 1) % 5 == 0 or (chunk_index + 1) == len(chunks):
        save_partial_results(output_path, chunk_index)

# 최종 결과 저장
final_save_path = f"{output_path}/translated_data_final_v.csv"
pd.concat(chunks).to_csv(final_save_path, index=False)
print(f"\n[INFO] Final DataFrame saved at {final_save_path}")
print(f"[INFO] Total translated texts: {total_translated}")
