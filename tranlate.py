import pandas as pd
import torch
from transformers import MarianTokenizer, MarianMTModel

# GPU 사용 여부 체크
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델과 토크나이저 로드
def load_model():
    model_name = 'Helsinki-NLP/opus-mt-ko-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    return tokenizer, model

# 번역 함수 정의
def translate_text(text, tokenizer, model):
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
df = pd.read_csv('/mount/nas/disk02/Data/Health/Mental_Health/BERT/real/train.csv')
df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"])

# 번역할 열 정의
columns_to_translate = ['sentenceInfo', 'newsSubcategory', 'newsTitle', 'newsSubTitle', 'newsContent']

# 모델과 토크나이저 로드
tokenizer, model = load_model()

# 테스트용으로 10개의 데이터만 샘플링
test_df = df.head(10)

# 각 열에 대해 번역 적용 (10개 샘플만 번역)
for column in columns_to_translate:
    if column in test_df.columns:
        print(f"Translating column: {column}")
        test_df[column] = test_df[column].apply(lambda x: translate_text(x, tokenizer, model))

# 번역 결과 출력
print("\nTranslation Results (First 10 Rows):")
print(test_df[columns_to_translate])

# 테스트 결과 저장
test_save_path = '/mount/nas/disk02/Data/Health/Mental_Health/BERT/real/translated_data_test.csv'
test_df.to_csv(test_save_path, index=False)
print(f"Test DataFrame saved at {test_save_path}")
