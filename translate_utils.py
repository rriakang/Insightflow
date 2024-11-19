import torch
from transformers import MarianTokenizer, MarianMTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델과 토크나이저 로드 (여기서 로드할 수도 있음)
model_name = 'Helsinki-NLP/opus-mt-ko-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

# 배치 번역 함수 정의
def batch_translate(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        translated = model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

# 번역 함수
def translate_chunk(chunk, columns_to_translate):
    for column in columns_to_translate:
        if column in chunk.columns:
            texts = chunk[column].astype(str).tolist()
            chunk[column] = batch_translate(texts)
    return chunk
