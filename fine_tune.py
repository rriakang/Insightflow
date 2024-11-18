import openai

# OpenAI API 키 설정
openai.api_key = ''

# 데이터 업로드
def upload_data(file_path):
    response = openai.File.create(
        file=open(file_path),
        purpose='fine-tune'
    )
    return response['id']

# Fine-tuning 시작
def start_fine_tuning(training_file_id):
    response = openai.FineTune.create(
        training_file=training_file_id,
        model="gpt-3.5-turbo"
    )
    return response['id']

# 상태 확인
def check_status(fine_tune_id):
    status_response = openai.FineTune.retrieve(id=fine_tune_id)
    return status_response

if __name__ == "__main__":
    # JSONL 파일 경로
    file_path = "train_title_content.jsonl"

    # 1. 데이터 업로드
    file_id = upload_data(file_path)
    print(f"File uploaded with ID: {file_id}")

    # 2. Fine-tuning 시작
    fine_tune_id = start_fine_tuning(file_id)
    print(f"Fine-tuning started with ID: {fine_tune_id}")

    # 3. 상태 확인
    print("Fine-tuning in progress. Use the check_status function to monitor progress.")
