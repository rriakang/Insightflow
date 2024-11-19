import openai

# OpenAI API 키 설정
openai.api_key = ''

# 데이터 업로드
def upload_data(file_path):
    with open(file_path, "rb") as file:
        response = openai.File.create(
            file=file,
            purpose="fine-tune"
        )
    return response['id']

# Fine-Tuning 시작
def start_fine_tuning(file_id):
    response = openai.FineTuningJob.create(
        training_file=file_id,
        model="gpt-3.5-turbo"
    )
    return response['id']

# Fine-Tuning 상태 확인
def check_status(job_id):
    return openai.FineTuningJob.retrieve(id=job_id)

if __name__ == "__main__":
    file_path = "/mount/nas/disk02/Data/Health/Mental_Health/BERT/real/train_title_content_1000.jsonl"
    
    # 데이터 업로드
    file_id = upload_data(file_path)
    print(f"File uploaded with ID: {file_id}")

    # Fine-Tuning 시작
    try:
        fine_tune_id = start_fine_tuning(file_id)
        print(f"Fine-tuning started with ID: {fine_tune_id}")
    except Exception as e:
        print(f"Error starting fine-tuning: {e}")