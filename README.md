# RoBERTa Model
Robustly Optimized BERT approach <br>
## RoBERTa 선정 이유
기존 BERT모델을 유지하며 학습단계의 hyperparameter를 조정하여 성능을 높이는 방법.<br>
BERT는 16GB를 pretraining한 반면, RoBERTa는 160GB를 pretraining함.<br><br>
< BERT 비교 추가된 부분 ><br>
1. dynamic masking
2. NSP 제거
3. 더 긴 시퀀스로 학습
4. 더 많은 데이터 사용하여 더 큰 배치로 학습

BERTbase 의 크기인 L=12, H=768, A=12, 110M params에 맞춰 학습을 진행.



<br><br><br><br><br><br><br><br><br><br><br>
# 참고자료
https://velog.io/@tobigs-nlp/RoBERTa
