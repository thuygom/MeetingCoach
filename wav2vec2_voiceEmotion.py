import torchaudio
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

# 모델 및 프로세서 불러오기
model_name = "m3hrdadfi/wav2vec2-large-xlsr-53-emotion-recognition"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# 음성 파일 경로 설정
audio_file = "path/to/your/audio/file.wav"  # 사용할 음성 파일의 경로로 변경하세요

# 음성 파일 로드
signal, sample_rate = torchaudio.load(audio_file)

# 발언 구간 (시작 시간, 종료 시간) 리스트 예시
# 단위는 초입니다. [(start1, end1), (start2, end2), ...]
speech_segments = [
    (0, 3),   # 첫 번째 발언: 0초부터 3초까지
    (3, 6),   # 두 번째 발언: 3초부터 6초까지
    (6, 10)   # 세 번째 발언: 6초부터 10초까지
]

# 각 발언 구간에 대해 감정 분석 수행
emotion_results = []

for idx, (start_time, end_time) in enumerate(speech_segments):
    # 음성 구간 추출
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    segment = signal[:, start_sample:end_sample].squeeze(0)

    # 모델 입력 형식에 맞게 변환
    input_values = processor(segment.numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True)

    # 감정 분석 수행
    with torch.no_grad():
        logits = model(**input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # 감정 레이블 변환 (필요 시 레이블 매핑을 정의)
    emotions = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]  # 모델에 따라 레이블 수정
    emotion_prediction = emotions[predicted_ids.item()]
    
    emotion_results.append((start_time, end_time, emotion_prediction))

# 결과 출력
for start, end, emotion in emotion_results:
    print(f"Segment ({start}s to {end}s): Predicted Emotion: {emotion}")
