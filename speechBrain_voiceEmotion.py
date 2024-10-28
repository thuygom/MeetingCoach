import torchaudio
from speechbrain.pretrained import SpeakerRecognition

# 감정 분석을 위한 미리 학습된 모델 불러오기
emotion_recognizer = SpeakerRecognition.from_hparams(source="speechbrain/emotion-recognition-ecapa-voxceleb", savedir="emotion_recognition_model")

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
    segment = signal[:, start_sample:end_sample]

    # 임시 파일에 저장
    temp_audio_file = f"temp_segment_{idx}.wav"
    torchaudio.save(temp_audio_file, segment, sample_rate)

    # 감정 분석 수행
    emotion_prediction = emotion_recognizer.classify_file(temp_audio_file)
    emotion_results.append((start_time, end_time, emotion_prediction))

# 결과 출력
for start, end, emotion in emotion_results:
    print(f"Segment ({start}s to {end}s): Predicted Emotion: {emotion}")
