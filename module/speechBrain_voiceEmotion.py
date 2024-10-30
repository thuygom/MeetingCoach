import json
import torch
from pydub import AudioSegment
from speechbrain.inference.diarization import Speech_Emotion_Diarization
import os

# 모델 불러오기 (GPU 사용 설정 포함)
classifier = Speech_Emotion_Diarization.from_hparams(
    source="speechbrain/emotion-diarization-wavlm-large",
    run_opts={"device": "cuda"}  # GPU 사용 설정
)

# 오디오 파일을 분할하고 감정 분석을 수행할 함수 정의
def analyze_emotion_segments(audio_path, segment_duration=30):
    # 오디오 파일 로드
    audio = AudioSegment.from_wav(audio_path)
    
    # 분할한 감정 분석 결과 저장할 리스트
    results = []

    # 30초 단위로 오디오 분할 및 감정 분석
    for i in range(0, len(audio), segment_duration * 1000):  # pydub은 ms 단위 사용
        segment = audio[i:i + segment_duration * 1000]
        
        # 각 세그먼트의 시간 보정
        base_time = i / 1000  # ms를 초로 변환
        
        # 세그먼트를 분석에 사용 (임시 파일을 현재 작업 디렉토리에 저장)
        temp_segment_path = "../resource/temp_segment.wav"
        segment.export(temp_segment_path, format="wav")
        diary = classifier.diarize_file(temp_segment_path)
        
        # 분석 결과에 시간 보정을 적용하여 저장
        for entry in diary[temp_segment_path]:
            adjusted_entry = {
                'start': entry['start'] + base_time,
                'end': entry['end'] + base_time,
                'emotion': entry['emotion']
            }
            results.append(adjusted_entry)
        
        # GPU 메모리 캐시 정리
        torch.cuda.empty_cache()

    return results

# 오디오 파일 목록
audio_files = ["../resource/audio.wav"]

# 전체 분석 결과 저장할 딕셔너리
emotion_analysis_results = {}

# 각 파일에 대해 감정 분석 수행
for audio_path in audio_files:
    results = analyze_emotion_segments(audio_path)
    emotion_analysis_results[audio_path] = results

    # 분석 결과 출력
    print(f"Results for {audio_path}:")
    for segment in results:
        print(f"Start: {segment['start']}s, End: {segment['end']}s, Emotion: {segment['emotion']}")
    print("-" * 40)

# 결과를 JSON 파일로 저장
with open("../result/emotion_analysis_results.txt", "w") as json_file:
    json.dump(emotion_analysis_results, json_file, indent=4)

print("Emotion analysis results saved to 'emotion_analysis_results.txt'")
