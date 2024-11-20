import json
from pydub import AudioSegment
from speechbrain.inference.diarization import Speech_Emotion_Diarization
import os
import argparse
from collections import Counter

# 모델 불러오기
classifier = Speech_Emotion_Diarization.from_hparams(
    source="speechbrain/emotion-diarization-wavlm-large"
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

    return results

def calculate_emotion_time_percentage(results, total_duration):
    # 감정별 시간 계산
    emotion_time = {emotion: 0 for emotion in set(entry['emotion'] for entry in results)}
    
    # 각 세그먼트에서 감정에 해당하는 시간 누적
    for entry in results:
        emotion_time[entry['emotion']] += entry['end'] - entry['start']
    
    # 전체 시간에 대한 퍼센티지 계산
    emotion_percentage = {emotion: (time / total_duration) * 100 for emotion, time in emotion_time.items()}
    
    return emotion_percentage, emotion_time

def main():
    # 명령줄 인자 처리 (오디오 파일 경로 받기)
    parser = argparse.ArgumentParser(description="Perform emotion analysis on an audio file.")
    parser.add_argument("audio_path", type=str, help="Path to the audio file (WAV format).")
    args = parser.parse_args()

    # 오디오 파일 로드
    audio = AudioSegment.from_wav(args.audio_path)
    total_duration = len(audio) / 1000  # 전체 시간 (초 단위)

    # 감정 분석 수행
    results = analyze_emotion_segments(args.audio_path)

    # 감정별 시간과 퍼센티지 계산
    emotion_percentage, emotion_time = calculate_emotion_time_percentage(results, total_duration)

    # 전체 분석 결과 저장할 딕셔너리
    emotion_analysis_results = {
        'audio_path': args.audio_path,
        'emotion_percentage': emotion_percentage,
        'emotion_time': emotion_time,
        'segments': results
    }

    # 분석 결과 출력
    print(f"Results for {args.audio_path}:")
    print(f"Emotion Time (seconds): {emotion_time}")
    print(f"Emotion Percentages: {emotion_percentage}")
    print("-" * 40)

    # 결과를 JSON 파일로 저장
    result_path = "emotion_analysis_results.json"
    with open(result_path, "w") as json_file:
        json.dump(emotion_analysis_results, json_file, indent=4)

    print(f"Emotion analysis results saved to {result_path}")

if __name__ == "__main__":
    main()
