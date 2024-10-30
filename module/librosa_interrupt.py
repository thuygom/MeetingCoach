import librosa
import numpy as np
import matplotlib.pyplot as plt

# 오디오 파일 경로
audio_path = '../resource/audio.wav'

# 1. 오디오 파일 불러오기
y, sr = librosa.load(audio_path, sr=None)  # 원본 샘플링 레이트 유지

# 2. 음성 에너지 계산 (RMS: Root Mean Square)
frame_length = 2048  # 분석할 프레임 길이
hop_length = 512     # 프레임 간격
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

# 3. 시간 축 계산
times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

# 4. 급격한 음량 증가 구간 탐지 (끊기 기준: 평균의 2배 이상)
rms_threshold = np.mean(rms) * 2  # 기준치: 평균 에너지의 2배
interruptions = []
for i in range(1, len(rms)):
    if rms[i] > rms_threshold and rms[i] > rms[i-1] * 1.5:  # 이전 프레임보다 1.5배 이상 증가 시
        interruptions.append((times[i], rms[i]))

# 5. 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(times, rms, label="RMS Energy", color='blue')
plt.axhline(rms_threshold, color='red', linestyle='--', label="Interruption Threshold")

# 끼어들기 구간 표시
for t, e in interruptions:
    plt.axvline(t, color='orange', linestyle='--', alpha=0.5)
    plt.plot(t, e, 'o', color='orange')  # 마커 추가
    plt.text(t, e + max(rms) * 0.05, f"{round(t, 2)}s", color='orange', verticalalignment='bottom', fontsize=8)  # 텍스트 위치 조정

plt.xlabel("Time (s)")
plt.ylabel("RMS Energy")
plt.ylim([0, rms_threshold * 1.5])  # y축 범위 제한
plt.title("Speech Interruption Detection based on Volume Increase")
plt.legend()
plt.tight_layout()  # 레이아웃 조정
plt.show()

# 결과 출력
print("Detected interruptions at:")
for t, e in interruptions:
    print(f"Time: {round(t, 2)} seconds, RMS Energy: {e:.2f}")
