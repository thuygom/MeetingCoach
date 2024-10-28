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
plt.figure(figsize=(10, 6))
plt.plot(times, rms, label="RMS Energy")
plt.axhline(rms_threshold, color='r', linestyle='--', label="Interruption Threshold")

# 끼어들기 구간 표시
for t, e in interruptions:
    plt.axvline(t, color='orange', linestyle='--', alpha=0.7)
    plt.text(t, e, f"Interrupt {round(t, 2)}s", color='orange', verticalalignment='bottom')

plt.xlabel("Time (s)")
plt.ylabel("RMS Energy")
plt.title("Speech Interruption Detection based on Volume Increase")
plt.legend()
plt.show()

# 결과 출력
print("Detected interruptions at:")
for t, e in interruptions:
    print(f"Time: {round(t, 2)} seconds, RMS Energy: {e:.2f}")
