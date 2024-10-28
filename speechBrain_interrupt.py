import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from speechbrain.processing.features import compute_amplitude

# 오디오 파일 경로
audio_path = '../resource/audio.wav'

# 1. 오디오 파일 불러오기
signal, sr = torchaudio.load(audio_path)  # signal: (채널, 샘플)
signal = signal[0]  # 모노 채널 선택

# 2. RMS 에너지 계산
frame_length = int(sr * 0.05)  # 50ms 프레임
hop_length = int(sr * 0.025)   # 25ms 간격

# 3. 프레임별 RMS 에너지 계산
energy = compute_amplitude(signal, frame_length=frame_length, hop_length=hop_length).squeeze().numpy()

# 4. 시간축 생성
times = np.arange(len(energy)) * (hop_length / sr)

# 5. 급격한 음량 증가 구간 탐지 (평균 에너지의 2배 이상인 구간)
energy_threshold = np.mean(energy) * 2
interruptions = []
for i in range(1, len(energy)):
    if energy[i] > energy_threshold and energy[i] > energy[i-1] * 1.5:  # 이전 프레임보다 1.5배 이상 증가
        interruptions.append((times[i], energy[i]))

# 6. 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(times, energy, label="Energy")
plt.axhline(energy_threshold, color='r', linestyle='--', label="Interruption Threshold")

# 끼어들기 구간 표시
for t, e in interruptions:
    plt.axvline(t, color='orange', linestyle='--', alpha=0.7)
    plt.text(t, e, f"Interrupt {round(t, 2)}s", color='orange', verticalalignment='bottom')

plt.xlabel("Time (s)")
plt.ylabel("Energy")
plt.title("Speech Interruption Detection based on Volume Increase")
plt.legend()
plt.show()

# 결과 출력
print("Detected interruptions at:")
for t, e in interruptions:
    print(f"Time: {round(t, 2)} seconds, Energy: {e:.2f}")
