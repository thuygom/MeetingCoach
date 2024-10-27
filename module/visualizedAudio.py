import numpy as np
import matplotlib.pyplot as plt
import librosa

def plot_pitch(file_path):
    # 오디오 파일 로드
    y, sr = librosa.load(file_path)
    
    # 피치 추정
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # 피치를 가져오기 위한 최대값 인덱스 찾기
    pitches = np.array([np.max(p) if np.max(p) > 0 else 0 for p in pitches.T])

    # 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(pitches, label='Pitch (Hz)')
    plt.title("Pitch Tracking")
    plt.xlabel("Frame Index")
    plt.ylabel("Pitch (Hz)")
    plt.grid()
    plt.legend()
    plt.show()

# 사용 예제
file_path = "../resource/audio.wav"  # 여기에 WAV 파일 경로를 입력하세요
plot_pitch(file_path)
