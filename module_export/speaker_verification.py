import argparse
import torch
from speechbrain.pretrained import SpeakerRecognition

# 1. 모델 로드 (CPU 사용 설정)
device = "cpu"  # CPU 사용
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model").to(device)

def verify_audio_files(audio_file1, audio_file2):
    """두 음성 파일의 유사도를 계산하고 결과를 반환합니다."""
    # 두 음성을 CPU에 맞게 로드
    score, prediction = model.verify_files(audio_file1, audio_file2)

    # 결과 출력
    result = f"Similarity Score: {score}"  # 점수를 변환하지 않고 사용

    # 결과를 ../result/similarity.txt 파일에 저장
    with open("../result/similarity.txt", "w") as file:
        file.write(result)

    return result

def main():
    # 명령줄 인자 처리 (두 음성 파일 경로 받기)
    parser = argparse.ArgumentParser(description="Calculate similarity between two audio files using SpeakerRecognition model.")
    parser.add_argument("audio_file1", type=str, help="Path to the first audio file.")
    parser.add_argument("audio_file2", type=str, help="Path to the second audio file.")
    args = parser.parse_args()

    # 두 음성 파일 유사도 계산
    result = verify_audio_files(args.audio_file1, args.audio_file2)

    # 콘솔에 결과 출력
    print(result)
    print("Results saved to '../result/similarity.txt'.")

if __name__ == "__main__":
    main()
