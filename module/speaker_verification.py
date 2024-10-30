import torch
from speechbrain.pretrained import SpeakerRecognition

# 1. 모델 로드 (CPU 사용 설정)
device = "cpu"  # CPU 사용
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model").to(device)

# 2. 음성 파일 목록 설정
audio_files = ["../resource/test1.wav", "../resource/test2.wav"]

# 3. 두 음성 파일의 유사도 계산
if len(audio_files) == 2:
    # 각 음성을 CPU에 맞게 로드
    score, prediction = model.verify_files(audio_files[0], audio_files[1])

    # 결과 출력
    result = (
        f"Results for comparing {audio_files[0]} and {audio_files[1]}:\n"
        f"Similarity Score: {score}\n"  # 점수를 변환하지 않고 사용
        f"Are the speakers the same? {'Yes' if prediction else 'No'}\n"
        + "-" * 40 + "\n"
    )

    # 결과를 ../result/similarity.txt 파일에 저장
    with open("../result/similarity.txt", "w") as file:
        file.write(result)

    # 콘솔에 결과 출력
    print(result)
    print("Results saved to '../result/similarity.txt'.")
