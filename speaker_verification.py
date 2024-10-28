from speechbrain.pretrained import SpeakerRecognition

# 1. 모델 로드
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")

# 2. 음성 파일 경로 설정
audio_file1 = "../resource/audio1.wav"
audio_file2 = "../resource/audio2.wav"

# 3. 두 음성 파일의 유사도 계산
score, prediction = model.verify_files(audio_file1, audio_file2)

# 4. 결과 출력
print(f"Similarity Score: {score}")
print(f"Are the speakers the same? {'Yes' if prediction else 'No'}")
