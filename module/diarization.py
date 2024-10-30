from pyannote.audio import Pipeline
import torchaudio
import torch
from pyannote.audio.pipelines.utils.hook import ProgressHook
import pandas as pd  # pandas 라이브러리 추가

# 여기에 본인의 액세스 토큰을 입력하세요
access_token = "-"

# 사전 훈련된 화자 분리 모델 로드
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=access_token)
pipeline.to(torch.device("cuda"))

# 사용자로부터 화자 수 입력 받기
num_speakers = int(input("화자 수를 입력하세요: "))  # 예: 2

# 오디오 파일 로드
waveform, sample_rate = torchaudio.load("../resource/audio.wav")

# 진행 상황을 확인하기 위해 ProgressHook 사용
with ProgressHook() as hook:
    # 전체 파일에 대해 추론 실행
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, 
                           hook=hook, 
                           num_speakers=num_speakers)

# 결과를 저장할 리스트 초기화
results = []

# 화자 분리 결과를 리스트에 저장
for turn, _, speaker in diarization.itertracks(yield_label=True):
    result_line = {
        'start': turn.start,
        'stop': turn.end,
        'speaker': f'speaker_{speaker}'
    }
    results.append(result_line)

# DataFrame 생성
df = pd.DataFrame(results)

# DataFrame을 엑셀 파일로 저장
output_file = "../resource/fullText.xlsx"
df.to_excel(output_file, index=False)  # 인덱스 없이 저장

print(f"화자 분리 결과가 {output_file}에 저장되었습니다.")
