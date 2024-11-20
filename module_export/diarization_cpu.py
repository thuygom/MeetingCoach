import sys
from pyannote.audio import Pipeline
import torchaudio
import pandas as pd
from pyannote.audio.pipelines.utils.hook import ProgressHook

# 여기에 본인의 액세스 토큰을 입력하세요
access_token = "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"

# 화자 수를 명령줄 인자로 받기
if len(sys.argv) != 3:
    print("Usage: python script.py <num_speakers> <audio_file_path>")
    sys.exit(1)

num_speakers = int(sys.argv[1])  # 첫 번째 인자: 화자 수
audio_file_path = sys.argv[2]   # 두 번째 인자: 오디오 파일 경로

# 사전 훈련된 화자 분리 모델 로드
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=access_token)

# 오디오 파일 로드
waveform, sample_rate = torchaudio.load(audio_file_path)

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
output_file = "fullText.xlsx"
df.to_excel(output_file, index=False)  # 인덱스 없이 저장

print(f"화자 분리 결과가 {output_file}에 저장되었습니다.")
