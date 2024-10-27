from pyannote.audio import Pipeline
import torchaudio
from pyannote.audio.pipelines.utils.hook import ProgressHook

# 여기에 본인의 액세스 토큰을 입력하세요
access_token = "-"

# 사전 훈련된 화자 분리 모델 로드
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=access_token)

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

# 전체 파일 결과 출력
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

# 특정 구간에 대해 추론 실행 (필요시 주석 해제)
# excerpt = Segment(start=2.0, end=5.0)  # 예시: 2초에서 5초 구간
# 크롭된 오디오 가져오기
# waveform, sample_rate = Audio().crop("../resource/audio.wav", excerpt)
# 크롭된 오디오에 대해 화자 분리 수행
# diarization_excerpt = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook, num_speakers=num_speakers)
# 크롭된 구간 결과 출력
# for turn, _, speaker in diarization_excerpt.itertracks(yield_label=True):
#     print(f"Excerpt - start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
