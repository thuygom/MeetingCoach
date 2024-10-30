from pyannote.audio import Pipeline
from google.cloud import speech_v1p1beta1 as speech
from pydub import AudioSegment
import io
import wave
import torchaudio
import pandas as pd
import torch

def get_sample_rate(file_path):
    """WAV 파일의 샘플 레이트를 확인합니다."""
    with wave.open(file_path, 'rb') as wf:
        sample_rate = wf.getframerate()
    return sample_rate

def convert_to_mono(audio):
    """오디오 파일을 모노로 변환합니다."""
    if audio.channels != 1:
        audio = audio.set_channels(1)
    return audio

def convert_to_16bit(audio):
    """WAV 파일을 16비트 샘플로 변환합니다."""
    return audio.set_sample_width(2)  # 16비트 샘플

def transcribe_audio_chunk(audio_chunk, sample_rate):
    """Google Cloud Speech-to-Text API를 사용하여 음성을 텍스트로 변환합니다."""
    client = speech.SpeechClient.from_service_account_file('../apiKey/myKey.json')
    
    # 오디오 조각을 메모리에서 처리
    with io.BytesIO() as audio_file:
        audio_chunk.export(audio_file, format="wav")
        audio_file.seek(0)
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="ko-KR",
    )

    # 파일이 길 경우, long_running_recognize 사용
    operation = client.long_running_recognize(config=config, audio=audio)
    print('Waiting for operation to complete...')
    response = operation.result(timeout=90)

    transcripts = []
    for result in response.results:
        transcripts.append(result.alternatives[0].transcript)

    return transcripts

def transcribe_audio_file(file_path, diarization_results):
    """오디오 파일을 텍스트로 변환하고 화자별로 대화 내용을 저장합니다."""
    # 오디오 파일의 샘플 레이트 확인
    sample_rate = get_sample_rate(file_path)

    # 오디오 파일 로드 및 변환
    audio = AudioSegment.from_file(file_path)
    audio = convert_to_mono(audio)
    audio = convert_to_16bit(audio)

    output_data = []

    # 화자 별로 음성을 인식
    for segment in diarization_results:
        start_time = segment['start'] * 1000  # milliseconds
        stop_time = segment['stop'] * 1000  # milliseconds
        speaker = segment['speaker']

        # 해당 구간의 오디오 조각 추출
        audio_chunk = audio[start_time:stop_time]

        # 오디오 조각 텍스트 변환
        print(f"Transcribing {speaker} from {segment['start']} to {segment['stop']} seconds...")
        transcripts = transcribe_audio_chunk(audio_chunk, sample_rate)

        # 대화 내용을 문자열로 합침
        dialog_content = ' '.join(transcripts)

        # 데이터 추가
        output_data.append({
            'start': segment['start'],
            'stop': segment['stop'],
            'speaker': speaker,
            'dialogue': dialog_content
        })

    return output_data

def diarize_audio(file_path, num_speakers):
    """화자 분리 수행 후 결과를 반환합니다."""
    access_token = "-"
    
    # 사전 훈련된 화자 분리 모델 로드
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=access_token)
    pipeline.to(torch.device("cuda"))

    # 오디오 파일 로드
    waveform, sample_rate = torchaudio.load(file_path)

    # 화자 분리 수행
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=num_speakers)

    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append({
            'start': turn.start,
            'stop': turn.end,
            'speaker': f'speaker_{speaker}'
        })

    return results

# 사용 예제
audio_file_path = '../resource/audio.wav'

# 화자 수 입력 받기
num_speakers = int(input("화자 수를 입력하세요: "))  # 예: 2

# 화자 분리 수행
diarization_results = diarize_audio(audio_file_path, num_speakers)

# 대화 내용을 텍스트로 변환
output_data = transcribe_audio_file(audio_file_path, diarization_results)

# DataFrame 생성
df = pd.DataFrame(output_data)

# DataFrame을 엑셀 파일로 저장
output_file = "../resource/fullText.xlsx"
df.to_excel(output_file, index=False)

print(f"화자별 대화 내용이 {output_file}에 저장되었습니다.")
