import argparse
from google.cloud import speech_v1p1beta1 as speech
from pydub import AudioSegment
import io
import wave
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

def load_diarization_results(excel_path):
    """엑셀 파일에서 다이어리제이션 결과를 읽어옵니다."""
    # 엑셀 파일에서 다이어리제이션 결과 읽기
    df = pd.read_excel(excel_path)
    
    # 다이어리제이션 결과를 딕셔너리 형태로 변환
    diarization_results = []
    for _, row in df.iterrows():
        diarization_results.append({
            'start': row['start'],
            'stop': row['stop'],
            'speaker': row['speaker']
        })
    return diarization_results

def process_audio(file_path, diarization_excel_path):
    """파일 경로와 다이어리제이션 결과 엑셀 파일을 받아 처리하여 엑셀로 저장"""
    # 다이어리제이션 결과를 엑셀에서 불러오기
    diarization_results = load_diarization_results(diarization_excel_path)
    
    # 대화 내용을 텍스트로 변환
    output_data = transcribe_audio_file(file_path, diarization_results)

    # DataFrame 생성
    df = pd.DataFrame(output_data)

    # DataFrame을 엑셀 파일로 저장
    output_file = file_path.replace(".wav", "_fullText.xlsx")
    df.to_excel(output_file, index=False)

    print(f"화자별 대화 내용이 {output_file}에 저장되었습니다.")
    return output_file

def main():
    # 커맨드라인 인자 처리
    parser = argparse.ArgumentParser(description="음성 파일을 텍스트로 변환하고 화자별 대화 내용을 저장합니다.")
    parser.add_argument('audio_file', type=str, help="WAV 오디오 파일 경로")
    parser.add_argument('diarization_excel', type=str, help="화자 분리 결과가 저장된 엑셀 파일 경로")

    args = parser.parse_args()

    # 파일 경로와 다이어리제이션 결과를 처리
    process_audio(args.audio_file, args.diarization_excel)

if __name__ == '__main__':
    main()
