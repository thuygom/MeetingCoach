import argparse
from google.cloud import speech_v1p1beta1 as speech
from pydub import AudioSegment
import io
import os
import wave

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

def resample_audio(audio, target_sample_rate=48000):
    """오디오 파일을 리샘플링합니다."""
    if audio.frame_rate != target_sample_rate:
        audio = audio.set_frame_rate(target_sample_rate)
    return audio

def convert_to_16bit(audio):
    """WAV 파일을 16비트 샘플로 변환합니다."""
    return audio.set_sample_width(2)  # 16비트 샘플

def transcribe_audio_chunk(audio_chunk, sample_rate, client):
    """Google Cloud Speech-to-Text API를 사용하여 음성을 텍스트로 변환합니다."""
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

def transcribe_audio_file(file_path):
    """오디오 파일을 텍스트로 변환합니다."""
    # Google Cloud Speech-to-Text 클라이언트 설정 (API 키 경로는 하드코딩)
    api_key_path = '../apiKey/myKey.json'
    client = speech.SpeechClient.from_service_account_file(api_key_path)

    # 오디오 파일의 샘플 레이트 확인
    sample_rate = get_sample_rate(file_path)

    # 오디오 파일 로드 및 변환
    audio = AudioSegment.from_file(file_path)
    audio = convert_to_mono(audio)
    audio = resample_audio(audio, target_sample_rate=sample_rate)
    audio = convert_to_16bit(audio)

    # 오디오 파일을 30초 조각으로 나누기
    chunk_length_ms = 30000  # 30초
    all_transcripts = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        print(f"Transcribing chunk {i // chunk_length_ms}...")
        transcripts = transcribe_audio_chunk(chunk, sample_rate, client)
        print(transcripts)
        all_transcripts.extend(transcripts)

    return all_transcripts

def main():
    # 명령줄 인자 처리 (오디오 파일 경로만 받음)
    parser = argparse.ArgumentParser(description='Process audio file and transcribe using Google Speech-to-Text API.')
    parser.add_argument('audio_file_path', type=str, help='Path to the audio file (WAV format).')
    args = parser.parse_args()

    # 오디오 파일 경로 받아서 처리
    transcripts = transcribe_audio_file(args.audio_file_path)

    # 전체 텍스트를 fullText.txt 파일에 저장
    with open("fullText.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(transcripts))

    print("전체 텍스트가 fullText.txt에 저장되었습니다.")

if __name__ == '__main__':
    main()
