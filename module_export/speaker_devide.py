from pydub import AudioSegment
import pandas as pd
import os
import argparse

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

def split_audio_by_speaker(file_path, diarization_results, output_dir):
    """
    주어진 음성 파일을 화자별로 나누어 저장하는 함수. 
    동일 화자의 음성은 하나의 파일로 묶어 저장합니다.
    
    Args:
    - file_path (str): 원본 음성 파일 경로
    - diarization_results (list): 다이어리제이션 결과 (화자, 시작, 종료 시간)
    - output_dir (str): 화자별로 음성 파일을 저장할 디렉토리
    """
    # 오디오 파일 로드 및 변환
    audio = AudioSegment.from_file(file_path)
    audio = convert_to_mono(audio)
    audio = convert_to_16bit(audio)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 화자별로 발언들을 모을 딕셔너리
    speaker_audio_chunks = {}

    # 화자별로 음성을 모아서 하나의 오디오로 합침
    for segment in diarization_results:
        start_time = segment['start'] * 1000  # milliseconds
        stop_time = segment['stop'] * 1000  # milliseconds
        speaker = segment['speaker']
        
        # 해당 구간의 오디오 조각 추출
        audio_chunk = audio[start_time:stop_time]

        # 같은 화자의 음성을 모은 후 합치기
        if speaker not in speaker_audio_chunks:
            speaker_audio_chunks[speaker] = audio_chunk
        else:
            speaker_audio_chunks[speaker] += audio_chunk

    # 화자별로 음성 파일을 저장
    for speaker, combined_audio in speaker_audio_chunks.items():
        speaker_file = os.path.join(output_dir, f"speaker_{speaker}.wav")
        combined_audio.export(speaker_file, format="wav")
        print(f"화자 {speaker}의 합쳐진 음성 파일을 저장: {speaker_file}")

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

def main():
    # 커맨드라인 인자 처리
    parser = argparse.ArgumentParser(description="음성 파일을 화자별로 나누어 저장합니다.")
    parser.add_argument('audio_file', type=str, help="WAV 오디오 파일 경로")
    parser.add_argument('diarization_excel', type=str, help="화자 분리 결과가 저장된 엑셀 파일 경로")
    parser.add_argument('output_dir', type=str, help="화자별 음성 파일을 저장할 디렉토리")

    args = parser.parse_args()

    # 다이어리제이션 결과를 엑셀에서 불러오기
    diarization_results = load_diarization_results(args.diarization_excel)
    
    # 화자별 음성 분리
    split_audio_by_speaker(args.audio_file, diarization_results, args.output_dir)

if __name__ == '__main__':
    main()
