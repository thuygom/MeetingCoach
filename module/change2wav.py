import os
from moviepy.editor import AudioFileClip

def mp4_to_wav(file_path):
    """MP4 파일에서 오디오를 추출하여 WAV 파일로 저장합니다."""
    audio = AudioFileClip(file_path)
    wav_file = file_path.replace(".mp4", ".wav")
    audio.write_audiofile(wav_file)
    return wav_file

def convert_all_mp4_to_wav(directory):
    """지정된 디렉토리 내 모든 MP4 파일을 WAV 파일로 변환합니다."""
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            mp4_file_path = os.path.join(directory, filename)
            wav_file_path = mp4_file_path.replace(".mp4", ".wav")
            
            # WAV 파일이 존재하지 않을 경우에만 변환
            if not os.path.exists(wav_file_path):
                print(f"Converting {mp4_file_path} to {wav_file_path}...")
                mp4_to_wav(mp4_file_path)
            else:
                print(f"{wav_file_path} already exists. Skipping conversion.")

# 사용 예제
convert_all_mp4_to_wav("../resource/sample")
