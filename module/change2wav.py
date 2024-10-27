from moviepy.editor import AudioFileClip

def mp4_to_wav(file_path):
    """MP4 파일에서 오디오를 추출하여 WAV 파일로 저장합니다."""
    audio = AudioFileClip(file_path)
    wav_file = file_path.replace(".mp4", ".wav")
    audio.write_audiofile(wav_file)
    return wav_file

# 사용 예제
wav_file_path = mp4_to_wav("../resource/audio.mp4")
