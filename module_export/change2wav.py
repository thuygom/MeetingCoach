import os
from moviepy.editor import AudioFileClip

def mp4_to_wav(file_path):
    """
    MP4 파일에서 오디오를 추출하여 WAV 파일로 저장합니다.
    
    Args:
        file_path (str): MP4 파일의 전체 경로.
        
    Returns:
        str: 생성된 WAV 파일의 경로.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    if not file_path.endswith(".mp4"):
        raise ValueError("The provided file is not an MP4 file.")
    
    # WAV 파일 경로 생성
    wav_file = file_path.replace(".mp4", ".wav")
    
    # MP4에서 WAV로 변환
    audio = AudioFileClip(file_path)
    print(f"Converting {file_path} to {wav_file}...")
    audio.write_audiofile(wav_file)
    audio.close()  # 리소스 해제
    
    return wav_file

if __name__ == "__main__":
    import sys
    
    # 명령줄 인자로 파일 경로를 받음
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        wav_file_path = mp4_to_wav(file_path)
        print(f"Conversion complete. WAV file saved at: {wav_file_path}")
    except Exception as e:
        print(f"Error: {e}")
