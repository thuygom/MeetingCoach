import pandas as pd
import argparse
from collections import defaultdict

def detect_interruptions(file_path):
    """엑셀 파일을 불러와서 화자별 끼어들기 횟수를 카운트하는 함수"""
    # 엑셀 파일에서 데이터 불러오기
    df = pd.read_excel(file_path)

    # 컬럼 이름 변경 (파일에 맞게 조정)
    df.columns = ['start_time', 'end_time', 'speaker', 'dialogue']

    # 화자별 끼어들기 횟수 카운트용 딕셔너리
    interrupter_count = defaultdict(int)

    # 끼어들기 검출
    for i in range(1, len(df)):
        # 현재 발화의 'dialogue'가 비어있는 경우(리액션으로 간주) 건너뛰기
        if pd.isna(df.loc[i, 'dialogue']) or df.loc[i, 'dialogue'].strip() == '':
            continue
        
        # 현재 발화 화자와 이전 발화 화자가 다를 경우 끼어들기 가능성
        if df.loc[i, 'speaker'] != df.loc[i - 1, 'speaker']:
            # 이전 발화가 끝난 시간과 현재 발화 시작 시간 비교
            prev_end = df.loc[i - 1, 'end_time']
            curr_start = df.loc[i, 'start_time']
            
            # 발화 시작 시간 차이가 0.5초 이하이면 끼어들기로 간주
            if curr_start - prev_end <= -0.3:
                interrupter_count[df.loc[i, 'speaker']] += 1

    return interrupter_count

def main():
    # 명령줄 인자 처리 (엑셀 파일 경로 받기)
    parser = argparse.ArgumentParser(description="Detect interruptions by speakers from an Excel file.")
    parser.add_argument("file_path", type=str, help="Path to the Excel file containing dialogues.")
    args = parser.parse_args()

    # 끼어들기 검출
    interrupter_count = detect_interruptions(args.file_path)

    # 화자별 끼어들기 횟수 출력
    print("Detected interruptions by speaker:")
    for speaker, count in interrupter_count.items():
        print(f"Speaker: {speaker}, Interruptions: {count}")

if __name__ == "__main__":
    main()
