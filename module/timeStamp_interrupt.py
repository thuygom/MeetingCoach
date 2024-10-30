import pandas as pd

# 엑셀 파일에서 데이터 불러오기
file_path = '../resource/fullText.xlsx'
df = pd.read_excel(file_path)

# 컬럼 이름 변경 (파일에 맞게 조정)
df.columns = ['start_time', 'end_time', 'speaker', 'text']

# 끼어들기 검출
interruptions = []
for i in range(1, len(df)):
    # 현재 발화의 'text'가 비어있는 경우(리액션으로 간주) 건너뛰기
    if pd.isna(df.loc[i, 'text']) or df.loc[i, 'text'].strip() == '':
        continue
    
    # 현재 발화 화자와 이전 발화 화자가 다를 경우 끼어들기 가능성
    if df.loc[i, 'speaker'] != df.loc[i - 1, 'speaker']:
        # 이전 발화가 끝난 시간과 현재 발화 시작 시간 비교
        prev_end = df.loc[i - 1, 'end_time']
        curr_start = df.loc[i, 'start_time']
        
        # 발화 시작 시간 차이가 0.5초 이하이면 끼어들기로 간주
        if curr_start - prev_end <= -0.3:
            interruptions.append({
                'interrupter': df.loc[i, 'speaker'],
                'time': curr_start,
                'text': df.loc[i, 'text']
            })

# 끼어들기 결과 출력
print("Detected interruptions:")
for interrupt in interruptions:
    print(f"Time: {interrupt['time']}s, Speaker: {interrupt['interrupter']}, Text: {interrupt['text']}")
