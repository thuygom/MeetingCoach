import pandas as pd
from openai import OpenAI
import sys

# API 키를 파일에서 읽어오기
def get_api_key(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()  # API 키를 반환하고, 불필요한 공백 제거

# OpenAI 클라이언트 설정
api_key = get_api_key("../apiKey/gptKey.txt")  # 파일에서 API 키 읽어오기
client = OpenAI(api_key=api_key)

# 퀴즈 생성 함수
def generate_quiz(summary_text):
    # 퀴즈 생성 요청
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": f"다음 요약 내용을 기반으로 객관식 퀴즈 3~5개와 각 문제의 정답을 생성해주세요.\n\n{summary_text}"
        }],
        model="gpt-3.5-turbo",
        max_tokens=1000,
        temperature=0.5,
    )

    # 퀴즈 및 정답 추출
    quiz_content = chat_completion.choices[0].message.content
    return quiz_content

# 외부 인자로 요약 텍스트 받기
if len(sys.argv) < 2:
    print("요약된 텍스트를 인자로 제공해주세요.")
    sys.exit(1)

# 요약된 텍스트 파일 경로 받기
summary_file_path = sys.argv[1]

# 요약 텍스트 읽기
with open(summary_file_path, "r", encoding="utf-8") as f:
    summary_text = f.read()

# 퀴즈 생성
quiz_text = generate_quiz(summary_text)

# 생성된 퀴즈 출력
print("생성된 객관식 퀴즈:")
print(quiz_text)

# 퀴즈와 정답을 pandas DataFrame으로 저장
quiz_lines = quiz_text.split("\n")
questions = []
choices = []
answers = []

# 퀴즈와 선택지를 처리하는 부분
for line in quiz_lines:
    if line.startswith("Q:"):
        question = line[3:].strip()
        questions.append(question)
        choice_set = []
    elif line.startswith("A:"):
        correct_answer = line[3:].strip()
        answers.append(correct_answer)
    elif line.startswith("-"):
        choice_set.append(line[2:].strip())

# DataFrame으로 변환
quiz_df = pd.DataFrame({
    "Question": questions,
    "Choices": choices,
    "Answer": answers
})

# DataFrame 출력
print("\n퀴즈 DataFrame:")
print(quiz_df)

# 퀴즈 결과를 CSV 파일로 저장
output_file = summary_file_path.replace(".txt", "_quiz.csv")
quiz_df.to_csv(output_file, index=False, encoding="utf-8")

print(f"퀴즈가 {output_file}에 저장되었습니다.")
