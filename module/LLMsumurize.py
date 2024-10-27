import os
from openai import OpenAI

client = OpenAI(api_key="-")  # 여기에서 YOUR_API_KEY를 실제 API 키로 변경하세요

def summarize_text(file_path):
    # 텍스트 파일 읽기
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # OpenAI API를 사용하여 텍스트 요약하기
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"한국어 회의 내용을 주요사건 및 논의점을 요약해서 설명해:\n\n{text}"
            }
        ],
        model="gpt-3.5-turbo",
        max_tokens=150,  # 요약의 최대 토큰 수
        temperature=0.5,  # 창의성 조절
    )

    # 요약 결과 추출
    summary = chat_completion.choices[0].message.content
    return summary

# 사용 예제
file_path = "../resource/fullText.txt"
summary = summarize_text(file_path)

# 요약 결과 출력
print("요약 결과:")
print(summary)

# 요약 결과를 파일에 저장
with open("../resource/summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("요약이 summary.txt에 저장되었습니다.")
