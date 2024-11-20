import cv2
from deepface import DeepFace
from collections import Counter
import argparse

# 커맨드라인 인자 파싱
parser = argparse.ArgumentParser(description="Emotion recognition from a video file.")
parser.add_argument('video_file', type=str, help="Path to the video file")
args = parser.parse_args()

# 비디오 파일 경로
video_file = args.video_file  # 외부 인자로 받은 비디오 파일 경로

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_file)

# 감정 빈도 카운터 초기화
emotion_counts = Counter()
total_faces_detected = 0  # 감정을 분석한 얼굴의 총 수

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV를 사용하여 얼굴 인식
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    detected_faces = faces.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # 각 얼굴에 대해 감정 분석 수행
    for (x, y, w, h) in detected_faces:
        face = frame[y:y+h, x:x+w]
        
        try:
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            # 감정 결과 추출
            emotion = result[0]['dominant_emotion']
            emotion_counts[emotion] += 1
            total_faces_detected += 1

            # 감정 결과를 프레임에 표시
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        except Exception as e:
            print(f"Error analyzing face: {e}")
            continue

    # 결과 프레임 보여주기
    cv2.imshow('OpenCV and DeepFace Emotion Recognition', frame)

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()

# 감정 퍼센티지 계산 및 출력
if total_faces_detected > 0:
    emotion_percentages = {emotion: (count / total_faces_detected) * 100 for emotion, count in emotion_counts.items()}
    print("Emotion percentages in the video:")
    for emotion, percentage in emotion_percentages.items():
        print(f"{emotion}: {percentage:.2f}%")
else:
    print("No faces detected for emotion analysis.")
