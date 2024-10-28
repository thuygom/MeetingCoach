import cv2
from deepface import DeepFace

# 비디오 파일 경로
video_file = "path/to/your/video.mp4"  # 사용할 MP4 파일의 경로로 변경하세요

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_file)

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
        result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)

        # 감정 결과 추출
        emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][emotion]

        # 감정 결과를 프레임에 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{emotion}: {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 결과 프레임 보여주기
    cv2.imshow('OpenCV and DeepFace Emotion Recognition', frame)

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
