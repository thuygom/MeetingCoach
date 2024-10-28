import cv2
import mediapipe as mp

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# 비디오 파일 경로
video_file = "path/to/your/video.mp4"  # 사용할 MP4 파일의 경로로 변경하세요

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_file)

# MediaPipe 얼굴 메쉬 및 손 인식 초기화 (GPU 사용)
with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh, \
     mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 색상 변환 (BGR to RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 얼굴 메쉬 인식
        face_results = face_mesh.process(rgb_frame)

        # 손 인식
        hand_results = hands.process(rgb_frame)

        # 얼굴 메쉬 그리기
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    h, w, _ = frame.shape
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # 손 제스처 그리기
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        # 결과 프레임 보여주기
        cv2.imshow('MediaPipe Face and Hand Recognition', frame)

        # 'q' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 비디오 캡처 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
