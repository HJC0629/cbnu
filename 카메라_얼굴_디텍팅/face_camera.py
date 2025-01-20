import cv2
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("best.pt")

# 카메라 열기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 영상을 가져올 수 없습니다.")
        break

    # YOLO 모델로 감지 수행
    results = model(frame)

    # 결과에서 감지된 객체를 박스 처리
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # 원래 좌표
        conf = result.conf[0]
        cls = result.cls[0]

        # 클래스가 얼굴인지 확인
        if cls == 0:  # 얼굴 클래스 번호가 0이라고 가정
            # 박스 크기 줄이기
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            scale = 0.8  # 박스 크기를 80%로 줄임
            new_width = int((x2 - x1) * scale)
            new_height = int((y2 - y1) * scale)

            # 새로운 좌표 계산
            x1_new = center_x - new_width // 2
            x2_new = center_x + new_width // 2
            y1_new = center_y - new_height // 2
            y2_new = center_y + new_height // 2

            # 줄어든 박스 그리기
            cv2.rectangle(frame, (x1_new, y1_new), (x2_new, y2_new), (0, 255, 0), 2)
            label = f"Face: {conf:.2f}"
            cv2.putText(frame, label, (x1_new, y1_new - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 화면 표시
    cv2.imshow("Face Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제 및 종료
cap.release()
cv2.destroyAllWindows()
