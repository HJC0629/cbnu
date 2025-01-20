from ultralytics import YOLO

# 1. YOLO 모델 초기화
# YOLOv8의 사전 학습된 Nano 모델을 기반으로 시작합니다. Nano 모델은 경량화된 모델로 빠른 학습과 추론이 가능합니다.
model = YOLO('yolov8n.pt')  # 'yolov8n.pt'은 Nano 모델 가중치 파일입니다.

# 2. 학습 시작
# 학습 파라미터 설정과 함께 학습을 시작합니다.
model.train(
    data='data.yaml',       # 데이터셋 정의 파일 (아래 설명 참조)
    epochs=50,               # 학습 epoch 수 (50번 반복 학습)
    batch=16,                # 배치 크기 (한 번에 학습하는 데이터의 수)
    imgsz=640,               # 학습 이미지 크기 (640x640)
    project='runs/train',    # 학습 결과 저장 디렉토리
    name='face_detection',   # 실험 이름 (결과 디렉토리에 반영됨)
    pretrained=True          # 사전 학습된 가중치 사용 여부
)

# 3. 학습 완료 후 결과
# 학습이 완료되면 모델 가중치 파일은 'runs/train/face_detection/weights/'에 저장됩니다.
# - 'best.pt': 검증 성능이 가장 좋은 가중치 파일
# - 'last.pt': 마지막 학습 epoch에서 저장된 가중치 파일
print("학습 완료! 결과는 'runs/train/face_detection/' 디렉토리에 저장되었습니다.")
