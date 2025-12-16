from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':

    base_dir = os.path.dirname(os.path.abspath(__file__))
    # data.yaml 파일의 절대 경로를 구합니다.
    yaml_path = os.path.join(base_dir, 'data.yaml')



    model = YOLO('yolov8n.pt')



    results = model.train(
        data=yaml_path,
        epochs=20,
        imgsz=640,
        plots=True
    )

    print(">> 학습 완료")
    # 학습된 결과는 runs/detect/train 폴더에 자동 저장됩니다.