import torch

if torch.cuda.is_available():
    print("CUDA가 사용 가능합니다. GPU를 사용할 준비가 되었습니다.")
    print(f"GPU의 개수: {torch.cuda.device_count()}")
    print(f"현재 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA가 사용 불가능합니다. PyTorch는 CPU를 사용합니다.")