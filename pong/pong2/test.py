import torch
import gymnasium as gym
import ale_py
import os

import gymnasium as gym
print([env for env in gym.envs.registry.keys() if 'Pong' in env])

"""

# 1. PyTorch CUDA 사용 가능 여부 확인
print("CUDA 사용 가능 여부:", torch.cuda.is_available())
print("PyTorch 버전:", torch.__version__)
print("ale_py 버전:", ale_py.__version__)

# 2. ROM 경로 환경변수 설정 (AutoROM이 설치한 기본 경로)
os.environ["ALE_ROM_DIR"] = r"C:\Users\hjc\anaconda3\envs\pong2\Lib\site-packages\AutoROM\roms"

# 3. 설치된 Atari 환경 목록 출력
env_list = [env for env in gym.envs.registry.keys() if 'ALE' in env]
print("설치된 ALE 환경들:", env_list)

# 4. Pong 환경 생성 테스트
try:
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    print("Pong 환경 생성 성공!")
    env.reset()
except Exception as e:
    print("Pong 환경 생성 실패:", e)
"""