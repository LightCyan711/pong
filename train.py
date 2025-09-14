# train.py
import os, time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, ProgressBarCallback  # ★ 추가
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces
from game_env import GameEnv 

TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_FREQ = 10_000
LOG_DIR = "sb3_logs"
CHECKPOINT_DIR = os.path.join(LOG_DIR, "checkpoints")

def find_latest_checkpoint():
    if not os.path.exists(CHECKPOINT_DIR): return None
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("rl_model_") and f.endswith(".zip")]
    if not checkpoints: return None
    latest_checkpoint = max(checkpoints, key=lambda f: int(f.split("_")[2].replace(".zip", "")))
    return os.path.join(CHECKPOINT_DIR, latest_checkpoint)

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    env = DummyVecEnv([lambda: GameEnv(render_mode=None)])
    obs_space = env.observation_space
    policy = "CnnPolicy" if isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 3 else "MlpPolicy"
    print(f"자동 선택된 정책: {policy}")
    latest_checkpoint = find_latest_checkpoint()
    if latest_checkpoint:
        print(f"최신 체크포인트에서 학습을 재개합니다: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env)
        log_name = os.path.basename(os.path.dirname(os.path.dirname(latest_checkpoint)))
        run_path = os.path.join(LOG_DIR, log_name)
        custom_logger = configure(run_path, ["stdout", "tensorboard"])
        model.set_logger(custom_logger)
    else:
        print("새로운 학습을 시작합니다.")
        log_name = f"PPO_{int(time.time())}"
        run_path = os.path.join(LOG_DIR, log_name)
        custom_logger = configure(run_path, ["stdout", "tensorboard"])
        model = PPO(policy, env, verbose=1, tensorboard_log=None)
        model.set_logger(custom_logger)

    # ★ 진행 상황 표시 콜백 추가
    checkpoint_callback = CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=CHECKPOINT_DIR, name_prefix="rl_model")
    progress_callback = ProgressBarCallback()  # tqdm 기반 진행 바
    combined_callback = CallbackList([checkpoint_callback, progress_callback])

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=combined_callback,            # ★ 콜백 전달
            reset_num_timesteps=False
        )
        model.save(os.path.join(LOG_DIR, "final_model.zip"))
        print(f"학습 완료! 최종 모델이 {os.path.join(LOG_DIR, 'final_model.zip')}에 저장되었습니다.")
    finally:
        env.close()

if __name__ == '__main__': 
    main()
