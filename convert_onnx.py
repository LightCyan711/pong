# convert_onnx.py
import torch, onnx, onnxruntime as ort, numpy as np
from stable_baselines3 import PPO
from game_env import GameEnv

MODEL_PATH = "sb3_logs/final_model.zip"
ONNX_PATH = "model.onnx"

class OnnxablePolicy(torch.nn.Module):
    """
    SB3 Policy의 순수 Torch 경로만 사용해서 ONNX로 내보내는 래퍼.
    - 입력: (N, obs_dim) float32
    - 출력: (N,) int64  (deterministic argmax)
    """
    def __init__(self, sb3_policy):
        super().__init__()
        # SB3 내부 서브모듈을 직접 참조 (전부 nn.Module들이라 ONNX 가능)
        self.features_extractor = sb3_policy.features_extractor
        self.mlp_extractor = sb3_policy.mlp_extractor
        self.action_net = sb3_policy.action_net
        # 관측치 정규화(VecNormalize)가 없다면 obs_rms는 없을 수 있음
        self.obs_rms = getattr(sb3_policy, "obs_rms", None)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # (N, obs_dim) 보장
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        # 필요시 정규화 (VecNormalize를 안 썼다면 None)
        if self.obs_rms is not None:
            # SB3 NormalizeObservation 로직과 유사
            eps = 1e-8
            mean = torch.as_tensor(self.obs_rms.mean, dtype=observation.dtype, device=observation.device)
            var  = torch.as_tensor(self.obs_rms.var,  dtype=observation.dtype, device=observation.device)
            observation = (observation - mean) / torch.sqrt(var + eps)
            observation = torch.clamp(observation, -10.0, 10.0)

        features = self.features_extractor(observation)
        latent_pi, _ = self.mlp_extractor(features)   # policy/ value 공용에서 policy쪽 latent만 사용
        logits = self.action_net(latent_pi)           # (N, action_dim)
        action = torch.argmax(logits, dim=-1).to(torch.int64)  # deterministic
        return action

def main():
    # 1) SB3 모델 로드
    model = PPO.load(MODEL_PATH, device="cpu")
    model.policy.eval()

    # 2) ONNX로 내보낼 Torch 모듈 준비
    onnxable = OnnxablePolicy(model.policy).eval()

    # 3) 더미 입력 (env에서 obs_dim 추출)
    env = GameEnv(render_mode=None)
    obs_dim = int(np.prod(env.observation_space.shape))
    dummy = torch.randn(1, obs_dim, dtype=torch.float32)
    env.close()

    # 4) ONNX Export (동적축/최신 opset)
    torch.onnx.export(
        onnxable,
        dummy,
        ONNX_PATH,
        export_params=True,
        opset_version=17,                 # 충분히 최신
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
    )
    print(f"ONNX 변환 완료: {ONNX_PATH}")

    # 5) 간단 검증 (onnxruntime)
    print("\n--- ONNX 모델 검증 ---")
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    test_in = np.random.randn(4, obs_dim).astype(np.float32)  # 배치 4
    out = sess.run([out_name], {in_name: test_in})[0]
    print(f"검증 성공! 입력 shape: {test_in.shape}, 출력 shape: {out.shape}, 예시 action: {out[:4]}")
    # 기대: 출력 dtype=int64, shape=(4,)
    assert out.dtype in (np.int64, np.int32)
    assert out.shape == (4,)
    print("형상/자료형 확인 완료.")

if __name__ == "__main__":
    main()
