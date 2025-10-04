import math
import numpy as np
from typing import List, Tuple, Optional, Dict

import gymnasium as gym
from gymnasium import spaces


class SingleStepOptimAntennaEnv(gym.Env):
    """
    An RL environment that frames the antenna optimization problem as a single-step task.
    The agent's action is a complete solution, and the reward is its evaluated quality.
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
            self,
            uav_doas_deg: List[float],
            N: int = 8,
            D_lambda: float = 10.0,
            dmin_lambda: float = 0.4,
            fc_hz: float = 2.4e9,
            seed: Optional[int] = None,
    ):
        super().__init__()

        self.N = N
        self.K = len(uav_doas_deg)
        self.uav_doas_deg = np.array(uav_doas_deg, dtype=np.float32)

        self.D_lambda = float(D_lambda)
        self.dmin_lambda = float(dmin_lambda)
        self.fc = float(fc_hz)
        self.c = 3e8
        self.lam = self.c / self.fc
        self.D_meters = self.D_lambda * self.lam
        self.dmin_meters = self.dmin_lambda * self.lam

        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3 * N,), dtype=np.float32
        )

        action_low = np.array([0.0] * N + [-1.0] * (2 * N), dtype=np.float32)
        action_high = np.array([1.0] * N + [1.0] * (2 * N), dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        self.x: np.ndarray = np.zeros(N)
        self.weights: np.ndarray = np.zeros((N, 1), dtype=np.complex64)

    def _get_steering_matrix(self, thetas_deg: np.ndarray) -> np.ndarray:
        thetas_rad = np.deg2rad(thetas_deg)
        phase_factor = (2 * np.pi / self.lam) * np.sin(thetas_rad)
        phase_matrix = np.outer(self.x, phase_factor)
        return np.exp(1j * phase_matrix)

    def _calculate_gains(self) -> np.ndarray:
        A = self._get_steering_matrix(self.uav_doas_deg)
        array_factor = self.weights.conj().T @ A
        gains = np.abs(array_factor) ** 2
        return gains.flatten()

    def _project_positions(self, x: np.ndarray) -> np.ndarray:
        y = np.sort(x)
        y = np.clip(y, 0.0, self.D_meters)
        for i in range(1, self.N):
            y[i] = max(y[i], y[i - 1] + self.dmin_meters)
        if y[-1] > self.D_meters:
            y[-1] = self.D_meters
            for i in range(self.N - 2, -1, -1):
                y[i] = min(y[i], y[i + 1] - self.dmin_meters)
        return np.clip(y, 0.0, self.D_meters)

    def _project_weights(self, w: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(w)
        if norm > 1e-6: return w / norm
        return np.ones_like(w) / np.sqrt(self.N)

    def _get_observation(self) -> np.ndarray:
        pos_norm = (self.x / self.D_meters) * 2 - 1
        w_col = self.weights.flatten()
        return np.concatenate([pos_norm, w_col.real, w_col.imag]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        initial_pos = np.linspace(0, self.D_meters, self.N)
        self.x = self._project_positions(initial_pos)
        self.weights = self._project_weights(np.ones((self.N, 1), dtype=np.complex64))

        obs = self._get_observation()
        info = {"uav_doas": self.uav_doas_deg.tolist()}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        target_pos_norm = action[:self.N]
        target_re = action[self.N: 2 * self.N]
        target_im = action[2 * self.N:]

        target_x_unconstrained = target_pos_norm * self.D_meters
        target_weights_unconstrained = (target_re + 1j * target_im).reshape(self.N, 1)

        self.x = self._project_positions(target_x_unconstrained)
        self.weights = self._project_weights(target_weights_unconstrained)

        gains = self._calculate_gains()
        total_gain = float(np.sum(gains))
        reward = total_gain

        # **CORRECTED LOGIC**
        # The episode is always "done" after one step in this formulation.
        # This is a natural end, so it's a "termination".
        terminated = True
        truncated = False  # It was not cut short by a time limit.

        obs = self._get_observation()
        info = {"gains": gains.tolist(), "total_gain": total_gain}

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        total_gain = np.sum(self._calculate_gains())
        # Added step count for clarity, even though it's always 1

        print(
            f"Step: 1 | Total Gain: {total_gain:7.4f} | Positions (lambda): {np.array2string(self.x / self.lam, precision=2)} | weight {self.weights}")


if __name__ == '__main__':
    from gymnasium.utils.env_checker import check_env

    print(">>> Creating and Checking Single-Step Optimization Environment...")

    FIXED_DOAS = [-50.0, 0.0, 30.0]
    env = SingleStepOptimAntennaEnv(uav_doas_deg=FIXED_DOAS, N=8)

    # This check will now pass.
    check_env(env.unwrapped)
    print(">>> Environment check passed!")

    print(f"\n>>> Running single-step optimization example for DOAs: {FIXED_DOAS}...")
    obs, info = env.reset()
    print("Initial State (before action):")
    # In this env, the initial state is just a standard placeholder.
    print(f"Total Gain: {np.sum(env._calculate_gains()):7.4f}| Positions (lambda): {np.array2string(env.x / env.lam, precision=2)} | weight {env.weights}")

    # Agent proposes a single, complete solution
    proposed_solution = env.action_space.sample()
    print(f"\nAgent proposed a random solution (action)...")

    # The environment evaluates this single solution in one step
    obs, reward, terminated, truncated, info = env.step(proposed_solution)

    print(f"\n--- Evaluation Result ---")
    print(f"Final State (the proposed solution after applying constraints):")
    env.render()
    print(f"Reward Received (Total Gain): {reward:.4f}")
    print(f"Episode Terminated: {terminated}")
    print(f"Episode Truncated: {truncated}")

    env.close()
