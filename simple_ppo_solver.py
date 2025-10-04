import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from tqdm import tqdm
import inspect
import time

# Make sure your environment file is accessible
# This is a placeholder since I don't have the file.
# If you run this, ensure 'env.antenna_env' is in your PYTHONPATH.
try:
    from env.antenna_env import SingleStepOptimAntennaEnv
except ImportError:
    print("Warning: 'env.antenna_env' not found. Using a dummy environment for code validation.")
    import gymnasium as gym

    SingleStepOptimAntennaEnv = lambda **kwargs: gym.make("Pendulum-v1")


################################################################################
# 1. MODIFICATION: Customizable Network Architectures
# We define different network modules that can be chosen dynamically.
################################################################################

def get_activation(activation_name: str):
    """Returns an activation function module from its name."""
    if activation_name.lower() == 'tanh':
        return nn.Tanh()
    elif activation_name.lower() == 'relu':
        return nn.ReLU()
    elif activation_name.lower() == 'leakyrelu':
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")


class MLP(nn.Module):
    """A standard Multi-Layer Perceptron."""

    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers, activation):
        super(MLP, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(get_activation(activation))
        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(get_activation(activation))
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CNN(nn.Module):
    """
    A 1D Convolutional Neural Network.
    NOTE: CNNs are typically used for data with spatial structure (like images).
    For a flat state vector, its effectiveness might be limited.
    """

    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers, activation):
        super(CNN, self).__init__()
        # We treat the flat state vector as a 1D sequence with 1 channel.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # Calculate the flattened size after conv and pool layers
        # This is a bit manual and depends on the input_dim and conv architecture
        # A more robust implementation might dynamically calculate this
        conv_output_size = 16 * (input_dim // 2)

        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = get_activation(activation)

    def forward(self, x):
        # Reshape input from [batch, features] to [batch, channels, length]
        x = x.unsqueeze(1)
        x = self.pool(self.activation(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class RNN(nn.Module):
    """
    A simple Recurrent Neural Network.
    NOTE: RNNs are for sequential data. In this single-step environment,
    we treat each state as a sequence of length 1. This is not the typical
    use case for an RNN.
    """

    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers, activation):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_hidden_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape input from [batch, features] to [batch, seq_len, features]
        x = x.unsqueeze(1)
        # We don't need the hidden state output h_n for this simple case
        out, _ = self.rnn(x)
        # Take the output of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class LSTM(nn.Module):
    """
    An LSTM Network.
    NOTE: Like RNNs, LSTMs are for sequential data. In this single-step environment,
    we treat each state as a sequence of length 1.
    """

    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers, activation):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_hidden_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape input from [batch, features] to [batch, seq_len, features]
        x = x.unsqueeze(1)
        # We don't need the hidden/cell state outputs (h_n, c_n)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def _create_network(net_type: str, input_dim: int, output_dim: int, net_config: dict):
    """Factory function to create a network based on configuration."""
    net_map = {
        'mlp': MLP,
        'cnn': CNN,
        'rnn': RNN,
        'lstm': LSTM,
    }
    net_class = net_map.get(net_type.lower())
    if net_class is None:
        raise ValueError(f"Unknown network type: {net_type}")

    # Extract relevant arguments for the network constructor
    # This makes it robust to extra keys in net_config
    constructor_args = inspect.signature(net_class.__init__).parameters
    model_params = {
        'input_dim': input_dim,
        'output_dim': output_dim,
    }
    for arg in constructor_args:
        if arg in net_config:
            model_params[arg] = net_config[arg]

    return net_class(**model_params)


################################################################################
# 2. MODIFICATION: RolloutBuffer and ActorCritic are updated
################################################################################

class RolloutBuffer:
    """Stores transitions collected from the environment."""

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]


class ActorCritic(nn.Module):
    """
    MODIFIED: The neural network for Policy (Actor) and Value (Critic).
    It now dynamically creates networks based on the provided configuration.
    """

    def __init__(self, state_dim, action_dim, net_type, net_config):
        super(ActorCritic, self).__init__()

        # Create the actor (policy) network
        self.actor = _create_network(net_type, state_dim, action_dim, net_config)

        # Create the critic (value) network
        # Note: The critic always has an output dimension of 1.
        self.critic = _create_network(net_type, state_dim, 1, net_config)

        # Action variance parameter
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        raise NotImplementedError

    def act(self, state):
        """Perform a forward pass to get an action and its details."""
        action_mean = self.actor(state)
        action_std = self.action_log_std.exp().expand_as(action_mean)

        dist = Normal(action_mean, action_std)
        action = dist.sample()

        action_logprob = dist.log_prob(action).sum(dim=-1)
        state_value = self.critic(state)

        return action.detach(), action_logprob.detach(), state_value.detach()

    def evaluate(self, state, action):
        """Evaluate a state-action pair to get logprobs, value, and entropy."""
        action_mean = self.actor(state)
        action_std = self.action_log_std.exp().expand_as(action_mean)

        dist = Normal(action_mean, action_std)

        action_logprob = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_value = self.critic(state)

        return action_logprob, state_value, dist_entropy


class PPOAgent:
    """
    MODIFIED: The PPO Agent now accepts network configuration.
    """

    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, net_type, net_config):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()

        # Initialize policy and old policy with the specified network architecture
        self.policy = ActorCritic(state_dim, action_dim, net_type, net_config)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
            {'params': self.policy.action_log_std, 'lr': lr_actor}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, net_type, net_config)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        """Select an action from the old policy for rollout collection."""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.values.append(state_val)

        return action.squeeze(0).cpu().numpy()

    def update(self):
        """Update the policy using the collected rollout data."""
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.cat(self.buffer.states, dim=0).detach()
        old_actions = torch.cat(self.buffer.actions, dim=0).detach()
        old_logprobs = torch.cat(self.buffer.logprobs, dim=0).detach()

        # Squeeze values to ensure correct shape for advantage calculation
        old_values = torch.cat(self.buffer.values, dim=0).squeeze().detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Ensure state_values is squeezed to match rewards dimension
            state_values = state_values.squeeze()

            # Advantages calculated using old values to stabilize training
            advantages = rewards - old_values
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Combine actor and critic losses
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()


################################################################################
# 3. MODIFICATION: Main training function updated to accept net configs
################################################################################

def solve_with_simple_ppo(
        env: SingleStepOptimAntennaEnv,
        net_type: str = 'mlp',
        net_config: dict = None,
        total_timesteps: int = 20000,
        update_timestep: int = 200,
):
    """
    Solves the antenna optimization problem using a simple from-scratch PPO agent
    with a customizable network architecture.
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # --- Set default network configuration if none is provided ---
    if net_config is None:
        net_config = {
            'hidden_dim': 128,
            'num_hidden_layers': 2,
            'activation': 'tanh'  # 'relu' or 'tanh'
        }

    print(f"Using Network Type: {net_type.upper()}")
    print(f"Network Config: {net_config}")

    agent = PPOAgent(
        state_dim, action_dim,
        lr_actor=3e-4, lr_critic=1e-3,
        gamma=0.99, K_epochs=4, eps_clip=0.2,
        net_type=net_type,
        net_config=net_config
    )

    print(f"Running Simple PPO for {total_timesteps} timesteps...")
    time_step = 0

    with tqdm(total=total_timesteps, desc=f"Simple PPO Training ({net_type.upper()})") as pbar:
        while time_step < total_timesteps:
            # The inner loop collects a batch of experience
            for _ in range(update_timestep):
                state, _ = env.reset()
                action = agent.select_action(state)
                _, reward, terminated, _, _ = env.step(action)

                agent.buffer.rewards.append(reward)
                agent.buffer.is_terminals.append(terminated)
                time_step += 1
                pbar.update(1)

            agent.update()

    print("\nTraining complete. Evaluating the final policy...")

    state, _ = env.reset()
    state_tensor = torch.FloatTensor(state)

    with torch.no_grad():
        final_action = agent.policy.actor(state_tensor).cpu().numpy()

    _, final_reward, _, _, _ = env.step(final_action)

    # Extract the actual projected positions and weights from the environment
    actual_positions = env.x.copy()  # Positions in meters (already projected)
    actual_weights = env.weights.flatten().copy()  # Weights (already normalized)

    print(f"Final Action (raw): {final_action}")
    print(f"Final Reward: {final_reward}")
    print(f"Actual Positions (meters): {actual_positions}")
    print(f"Actual Weights: {actual_weights}")

    return final_action, final_reward, actual_positions, actual_weights


################################################################################
# 4. FIXED BASELINE: Maximum Ratio Combining (MRC) for Fixed Antenna Array
################################################################################

def compute_fixed_baseline(target_doas_deg, antenna_count, spacing_lambda=0.5):
    """
    Computes the optimal beam weights for a fixed uniform linear antenna array
    using Maximum Ratio Combining (MRC) - the classical optimal solution.

    Args:
        target_doas_deg: Array of target DOAs in degrees
        antenna_count: Number of antennas (N)
        spacing_lambda: Antenna spacing in wavelengths (default 0.5)

    Returns:
        dict with:
            - weights: Complex beam weights (normalized)
            - positions: Antenna positions in wavelengths
            - gain: Total gain at target DOAs
            - gains_at_targets: Individual gains at each target
            - beam_pattern: Full beam pattern for visualization
    """
    N = antenna_count
    target_angles_rad = np.deg2rad(target_doas_deg)

    # Fixed uniform linear array positions
    positions_lambda = np.arange(N) * spacing_lambda

    # Steering matrix for target DOAs
    A_target = np.exp(1j * np.outer(positions_lambda, 2 * np.pi * np.sin(target_angles_rad)))

    # MRC solution: sum steering vectors (maximize total gain)
    optimal_weights = np.sum(A_target, axis=1)
    optimal_weights = optimal_weights / np.linalg.norm(optimal_weights)  # Normalize

    # Compute gains at target DOAs
    gains_at_targets = np.abs(optimal_weights.conj() @ A_target) ** 2
    total_gain = np.sum(gains_at_targets)

    # Compute full beam pattern for visualization
    theta_scan = np.linspace(-90, 90, 361)
    theta_scan_rad = np.deg2rad(theta_scan)
    A_scan = np.exp(1j * np.outer(positions_lambda, 2 * np.pi * np.sin(theta_scan_rad)))
    beam_pattern = np.abs(optimal_weights.conj() @ A_scan) ** 2

    return {
        'weights': optimal_weights,
        'positions': positions_lambda,
        'gain': total_gain,
        'gains_at_targets': gains_at_targets,
        'beam_pattern': beam_pattern,
        'theta_scan': theta_scan
    }


if __name__ == '__main__':
    # --- MODIFIED: Main execution block to match user's new environment setup ---

    # 1. Define Environment Parameters
    FIXED_DOAS = np.array([-50.0, 0.0, 30.0])
    ANTENNA_COUNT = 8

    # 2. Compute Fixed Baseline (MRC) for comparison
    print("=" * 70)
    print("--- Computing Fixed Antenna Baseline (MRC) ---")
    start_time_fixed = time.time()
    fixed_baseline = compute_fixed_baseline(FIXED_DOAS, ANTENNA_COUNT)
    end_time_fixed = time.time()

    print(f"Fixed Baseline Results:")
    print(f"  Time taken: {end_time_fixed - start_time_fixed:.4f} seconds")
    print(f"  Total Gain: {fixed_baseline['gain']:.4f}")
    print(f"  Gains at targets: {fixed_baseline['gains_at_targets']}")
    print(f"  Weights (mag): {np.abs(fixed_baseline['weights'])}")
    print(f"  Weights (phase): {np.angle(fixed_baseline['weights'], deg=True)}")

    # 3. Create the environment
    print("\n" + "=" * 70)
    print("--- Initializing Environment ---")
    print(f"DOAs: {FIXED_DOAS}, Antenna Count: {ANTENNA_COUNT}")
    rl_env = SingleStepOptimAntennaEnv(uav_doas_deg=FIXED_DOAS, N=ANTENNA_COUNT)

    # 4. Run the PPO solver (movable antenna with RL)
    print("\n--- Running Method: Simple PPO (Movable Antenna) ---")
    start_time_rl = time.time()

    # You can still customize the network here if you want, for example:
    deep_mlp_config = {
        'hidden_dim': 64,
        'num_hidden_layers': 3,
        'activation': 'relu'
    }
    rl_solution, rl_gain, rl_positions_meters, rl_weights = solve_with_simple_ppo(
        rl_env,
        net_type='mlp',
        net_config=deep_mlp_config,
        total_timesteps=40000,
        update_timestep=200
    )

    # Using default MLP network
    # rl_solution, rl_gain, rl_positions_meters, rl_weights = solve_with_simple_ppo(
    #     rl_env, total_timesteps=40000, update_timestep=200
    # )

    end_time_rl = time.time()

    # 5. Convert actual positions from meters to wavelengths
    # These are the ACTUAL positions after constraint projection
    rl_positions_lambda = rl_positions_meters / rl_env.lam

    # Re-normalize weights to ensure ||w|| = 1 (should already be normalized, but double-check)
    rl_weights = rl_weights / np.linalg.norm(rl_weights)

    # Also verify and re-normalize fixed baseline weights
    fixed_weights = fixed_baseline['weights'] / np.linalg.norm(fixed_baseline['weights'])
    fixed_positions_lambda = fixed_baseline['positions']

    # Verify position constraints for movable antenna
    position_spacing = np.diff(rl_positions_lambda)
    min_spacing = np.min(position_spacing) if len(position_spacing) > 0 else np.inf

    # 6. Print constraints
    print("\n" + "=" * 70)
    print("--- CONSTRAINTS ---")
    print(f"Total aperture:       D = {rl_env.D_lambda:.2f}λ")
    print(f"Minimum spacing:      d_min = {rl_env.dmin_lambda:.2f}λ")
    print(f"Number of antennas:   N = {ANTENNA_COUNT}")
    print(f"Weight power:         ||w||² = 1 (normalized)")
    print(f"Position range:       [0, {rl_env.D_lambda:.2f}]λ")
    print("=" * 70)

    # 7. Print RL results with normalization check
    print("\n--- Simple PPO Results (Movable Antenna) ---")
    print(f"Time taken: {end_time_rl - start_time_rl:.2f} seconds")
    print(f"Best Gain Found: {rl_gain:.4f}")
    print(f"\nPositions (λ): {rl_positions_lambda}")
    print(f"Min spacing: {min_spacing:.4f}λ (constraint: ≥{rl_env.dmin_lambda:.2f}λ)")
    print(f"Max position: {np.max(rl_positions_lambda):.4f}λ (constraint: ≤{rl_env.D_lambda:.2f}λ)")
    print(f"\nWeights norm (should be 1.0): {np.linalg.norm(rl_weights):.6f}")
    print(f"Weights (mag): {np.abs(rl_weights)}")
    print(f"Weights (phase): {np.angle(rl_weights, deg=True)}")

    # 8. Re-verify fixed baseline normalization
    print("\n--- Fixed Baseline (Re-verified) ---")
    print(f"Positions (λ): {fixed_positions_lambda}")
    print(f"Spacing: uniform {rl_env.dmin_lambda:.2f}λ (minimum spacing)")
    print(f"\nWeights norm (should be 1.0): {np.linalg.norm(fixed_weights):.6f}")
    print(f"Weights (mag): {np.abs(fixed_weights)}")
    print(f"Weights (phase): {np.angle(fixed_weights, deg=True)}")

    # 9. Comparison
    print("\n" + "=" * 70)
    print("--- COMPARISON: Fixed vs Movable Antenna ---")
    print(f"\nGain Performance:")
    print(f"  Fixed Baseline Gain:  {fixed_baseline['gain']:.4f}")
    print(f"  Movable RL Gain:      {rl_gain:.4f}")
    gain_improvement = rl_gain - fixed_baseline['gain']
    gain_improvement_pct = (gain_improvement / fixed_baseline['gain']) * 100
    print(f"  Gain Improvement:     {gain_improvement:+.4f} ({gain_improvement_pct:+.2f}%)")

    print(f"\nArray Configuration:")
    print(f"  Fixed positions:   uniform spacing at {rl_env.dmin_lambda:.2f}λ")
    print(f"  Movable positions: optimized non-uniform spacing")

    print(f"\nWeight Normalization Check:")
    print(f"  Fixed weights norm:   {np.linalg.norm(fixed_weights):.6f}")
    print(f"  Movable weights norm: {np.linalg.norm(rl_weights):.6f}")

    print(f"\nConstraint Satisfaction:")
    constraint_satisfied = min_spacing >= rl_env.dmin_lambda and np.max(rl_positions_lambda) <= rl_env.D_lambda
    print(f"  Movable spacing constraint: {'✓ SATISFIED' if constraint_satisfied else '✗ VIOLATED'}")
    print(f"    Min spacing: {min_spacing:.4f}λ ≥ {rl_env.dmin_lambda:.2f}λ")
    print(f"    Max position: {np.max(rl_positions_lambda):.4f}λ ≤ {rl_env.D_lambda:.2f}λ")
    print("=" * 70)

    rl_env.close()

simple_ppo_solver.py
