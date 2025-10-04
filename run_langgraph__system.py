import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import List, Dict, TypedDict, Any, Optional

# LangChain & LangGraph Imports
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import PPO solver components
from simple_ppo_solver import solve_with_simple_ppo
from env.antenna_env import SingleStepOptimAntennaEnv

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# --- 1. Define Tools for the Agents (Tools remain the same) ---

@tool
def estimate_doas_from_csi_file(csi_file_path: str, K: int) -> Dict:
    """
    Loads a CSI data file (.npy), estimates the DOAs of K UAVs using the MUSIC algorithm.
    """
    try:
        Y = np.load(csi_file_path)
        N, T = Y.shape
        if K <= 0 or K >= N:
            return {"error": f"Number of UAVs K={K} must be between 1 and N-1={N - 1}."}

        R = (Y @ Y.conj().T) / T
        _, eigvecs = np.linalg.eigh(R)
        En = eigvecs[:, :N - K]

        angles_grid = np.arange(-90, 90.5, 0.5)
        positions_lambda = np.arange(N) * 0.5

        thetas_rad = np.deg2rad(angles_grid)
        A = np.exp(1j * np.outer(positions_lambda, 2 * np.pi * np.sin(thetas_rad)))

        EnH_A = En.conj().T @ A
        denom = np.sum(np.abs(EnH_A) ** 2, axis=0).real
        P_music = 1.0 / np.maximum(denom, 1e-12)

        from scipy.signal import find_peaks
        peaks, _ = find_peaks(P_music, height=np.max(P_music) * 0.1, distance=5)
        if len(peaks) < K:
            peaks = np.argsort(P_music)[-K:]

        strongest_peaks = peaks[np.argsort(P_music[peaks])[-K:]]
        estimated_doas = sorted(angles_grid[strongest_peaks].tolist())

        return {"estimated_doas": estimated_doas}
    except Exception as e:
        return {"error": f"An error occurred during DOA estimation: {str(e)}"}


@tool
def train_ppo_directly(
    estimated_doas: List[float],
    antenna_count: int,
    net_type: str = "mlp",
    net_config: Optional[Dict[str, Any]] = None,
    total_timesteps: int = 20000,
    update_timestep: int = 200
) -> Dict:
    """
    Directly trains a PPO agent using the simple_ppo_solver module.
    Returns the final gain and optimized weights.
    """
    try:
        # Set default net_config if not provided
        if net_config is None:
            net_config = {
                'hidden_dim': 128,
                'num_hidden_layers': 2,
                'activation': 'tanh'
            }

        # Create environment
        env = SingleStepOptimAntennaEnv(uav_doas_deg=np.array(estimated_doas), N=antenna_count)

        # Train with PPO
        print(f"Training PPO with {net_type} network on DOAs: {estimated_doas}")
        final_action, final_reward, actual_positions_meters, actual_weights = solve_with_simple_ppo(
            env=env,
            net_type=net_type,
            net_config=net_config,
            total_timesteps=total_timesteps,
            update_timestep=update_timestep
        )

        # Convert positions from meters to wavelengths
        positions_lambda = (actual_positions_meters / env.lam).tolist()

        # Convert complex weights to string format
        weights_str = [str(w) for w in actual_weights]

        result = {
            "gain": float(final_reward),
            "weights_complex_str": weights_str,
            "positions_lambda": positions_lambda
        }

        env.close()

        return {"training_results": result, "stdout": json.dumps(result)}
    except Exception as e:
        import traceback
        return {"error": f"Failed to train PPO: {str(e)}\n{traceback.format_exc()}"}


@tool
def compute_fixed_antenna_baseline(target_doas: List[float], antenna_count: int, spacing_lambda: float = 0.5) -> Dict:
    """
    Computes the optimal beam weights for a fixed antenna array with uniform spacing.
    Uses Maximum Ratio Combining (MRC) to maximize signal at target DOAs.

    Args:
        target_doas: List of target Direction of Arrivals in degrees
        antenna_count: Number of antennas in the array
        spacing_lambda: Antenna spacing in wavelengths (default: 0.5 for half-wavelength)

    Returns:
        Dictionary containing optimal weights and achieved gain
    """
    try:
        N = antenna_count
        K = len(target_doas)

        # Create uniform linear array positions
        positions_lambda = np.arange(N) * spacing_lambda

        # Compute steering vectors for target DOAs
        target_angles_rad = np.deg2rad(target_doas)
        A_target = np.exp(1j * np.outer(positions_lambda, 2 * np.pi * np.sin(target_angles_rad)))

        # Use MRC: weights = sum of steering vectors (equal gain combining for all targets)
        optimal_weights = np.sum(A_target, axis=1)

        # Normalize weights
        optimal_weights = optimal_weights / np.linalg.norm(optimal_weights)

        # Compute gain at target DOAs
        gains_at_targets = np.abs(optimal_weights.conj() @ A_target) ** 2
        total_gain = np.sum(gains_at_targets)

        # Convert weights to string format
        weights_str = [str(w) for w in optimal_weights]

        result = {
            "method": "Fixed Antenna (MRC)",
            "spacing_lambda": spacing_lambda,
            "positions_lambda": positions_lambda.tolist(),
            "weights_complex_str": weights_str,
            "gain": float(total_gain),
            "individual_gains": gains_at_targets.tolist()
        }

        print(f"✅ Fixed Antenna Baseline: Gain = {total_gain:.4f}")
        return result

    except Exception as e:
        import traceback
        return {"error": f"Failed to compute fixed antenna baseline: {str(e)}\n{traceback.format_exc()}"}


@tool
def save_optimization_results(
    movable_weights_str: List[str],
    movable_gain: float,
    movable_positions_lambda: List[float],
    fixed_weights_str: List[str],
    fixed_gain: float,
    fixed_positions_lambda: List[float],
    target_doas: List[float],
    antenna_count: int,
    net_type: str = "mlp",
    net_config: Optional[Dict[str, Any]] = None
) -> Dict:
    """
    Saves the optimization results to local files and generates a deployment README.

    Args:
        movable_weights_str: Optimized weights from movable antenna
        movable_gain: Gain achieved by movable antenna
        movable_positions_lambda: Optimized positions from movable antenna (in wavelengths)
        fixed_weights_str: Weights from fixed antenna baseline
        fixed_gain: Gain achieved by fixed antenna
        fixed_positions_lambda: Positions from fixed antenna baseline (in wavelengths)
        target_doas: Target DOA angles
        antenna_count: Number of antennas
        net_type: Neural network type used
        net_config: Neural network configuration

    Returns:
        Dictionary with file paths and README content
    """
    try:
        import json
        from datetime import datetime

        # Set default net_config if not provided
        if net_config is None:
            net_config = {
                'hidden_dim': 128,
                'num_hidden_layers': 2,
                'activation': 'tanh'
            }

        # Convert weights to complex numbers
        movable_weights = np.array([complex(w) for w in movable_weights_str])
        fixed_weights = np.array([complex(w) for w in fixed_weights_str])

        # Use actual positions passed as parameters
        movable_positions = np.array(movable_positions_lambda)
        fixed_positions = np.array(fixed_positions_lambda)

        # Prepare results dictionary
        results = {
            "timestamp": datetime.now().isoformat(),
            "target_doas_deg": target_doas,
            "antenna_count": antenna_count,
            "movable_antenna": {
                "positions_lambda": movable_positions.tolist(),
                "weights_real": movable_weights.real.tolist(),
                "weights_imag": movable_weights.imag.tolist(),
                "weights_complex_str": movable_weights_str,
                "gain": float(movable_gain),
                "method": "PPO Reinforcement Learning",
                "net_type": net_type,
                "net_config": net_config
            },
            "fixed_antenna": {
                "positions_lambda": fixed_positions.tolist(),
                "weights_real": fixed_weights.real.tolist(),
                "weights_imag": fixed_weights.imag.tolist(),
                "weights_complex_str": fixed_weights_str,
                "gain": float(fixed_gain),
                "method": "Maximum Ratio Combining (MRC)"
            },
            "performance": {
                "gain_improvement_percent": ((movable_gain - fixed_gain) / fixed_gain) * 100,
                "movable_vs_fixed_ratio": movable_gain / fixed_gain
            }
        }

        # Save results to JSON file
        results_file = "optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save weights to numpy files for easy loading
        np.save("movable_weights.npy", movable_weights)
        np.save("fixed_weights.npy", fixed_weights)
        np.save("movable_positions.npy", movable_positions)
        np.save("fixed_positions.npy", fixed_positions)

        # Generate deployment README
        gain_improvement = ((movable_gain - fixed_gain) / fixed_gain) * 100
        readme_content = f"""# Antenna Array Optimization Results

## Summary
This document contains the results of antenna array optimization for UAV communication.

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Problem Configuration
- **Number of Antennas:** {antenna_count}
- **Target UAV DOAs:** {target_doas} degrees
- **Number of UAVs:** {len(target_doas)}

## Optimization Methods

### 1. Movable Antenna (RL-based Optimization)
- **Algorithm:** Proximal Policy Optimization (PPO)
- **Network Type:** {net_type.upper()}
- **Network Config:** {net_config}
- **Final Gain:** {movable_gain:.6f}
- **Optimized Weights:** Saved to `movable_weights.npy`
- **Antenna Positions:** Saved to `movable_positions.npy`

### 2. Fixed Antenna (Baseline)
- **Algorithm:** Maximum Ratio Combining (MRC)
- **Antenna Spacing:** {spacing_lambda}λ (wavelengths)
- **Final Gain:** {fixed_gain:.6f}
- **Optimal Weights:** Saved to `fixed_weights.npy`
- **Antenna Positions:** Saved to `fixed_positions.npy`

## Performance Comparison
- **Gain Improvement:** {gain_improvement:+.2f}%
- **Movable/Fixed Ratio:** {movable_gain/fixed_gain:.4f}x

## Visualization
- Comparison plot: `comparison_beam_pattern.png`

## Files Generated
1. `optimization_results.json` - Complete optimization results in JSON format
2. `movable_weights.npy` - Movable antenna optimized weights (complex numpy array)
3. `fixed_weights.npy` - Fixed antenna optimal weights (complex numpy array)
4. `movable_positions.npy` - Movable antenna positions in wavelengths
5. `fixed_positions.npy` - Fixed antenna positions in wavelengths
6. `comparison_beam_pattern.png` - Visualization comparing both methods

## How to Load Results

```python
import numpy as np
import json

# Load weights
movable_weights = np.load('movable_weights.npy')
fixed_weights = np.load('fixed_weights.npy')

# Load positions
movable_positions = np.load('movable_positions.npy')
fixed_positions = np.load('fixed_positions.npy')

# Load complete results
with open('optimization_results.json', 'r') as f:
    results = json.load(f)
```

## Conclusion
The {net_type.upper()}-based movable antenna optimization achieved a **{gain_improvement:+.2f}% improvement**
over the traditional fixed antenna array with MRC beamforming.

---
*This README was automatically generated by the LangGraph antenna optimization system.*
"""

        # Save README
        readme_file = "DEPLOYMENT_README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)

        print(f"✅ Results saved to {results_file}")
        print(f"✅ README saved to {readme_file}")
        print(f"✅ Weight files: movable_weights.npy, fixed_weights.npy")
        print(f"✅ Position files: movable_positions.npy, fixed_positions.npy")

        return {
            "results_file": results_file,
            "readme_file": readme_file,
            "readme_content": readme_content,
            "weight_files": ["movable_weights.npy", "fixed_weights.npy"],
            "position_files": ["movable_positions.npy", "fixed_positions.npy"]
        }

    except Exception as e:
        import traceback
        return {"error": f"Failed to save optimization results: {str(e)}\n{traceback.format_exc()}"}


@tool
def compare_and_plot_results(
    movable_weights_str: List[str],
    movable_gain: float,
    fixed_weights_str: List[str],
    fixed_gain: float,
    target_doas: List[float],
    antenna_count: int,
    spacing_lambda: float = 0.5
) -> Dict:
    """
    Compares movable and fixed antenna results and generates comparison plots.

    Args:
        movable_weights_str: Weights from movable antenna optimization
        movable_gain: Gain achieved by movable antenna
        fixed_weights_str: Weights from fixed antenna baseline
        fixed_gain: Gain achieved by fixed antenna
        target_doas: Target DOA angles in degrees
        antenna_count: Number of antennas
        spacing_lambda: Antenna spacing in wavelengths

    Returns:
        Dictionary with comparison results and plot path
    """
    try:
        # Convert weights
        movable_weights = np.array([complex(w) for w in movable_weights_str])
        fixed_weights = np.array([complex(w) for w in fixed_weights_str])

        # Fixed antenna positions (uniform spacing)
        fixed_positions = np.arange(antenna_count) * spacing_lambda

        # Compute beam patterns
        angles_deg = np.linspace(-90, 90, 361)
        thetas_rad = np.deg2rad(angles_deg)

        # Fixed antenna beam pattern
        A_fixed = np.exp(1j * np.outer(fixed_positions, 2 * np.pi * np.sin(thetas_rad)))
        gain_fixed = np.abs(fixed_weights.conj() @ A_fixed) ** 2
        gain_fixed_db = 10 * np.log10(gain_fixed / np.max(gain_fixed))

        # Movable antenna beam pattern (assuming optimized positions are embedded in the gain)
        # For comparison, we use the same fixed positions but with optimized weights
        gain_movable = np.abs(movable_weights.conj() @ A_fixed) ** 2
        gain_movable_db = 10 * np.log10(gain_movable / np.max(gain_movable))

        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Beam patterns comparison
        ax1.plot(angles_deg, gain_fixed_db, 'b-', label=f'Fixed Antenna (Gain={fixed_gain:.4f})', linewidth=2)
        ax1.plot(angles_deg, gain_movable_db, 'r-', label=f'Movable Antenna (Gain={movable_gain:.4f})', linewidth=2)
        for doa in target_doas:
            ax1.axvline(x=doa, color='g', linestyle='--', alpha=0.7, label=f'Target DOA: {doa}°')
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())
        ax1.set_title("Beam Pattern Comparison: Movable vs Fixed Antenna", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Angle (degrees)", fontsize=12)
        ax1.set_ylabel("Normalized Gain (dB)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-40, 5)

        # Plot 2: Gain improvement bar chart
        gain_improvement = ((movable_gain - fixed_gain) / fixed_gain) * 100
        ax2.bar(['Fixed Antenna', 'Movable Antenna'], [fixed_gain, movable_gain],
                color=['blue', 'red'], alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel("Total Gain", fontsize=12)
        ax2.set_title(f"Gain Comparison (Improvement: {gain_improvement:+.2f}%)", fontsize=14, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (gain, name) in enumerate([(fixed_gain, 'Fixed'), (movable_gain, 'Movable')]):
            ax2.text(i, gain, f'{gain:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plot_path = "comparison_beam_pattern.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

        result = {
            "plot_path": plot_path,
            "fixed_gain": fixed_gain,
            "movable_gain": movable_gain,
            "gain_improvement_percent": gain_improvement,
            "comparison_summary": f"Movable antenna achieves {gain_improvement:+.2f}% gain improvement over fixed antenna"
        }

        print(f"✅ Comparison complete: Movable gain = {movable_gain:.4f}, Fixed gain = {fixed_gain:.4f}, Improvement = {gain_improvement:+.2f}%")
        return result

    except Exception as e:
        import traceback
        return {"error": f"Failed to compare and plot results: {str(e)}\n{traceback.format_exc()}"}


@tool
def monitor_doa_drift(
    csi_data_path: str,
    original_doas: List[float],
    antenna_count: int,
    monitoring_iteration: int = 0,
    drift_threshold_degrees: float = 3.0,
    current_movable_weights: Optional[List[str]] = None,
    current_movable_positions: Optional[List[float]] = None
) -> Dict:
    """
    Continuously monitors CSI data to detect DOA drift caused by UAV movement.
    Reads CSI files from a sequence based on monitoring_iteration.
    Computes beam gains for both movable and fixed antenna configurations.

    Args:
        csi_data_path: Base path to CSI data (used to determine sequence directory)
        original_doas: The DOA angles used during training
        antenna_count: Number of antennas
        monitoring_iteration: Current monitoring iteration (determines which CSI file to read)
        drift_threshold_degrees: Maximum allowable DOA drift before re-optimization
        current_movable_weights: Current movable antenna weights (complex string format)
        current_movable_positions: Current movable antenna positions (in wavelengths)

    Returns:
        Dictionary with drift detection results, beam gains, and recommendation
    """
    try:
        from tool.doa_est import estimate_doas_music

        # Determine which CSI file to read based on monitoring iteration
        # monitoring_iteration=0 -> read csi_t1.npy (first monitoring after deployment)
        # monitoring_iteration=1 -> read csi_t2.npy (second monitoring)
        # t=0 was used for initial training
        time_step = monitoring_iteration + 1
        csi_seq_dir = "csi_sequence"
        current_csi_file = f"{csi_seq_dir}/csi_t{time_step}.npy"

        # Check if file exists
        if not os.path.exists(current_csi_file):
            return {
                "status": "completed",
                "no_more_data": True,
                "message": f"All CSI data processed. No more data available at {current_csi_file}.",
                "monitoring_complete": True
            }

        # Load current CSI data
        print(f"📡 Loading CSI data from: {current_csi_file}")
        Y = np.load(current_csi_file)
        N, T = Y.shape

        if N != antenna_count:
            return {"error": f"CSI data has {N} antennas but expected {antenna_count}"}

        # Create antenna positions for DOA estimation
        spacing_lambda = 0.5
        positions_lambda = np.arange(N) * spacing_lambda

        # Estimate current DOAs from new CSI data
        K = len(original_doas)
        current_doas = estimate_doas_music(Y, positions_lambda, K)

        # Calculate drift for each UAV
        # Match current DOAs to original DOAs (find closest pairs)
        original_sorted = sorted(original_doas)
        current_sorted = sorted(current_doas)

        doa_drifts = [abs(curr - orig) for curr, orig in zip(current_sorted, original_sorted)]
        max_drift = max(doa_drifts)
        avg_drift = np.mean(doa_drifts)

        # Determine if re-optimization is needed
        drift_detected = max_drift > drift_threshold_degrees

        # ============ Compute Beam Gains ============
        # Both fixed and movable antennas use weights computed from ORIGINAL DOAs
        # This allows fair comparison of performance degradation when DOAs drift

        spacing_lambda = 0.5
        fixed_positions_lambda = np.arange(N) * spacing_lambda

        # Convert DOAs to radians
        original_doas_rad = np.deg2rad(original_doas)
        current_doas_rad = np.deg2rad(current_doas)

        # 1. Fixed Antenna Baseline (MRC with uniform spacing)
        # Compute fixed antenna weights based on ORIGINAL DOAs (training time)
        A_fixed_original = np.exp(1j * np.outer(fixed_positions_lambda, 2 * np.pi * np.sin(original_doas_rad)))
        fixed_weights = np.sum(A_fixed_original, axis=1)
        fixed_weights = fixed_weights / np.linalg.norm(fixed_weights)

        # Compute steering matrix for CURRENT DOAs to evaluate performance
        A_fixed_current = np.exp(1j * np.outer(fixed_positions_lambda, 2 * np.pi * np.sin(current_doas_rad)))

        # Compute fixed antenna beam gain (original weights applied to current DOAs)
        fixed_gain = float(np.sum(np.abs(fixed_weights.conj() @ A_fixed_current) ** 2))

        # 2. Movable Antenna Beam Gain (if weights and positions provided)
        # Use training-time optimized weights and positions
        movable_gain = None
        if current_movable_weights is not None and current_movable_positions is not None:
            # Convert weights from string format to complex
            movable_weights_complex = np.array([complex(w) for w in current_movable_weights])
            movable_weights_complex = movable_weights_complex / np.linalg.norm(movable_weights_complex)

            # Use provided positions
            movable_positions_lambda = np.array(current_movable_positions)

            # Compute steering matrix for CURRENT DOAs
            A_movable_current = np.exp(1j * np.outer(movable_positions_lambda, 2 * np.pi * np.sin(current_doas_rad)))

            # Compute movable antenna beam gain (training weights applied to current DOAs)
            movable_gain = float(np.sum(np.abs(movable_weights_complex.conj() @ A_movable_current) ** 2))

        # Check if movable antenna performs worse than fixed antenna
        performance_degraded = False
        degradation_reason = ""
        if movable_gain is not None and movable_gain < fixed_gain:
            performance_degraded = True
            degradation_percent = ((fixed_gain - movable_gain) / fixed_gain) * 100
            degradation_reason = f"Movable gain ({movable_gain:.4f}) < Fixed gain ({fixed_gain:.4f}), degradation: {degradation_percent:.2f}%"

        # Overall drift detection: either DOA drift OR performance degradation
        overall_drift_detected = drift_detected or performance_degraded

        # Determine drift reason
        drift_reasons = []
        if drift_detected:
            drift_reasons.append(f"DOA drift: {max_drift:.2f}° > {drift_threshold_degrees}°")
        if performance_degraded:
            drift_reasons.append(degradation_reason)

        drift_message = "; ".join(drift_reasons) if drift_reasons else f"Max DOA drift: {max_drift:.2f}° (threshold: {drift_threshold_degrees}°)"

        result = {
            "status": "monitoring",
            "time_step": time_step,
            "csi_file": current_csi_file,
            "original_doas": original_doas,
            "current_doas": current_doas,
            "doa_drifts": doa_drifts,
            "max_drift_degrees": float(max_drift),
            "avg_drift": float(avg_drift),
            "drift_threshold_degrees": drift_threshold_degrees,
            "drift_detected": overall_drift_detected,
            "doa_drift_detected": drift_detected,
            "performance_degraded": performance_degraded,
            "fixed_antenna_gain": fixed_gain,
            "movable_antenna_gain": movable_gain if movable_gain is not None else "N/A",
            "recommendation": "re-optimize" if overall_drift_detected else "continue",
            "message": f"Time t={time_step}: {drift_message}"
        }

        print(f"🔍 DEBUG - Tool returning result:")
        print(f"   original_doas in result: {result['original_doas']}")
        print(f"   current_doas in result: {result['current_doas']}")
        print(f"   fixed_antenna_gain in result: {result['fixed_antenna_gain']}")
        print(f"   movable_antenna_gain in result: {result['movable_antenna_gain']}")

        if overall_drift_detected:
            print(f"⚠️  Re-optimization Triggered at t={time_step}!")
            if drift_detected:
                print(f"   [DOA Drift] Original DOAs: {[f'{d:.1f}°' for d in original_doas]}")
                print(f"   [DOA Drift] Current DOAs:  {[f'{d:.1f}°' for d in current_doas]}")
                print(f"   [DOA Drift] Drifts:        {[f'{d:.1f}°' for d in doa_drifts]}")
                print(f"   [DOA Drift] Max drift: {max_drift:.2f}° > {drift_threshold_degrees}°")
            if performance_degraded:
                print(f"   [Performance] {degradation_reason}")
            print(f"   → Recommendation: RE-OPTIMIZE")
        else:
            print(f"✅ t={time_step}: System performing well")
            print(f"   DOA drift: {max_drift:.2f}° < {drift_threshold_degrees}°")
            print(f"   Movable gain: {movable_gain:.4f}, Fixed gain: {fixed_gain:.4f}")
            print(f"   → Recommendation: CONTINUE MONITORING")

        return result

    except Exception as e:
        import traceback
        return {"error": f"Failed to monitor DOA drift: {str(e)}\n{traceback.format_exc()}"}


# --- 2. Define Graph State ---
# --- MODIFICATION: Enhanced state to support supervisor LLM agent ---
class AgentState(TypedDict):
    # User input
    user_request: str  # Natural language task description
    constraints: Dict[str, Any]  # Task-specific constraints (e.g., spacing, aperture)
    available_tools: List[str]  # List of available tool names

    # System parameters
    k_uavs: int
    antenna_count: int
    csi_data_path: str

    # Agent outputs
    estimated_doas: List[float]
    model_description: str
    model_config: Dict[str, Any]
    training_code: str
    training_results: Dict[str, Any]
    fixed_baseline_results: Dict[str, Any]
    code_execution_logs: str
    evaluation_report: Dict
    deployment_readme: str
    monitoring_report: Dict[str, Any]

    # Workflow control
    task_history: List[str]
    retries: int
    supervisor_decision: Dict[str, Any]  # Supervisor's decision: {next_agent, instruction}

    # Monitoring and re-optimization control
    optimization_cycle: int  # Number of optimization cycles completed
    max_optimization_cycles: int  # Maximum allowed cycles to prevent infinite loop
    monitoring_iterations: int  # Number of monitoring checks performed
    drift_detected: bool  # Whether DOA drift was detected
    all_data_processed: bool  # Whether all CSI data files have been processed


# --- 3. Define Agent Nodes ---
# --- MODIFICATION: Generic agent that receives instructions from supervisor ---
def create_agent_node(llm, system_prompt: str, tools: list, agent_name: str):
    """Factory function to create a generic agent node."""
    agent = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        """The actual node function that will be executed in the graph."""
        print(f"\n--- Entering {agent_name} Agent ---")

        # Debug: print state info for monitoring agent
        if agent_name == "monitoring":
            print(f"🔍 DEBUG - State info for monitoring:")
            print(f"   estimated_doas: {state.get('estimated_doas', 'NOT FOUND')}")
            print(f"   monitoring_iterations: {state.get('monitoring_iterations', 'NOT FOUND')}")
            print(f"   training_results keys: {list(state.get('training_results', {}).keys())}")
            if state.get('training_results'):
                print(f"   training_results.weights_complex_str: {state.get('training_results', {}).get('weights_complex_str', 'NOT FOUND')[:100] if state.get('training_results', {}).get('weights_complex_str') else 'NONE'}...")
                print(f"   training_results.positions_lambda: {state.get('training_results', {}).get('positions_lambda', 'NOT FOUND')}")

        # Get supervisor's instruction for this agent (if available)
        supervisor_instruction = state.get(f"{agent_name}_instruction", "")

        # Debug: print supervisor instruction for monitoring
        if agent_name == "monitoring":
            print(f"📋 Supervisor instruction for monitoring:")
            print(f"   {supervisor_instruction[:500]}...")

        # Build prompt: system role + supervisor instruction + current state
        prompt = f"""{system_prompt}

**Supervisor Instruction:**
{supervisor_instruction}

**Current State:**
{state}

Execute the instruction using the available tools.
"""
        response = agent.invoke(prompt)
        new_history = state.get('task_history', []) + [agent_name]
        updates = {"task_history": new_history}

        if not response.tool_calls:
            # Handle model_selection agent's direct response
            agent_output = response.content
            updates[f"{agent_name}_output"] = agent_output
            if agent_name == "model_selection":
                print("\n" + "="*70)
                print("MODEL SELECTION AGENT OUTPUT:")
                print("="*70)
                print(agent_output)
                print("="*70)

                try:
                    # Try to extract JSON from the response
                    config_text = agent_output

                    # Try to extract JSON from markdown code blocks if present
                    if "```json" in config_text:
                        print("ℹ️  Detected markdown JSON block, extracting...")
                        config_text = config_text.split("```json")[1].split("```")[0]
                    elif "```" in config_text:
                        print("ℹ️  Detected markdown code block, extracting...")
                        config_text = config_text.split("```")[1].split("```")[0]

                    config_data = json.loads(config_text.strip())

                    # Validate the config has required fields
                    if "net_type" not in config_data:
                        raise ValueError("Missing 'net_type' in config")
                    if "net_config" not in config_data:
                        raise ValueError("Missing 'net_config' in config")

                    # Validate net_type is valid
                    valid_net_types = ['mlp', 'cnn', 'rnn', 'lstm']
                    if config_data['net_type'].lower() not in valid_net_types:
                        raise ValueError(f"Invalid net_type '{config_data['net_type']}'. Must be one of {valid_net_types}")

                    # Validate net_config has required fields
                    required_config_fields = ['hidden_dim', 'num_hidden_layers', 'activation']
                    missing_fields = [f for f in required_config_fields if f not in config_data['net_config']]
                    if missing_fields:
                        raise ValueError(f"Missing fields in net_config: {missing_fields}")

                    updates["model_config"] = config_data
                    updates["model_description"] = f"Selected model: {config_data.get('net_type', 'N/A').upper()}"
                    print(f"\n✅ Model Selection VALID and PARSED successfully!")
                    print(f"   Net Type: {config_data['net_type']}")
                    print(f"   Net Config: {config_data['net_config']}")

                except json.JSONDecodeError as e:
                    print(f"\n❌ JSON PARSING FAILED!")
                    print(f"   Error: {str(e)}")
                    print(f"   Attempted to parse: {config_text[:200]}...")
                    updates[f"{agent_name}_output"] = {"error": f"JSON parsing failed: {str(e)}"}
                    updates["model_config"] = {}

                except ValueError as e:
                    print(f"\n❌ CONFIG VALIDATION FAILED!")
                    print(f"   Error: {str(e)}")
                    print(f"   Parsed JSON: {config_data if 'config_data' in locals() else 'N/A'}")
                    updates[f"{agent_name}_output"] = {"error": f"Config validation failed: {str(e)}"}
                    updates["model_config"] = {}

                except Exception as e:
                    print(f"\n❌ UNEXPECTED ERROR in model selection!")
                    print(f"   Error type: {type(e).__name__}")
                    print(f"   Error message: {str(e)}")
                    updates[f"{agent_name}_output"] = {"error": f"Unexpected error: {str(e)}"}
                    updates["model_config"] = {}
        else:
            # Handle agents that call tools
            tool_outputs = {}
            for tool_call in response.tool_calls:
                tool_name, tool_args = tool_call['name'], tool_call['args']
                print(f"Agent wants to call tool: {tool_name}")
                print(f"Tool arguments: {json.dumps(tool_args, indent=2)[:500]}...")
                tool_func = next((t for t in tools if t.name == tool_name), None)
                if tool_func:
                    output = tool_func.invoke(tool_args)
                    tool_outputs[tool_name] = output
                else:
                    tool_outputs[tool_name] = {"error": f"Tool '{tool_name}' not found."}
            updates[f"{agent_name}_output"] = tool_outputs

            # --- MODIFICATION START ---
            # Special handling for different agents' tool outputs
            if agent_name == "data_collection":
                tool_result = tool_outputs.get("estimate_doas_from_csi_file", {})
                if "estimated_doas" in tool_result:
                    updates["estimated_doas"] = tool_result["estimated_doas"]
                    print(f"✅ Data Collection successful. DOAs: {tool_result['estimated_doas']}")

            elif agent_name == "training":
                # Get training results directly from the tool
                tool_result = tool_outputs.get("train_ppo_directly", {})
                if "training_results" in tool_result:
                    updates["training_results"] = tool_result["training_results"]
                    print(f"✅ Training successful. Final Gain: {tool_result['training_results'].get('gain', 'N/A')}")

            elif agent_name == "evaluation":
                # Save fixed baseline results if computed
                fixed_baseline_result = tool_outputs.get("compute_fixed_antenna_baseline", {})
                if fixed_baseline_result and "gain" in fixed_baseline_result:
                    updates["fixed_baseline_results"] = fixed_baseline_result
                    print(f"✅ Fixed baseline computed. Gain: {fixed_baseline_result.get('gain', 'N/A')}")

                # Save evaluation report from comparison
                comparison_result = tool_outputs.get("compare_and_plot_results", {})
                if comparison_result and "comparison_summary" in comparison_result:
                    updates["evaluation_report"] = comparison_result
                    print(f"✅ Evaluation complete: {comparison_result.get('comparison_summary', 'N/A')}")

            elif agent_name == "deployment":
                # Save deployment README
                deploy_result = tool_outputs.get("save_optimization_results", {})
                if deploy_result and "readme_content" in deploy_result:
                    updates["deployment_readme"] = deploy_result["readme_content"]
                    print(f"✅ Deployment files saved: {', '.join(deploy_result.get('weight_files', []))}")

            elif agent_name == "monitoring":
                # Handle monitoring results
                monitor_result = tool_outputs.get("monitor_doa_drift", {})
                print(f"🔍 DEBUG - monitor_result received by agent_node:")
                print(f"   Keys: {list(monitor_result.keys()) if monitor_result else 'EMPTY'}")
                if "error" in monitor_result:
                    print(f"❌ ERROR in monitoring tool:")
                    print(f"   {monitor_result['error']}")
                else:
                    print(f"   original_doas: {monitor_result.get('original_doas', 'NOT FOUND')}")
                    print(f"   current_doas: {monitor_result.get('current_doas', 'NOT FOUND')}")
                if monitor_result:
                    updates["monitoring_report"] = monitor_result

                    # Check if all data has been processed
                    if monitor_result.get("no_more_data", False):
                        updates["all_data_processed"] = True
                        print(f"✅ All CSI data has been processed. Workflow will end.")
                        print(f"   {monitor_result.get('message', '')}")
                    else:
                        drift_detected_now = monitor_result.get("drift_detected", False)
                        updates["drift_detected"] = drift_detected_now
                        updates["monitoring_iterations"] = state.get("monitoring_iterations", 0) + 1

                        # If drift detected, save the current CSI file path for re-optimization
                        if drift_detected_now:
                            updates["csi_data_path"] = monitor_result.get("csi_file", "csi_data.npy")
                            print(f"💾 Saved current CSI file for re-optimization: {updates['csi_data_path']}")
                        else:
                            # If monitoring is stable and we're in a re-optimization cycle, reset the cycle counter
                            current_cycle = state.get("optimization_cycle", 0)
                            if current_cycle > 0:
                                updates["optimization_cycle"] = 0
                                print(f"✅ Re-optimization cycle {current_cycle} completed successfully. Resetting cycle counter to 0.")

                    # Display monitoring results (only if there's actual monitoring data)
                    if not monitor_result.get("no_more_data", False):
                        print("\n" + "="*70)
                        print(f"📊 MONITORING ITERATION {updates['monitoring_iterations']} RESULTS")
                        print("="*70)

                        # Show estimated DOAs
                        current_doas = monitor_result.get("current_doas", [])
                        original_doas = monitor_result.get("original_doas", [])
                        print(f"📍 Original DOAs: {original_doas}")
                        print(f"📍 Current DOAs:  {current_doas}")

                        # Show beam gains
                        movable_gain = monitor_result.get("movable_antenna_gain", "N/A")
                        fixed_gain = monitor_result.get("fixed_antenna_gain", "N/A")
                        print(f"\n📡 Beam Gain Performance:")
                        print(f"   Movable Antenna: {movable_gain}")
                        print(f"   Fixed Antenna:   {fixed_gain}")

                        # Show drift status
                        max_drift = monitor_result.get("max_drift_degrees", 0.0)
                        drift_threshold = monitor_result.get("drift_threshold_degrees", 3.0)
                        doa_drift = monitor_result.get("doa_drift_detected", False)
                        perf_degraded = monitor_result.get("performance_degraded", False)

                        print(f"\n🔍 Analysis:")
                        print(f"   DOA Drift: {max_drift:.2f}° (threshold: {drift_threshold:.2f}°) - {'⚠️ EXCEEDED' if doa_drift else '✅ OK'}")

                        if isinstance(movable_gain, (int, float)) and isinstance(fixed_gain, (int, float)):
                            gain_comparison = "⚠️ WORSE" if perf_degraded else "✅ BETTER"
                            print(f"   Performance: Movable vs Fixed - {gain_comparison}")

                        if monitor_result.get("drift_detected", False):
                            trigger_reasons = []
                            if doa_drift:
                                trigger_reasons.append("DOA drift")
                            if perf_degraded:
                                trigger_reasons.append("Performance degradation")
                            print(f"\n   ⚠️  RE-OPTIMIZATION TRIGGERED")
                            print(f"   Reason(s): {', '.join(trigger_reasons)}")
                        else:
                            print(f"\n   ✅ System stable - Continue monitoring")
                        print("="*70 + "\n")
            # --- MODIFICATION END ---
        return updates

    return agent_node


# --- 4. Define the Supervisor Agent (LLM-based coordinator) ---
MAX_RETRIES = 2


def create_supervisor_agent(llm):
    """Create a supervisor agent that uses LLM to make routing decisions."""

    def supervisor_node(state: AgentState):
        """
        Supervisor agent analyzes the state and decides:
        1. Which agent to route to next
        2. What specific instruction to give that agent
        """
        print("\n" + "="*70)
        print("--- SUPERVISOR AGENT: Analyzing state and making decision ---")
        print("="*70)

        history = state.get('task_history', [])
        user_request = state.get('user_request', '')
        constraints = state.get('constraints', {})
        available_tools = state.get('available_tools', [])

        # Get monitoring and cycle control info
        drift_detected = state.get('drift_detected', False)
        optimization_cycle = state.get('optimization_cycle', 0)
        max_optimization_cycles = state.get('max_optimization_cycles', 3)
        monitoring_iterations = state.get('monitoring_iterations', 0)
        all_data_processed = state.get('all_data_processed', False)

        # Flags for state updates
        clear_history_flag = False
        new_optimization_cycle = None

        # Determine the required workflow sequence
        WORKFLOW_SEQUENCE = [
            "data_collection",
            "model_selection",
            "training",
            "evaluation",
            "deployment",
            "monitoring"
        ]

        # Special handling for monitoring loop and re-optimization
        last_agent = history[-1] if history else None

        # Check if all CSI data has been processed (end condition)
        if all_data_processed:
            print(f"✅ All CSI data processed. Ending workflow.")
            next_required_agent = "END"
        # If monitoring detected drift, restart the cycle
        elif last_agent == "monitoring" and drift_detected:
            if optimization_cycle >= max_optimization_cycles:
                print(f"⚠️  Max optimization cycles ({max_optimization_cycles}) reached. Ending workflow.")
                next_required_agent = "END"
            else:
                print(f"🔄 DOA drift detected. Starting re-optimization cycle {optimization_cycle + 1}/{max_optimization_cycles}")
                print(f"🔄 Clearing task history to restart workflow from data_collection")
                # Clear history to restart workflow - will be returned in updates
                next_required_agent = "data_collection"
                # Set flag to clear history in the return statement
                clear_history_flag = True
                new_optimization_cycle = optimization_cycle + 1
        # If monitoring didn't detect drift, continue monitoring
        elif last_agent == "monitoring" and not drift_detected:
            print(f"📊 Monitoring iteration {monitoring_iterations}. Continuing to monitor...")
            next_required_agent = "monitoring"
        else:
            # Normal workflow progression
            next_required_agent = None
            for agent in WORKFLOW_SEQUENCE:
                if agent not in history:
                    next_required_agent = agent
                    break

            # If all stages completed, should end
            if next_required_agent is None:
                next_required_agent = "END"

        # Build supervisor prompt with workflow sequence constraint
        supervisor_prompt = f"""You are a Supervisor Agent coordinating a multi-agent antenna optimization workflow with continuous monitoring.

**User Request:**
{user_request}

**Constraints:**
{json.dumps(constraints, indent=2)}

**Available Tools:**
{', '.join(available_tools)}

**Workflow Sequence (MUST follow this order):**
1. data_collection → 2. model_selection → 3. training → 4. evaluation → 5. deployment → 6. monitoring
   - If monitoring detects DOA drift → restart from data_collection (re-optimization)
   - If monitoring is stable → continue monitoring (loop)
   - Maximum {max_optimization_cycles} optimization cycles allowed

**Current State:**
- Optimization Cycle: {optimization_cycle}/{max_optimization_cycles}
- Monitoring Iterations: {monitoring_iterations}
- Drift Detected: {drift_detected}
- Task History: {history}
- Completed Agents: {', '.join(history) if history else 'None'}
- Next Required Agent: {next_required_agent}
- Current CSI Data Path: {state.get('csi_data_path', 'csi_data.npy')}
- Estimated DOAs: {state.get('estimated_doas', [])}
- Training Results Available: {'weights_complex_str' in state.get('training_results', {})}
- Training Weights: {state.get('training_results', {}).get('weights_complex_str', 'N/A')}
- Training Positions (λ): {state.get('training_results', {}).get('positions_lambda', 'N/A')}

**Last Agent Output:**
{json.dumps(state.get(f"{history[-1]}_output", {}) if history else {}, indent=2)[:500]}...

**Your Task:**
You MUST follow the workflow sequence. The next agent to call is: **{next_required_agent}**

Generate a specific instruction for {next_required_agent} based on:
1. The user's request
2. The constraints provided
3. The outputs from previous agents
4. The available tools

**IMPORTANT:** You cannot skip stages. You must call "{next_required_agent}" next.

**Special Instructions for data_collection Agent:**
If the next agent is "data_collection", you MUST instruct it to use the current CSI data path from state:
- csi_file_path: {state.get('csi_data_path', 'csi_data.npy')}
- K (number of UAVs): {state.get('k_uavs', 3)}
IMPORTANT: During re-optimization (optimization_cycle > 0), use the updated CSI file path that reflects the current monitoring data, NOT the original csi_data.npy.

**Special Instructions for model_selection Agent:**
If the next agent is "model_selection", your instruction MUST explicitly state:
"Output ONLY a JSON object with no extra text. The JSON must have exactly these fields:
- net_type: one of ['mlp', 'cnn', 'rnn', 'lstm']
- net_config: a dictionary with 'hidden_dim', 'num_hidden_layers', 'activation'
Example: {{"net_type": "mlp", "net_config": {{"hidden_dim": 128, "num_hidden_layers": 2, "activation": "tanh"}}}}"

**Special Instructions for monitoring Agent:**
If the next agent is "monitoring", you MUST instruct it to call monitor_doa_drift with these exact parameters:
- csi_data_path: "csi_data.npy"
- original_doas: {state.get('estimated_doas', [])} (from state)
- antenna_count: {state.get('antenna_count', 8)}
- monitoring_iteration: {monitoring_iterations}
- drift_threshold_degrees: 3.0
- current_movable_weights: {state.get('training_results', {}).get('weights_complex_str', [])} (from training_results in state)
- current_movable_positions: {state.get('training_results', {}).get('positions_lambda', [])} (from training_results in state)

Your instruction must tell the agent to use these EXACT values from the state, not placeholders.

**Output Format (JSON only):**
{{
    "next_agent": "{next_required_agent}",
    "reasoning": "brief explanation of what this agent needs to do",
    "instruction": "specific detailed instruction for {next_required_agent}"
}}
"""

        # Call LLM to make decision
        response = llm.invoke(supervisor_prompt)

        try:
            # Parse supervisor's decision
            decision_text = response.content
            # Try to extract JSON from markdown code blocks if present
            if "```json" in decision_text:
                decision_text = decision_text.split("```json")[1].split("```")[0]
            elif "```" in decision_text:
                decision_text = decision_text.split("```")[1].split("```")[0]

            decision = json.loads(decision_text.strip())

            # Validate that supervisor followed the workflow sequence
            decided_agent = decision.get('next_agent', 'END')
            if decided_agent != next_required_agent:
                print(f"⚠️  WARNING: Supervisor tried to skip to '{decided_agent}', but workflow requires '{next_required_agent}'")
                print(f"   Overriding decision to enforce workflow sequence.")
                decision['next_agent'] = next_required_agent
                decision['reasoning'] = f"[Enforced] {decision.get('reasoning', 'Following workflow sequence')}"

            print(f"✅ Supervisor Decision:")
            print(f"   Next Agent: {decision.get('next_agent', 'UNKNOWN')}")
            print(f"   Reasoning: {decision.get('reasoning', 'N/A')}")
            instruction_preview = str(decision.get('instruction', 'N/A'))[:100]
            print(f"   Instruction: {instruction_preview}...")

            # Update state with supervisor's decision
            next_agent = decision.get('next_agent', 'END')
            instruction = decision.get('instruction', '')

            if next_agent != 'END' and next_agent != END:
                # Inject instruction for the next agent
                state[f"{next_agent}_instruction"] = instruction

            # Build return updates
            updates = {'supervisor_decision': decision}

            # If re-optimization is triggered, clear history and update cycle
            if clear_history_flag:
                updates['task_history'] = []
                updates['optimization_cycle'] = new_optimization_cycle
                updates['drift_detected'] = False
                print(f"🔄 Supervisor returning updates: task_history=[], optimization_cycle={new_optimization_cycle}")

            return updates

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse supervisor decision: {e}")
            print(f"Raw response: {response.content}")
            # Fallback: use the required next agent
            fallback_decision = {
                'next_agent': next_required_agent,
                'reasoning': 'Fallback due to parse error',
                'instruction': f'Please proceed with {next_required_agent} stage.'
            }
            if next_required_agent != 'END' and next_required_agent != END:
                state[f"{next_required_agent}_instruction"] = fallback_decision['instruction']

            # Build fallback updates
            fallback_updates = {'supervisor_decision': fallback_decision}

            # If re-optimization is triggered, clear history and update cycle
            if clear_history_flag:
                fallback_updates['task_history'] = []
                fallback_updates['optimization_cycle'] = new_optimization_cycle
                fallback_updates['drift_detected'] = False
                print(f"🔄 Supervisor (fallback) returning updates: task_history=[], optimization_cycle={new_optimization_cycle}")

            return fallback_updates

    return supervisor_node


def supervisor_router(state: AgentState) -> str:
    """
    Simple router that extracts the supervisor's decision from state.
    The actual decision-making is done by the supervisor_agent node.
    """
    decision = state.get('supervisor_decision', {})
    next_agent = decision.get('next_agent', 'END')

    # Handle error retry logic
    history = state.get('task_history', [])
    if history:
        last_agent = history[-1]
        last_output = state.get(f"{last_agent}_output", {})

        error_in_output = False
        if isinstance(last_output, dict):
            for tool_result in last_output.values():
                if isinstance(tool_result, dict) and "error" in tool_result:
                    error_in_output = True
                    break

        if error_in_output and last_agent == "training" and state.get('retries', 0) < MAX_RETRIES:
            state['retries'] = state.get('retries', 0) + 1
            print(f"⚠️  Error detected, retrying training (Attempt {state['retries']}/{MAX_RETRIES})")
            return "training"

    print(f"→ Routing to: {next_agent}")
    return next_agent if next_agent != 'END' else END




# --- 5. Tool Registry for Task-Specific Tools ---
TOOL_REGISTRY = {
    "movable_antenna_optimization": {
        "data_collection": [estimate_doas_from_csi_file],
        "model_selection": [],
        "training": [train_ppo_directly],
        "evaluation": [compute_fixed_antenna_baseline, compare_and_plot_results],
        "deployment": [save_optimization_results],
        "monitoring": [monitor_doa_drift]  # Continuous monitoring tool
    },
    # Future tasks can be added here
    # "phase_shift_optimization": {
    #     "data_collection": [different_tool],
    #     "training": [different_training_tool],
    #     ...
    # }
}


def get_tools_for_task(task_type: str, agent_name: str) -> list:
    """Get the appropriate tools for a given task and agent."""
    if task_type in TOOL_REGISTRY:
        return TOOL_REGISTRY[task_type].get(agent_name, [])
    else:
        # Default to movable antenna optimization tools
        return TOOL_REGISTRY["movable_antenna_optimization"].get(agent_name, [])


# --- 6. Build and Run the Graph ---
if __name__ == '__main__':
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    # Dummy files for the environment code to be written into
    env_dir = "env"
    if not os.path.exists(env_dir): os.makedirs(env_dir)

    # Create CSI sequence directory
    csi_seq_dir = "csi_sequence"
    if not os.path.exists(csi_seq_dir):
        os.makedirs(csi_seq_dir)

    # --- Generate CSI Data Sequence Simulating UAV Movement ---
    print("\n" + "="*70)
    print("Generating CSI Data Sequence (Simulating UAV Movement)")
    print("="*70)

    N, T, K = 8, 1024, 3
    initial_doas = [-50.0, 0.0, 30.0]

    # Define UAV movement scenario (10 time steps)
    csi_scenarios = [
        # (time_step, DOAs, drift_from_initial, description)
        (0, [-50.0, 0.0, 30.0], 0.0, "Initial deployment - Training data"),
        (1, [-50.0, 0.0, 30.0], 0.0, "Monitoring t=1: Stable position"),
        (2, [-49.5, 0.5, 30.5], 0.7, "Monitoring t=2: Slight movement"),
        (3, [-49.0, 1.0, 31.0], 1.4, "Monitoring t=3: Continued drift"),
        (4, [-47.0, 3.0, 33.0], 4.2, "Monitoring t=4: Moderate drift"),
        (5, [-45.0, 5.0, 35.0], 7.1, "Monitoring t=5: SIGNIFICANT DRIFT! (>5°) - Trigger re-optimization"),
        (6, [-45.0, 5.0, 35.0], 0.0, "Monitoring t=6: New stable position after re-opt"),
        (7, [-44.0, 6.0, 36.0], 1.4, "Monitoring t=7: Minor adjustment"),
        (8, [-42.0, 8.0, 38.0], 3.6, "Monitoring t=8: Gradual drift"),
        # (9, [-38.0, 12.0, 42.0], 8.5, "Monitoring t=9: LARGE DRIFT! (>5°) - Trigger 2nd re-opt"),
    ]

    # Generate and save CSI data for each time step
    for t, doas, max_drift, description in csi_scenarios:
        # Generate array steering matrix
        pos = np.arange(N) * 0.5  # Fixed positions for CSI generation
        A = np.exp(1j * np.outer(pos, 2 * np.pi * np.sin(np.deg2rad(doas))))

        # Generate source signals
        np.random.seed(42 + t)  # Reproducible random seed
        S = (np.random.randn(K, T) + 1j * np.random.randn(K, T)) / np.sqrt(2)

        # Generate received signal with noise (SNR = 10 dB)
        noise_power = 0.1
        N_noise = np.sqrt(noise_power / 2) * (np.random.randn(N, T) + 1j * np.random.randn(N, T))
        Y = (A @ S) + N_noise

        # Save CSI data
        filename = f"{csi_seq_dir}/csi_t{t}.npy"
        np.save(filename, Y)

        # Print scenario info
        status_icon = "⚠️ DRIFT!" if max_drift > 5.0 else "✅"
        print(f"{status_icon} t={t}: DOAs={[f'{d:+6.1f}°' for d in doas]}, "
              f"Max Drift={max_drift:.1f}° - {description}")

    # Also save the initial CSI as the main training data
    np.save("csi_data.npy", np.load(f"{csi_seq_dir}/csi_t0.npy"))
    print(f"\n✅ Generated {len(csi_scenarios)} CSI files in '{csi_seq_dir}/' directory")
    print(f"✅ Initial training data saved as 'csi_data.npy'")
    print("="*70 + "\n")

    # Determine task type (can be passed as argument or detected from user request)
    TASK_TYPE = "movable_antenna_optimization"
    print(f"\n{'='*70}")
    print(f"TASK TYPE: {TASK_TYPE}")
    print(f"{'='*70}\n")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

    # --- Create Generic Agent Nodes with Task-Specific Tools ---
    # All agents have generic system prompts; specific instructions come from supervisor
    # Tools are dynamically assigned based on TASK_TYPE

    data_collection_node = create_agent_node(
        llm,
        "You are a data collection agent. Execute the supervisor's instructions using the provided tools.",
        get_tools_for_task(TASK_TYPE, "data_collection"),
        "data_collection"
    )

    model_selection_node = create_agent_node(
        llm,
        """You are a model selection agent. Follow the supervisor's instructions.

        When selecting neural network architectures, consider:
        - **MLP**: General-purpose, suitable for flat vectors
        - **CNN**: For spatial/grid data (e.g., images)
        - **RNN/LSTM**: For sequential data (e.g., time series)

        Output format: JSON with `net_type` and `net_config` keys.
        Example: {"net_type": "mlp", "net_config": {"hidden_dim": 128, "num_hidden_layers": 2, "activation": "tanh"}}
        """,
        get_tools_for_task(TASK_TYPE, "model_selection"),
        "model_selection"
    )

    training_node = create_agent_node(
        llm,
        "You are a training agent. Execute the supervisor's training instructions using the provided tools.",
        get_tools_for_task(TASK_TYPE, "training"),
        "training"
    )

    evaluation_node = create_agent_node(
        llm,
        "You are an evaluation agent. Execute the supervisor's evaluation instructions using the provided tools.",
        get_tools_for_task(TASK_TYPE, "evaluation"),
        "evaluation"
    )

    deployment_node = create_agent_node(
        llm,
        "You are a deployment agent. Execute the supervisor's deployment instructions using the provided tools.",
        get_tools_for_task(TASK_TYPE, "deployment"),
        "deployment"
    )

    monitoring_node = create_agent_node(
        llm,
        "You are a monitoring agent. Execute the supervisor's monitoring instructions.",
        get_tools_for_task(TASK_TYPE, "monitoring"),
        "monitoring"
    )

    # --- Create Supervisor Agent Node ---
    supervisor_node = create_supervisor_agent(llm)

    # --- Define Graph ---
    graph = StateGraph(AgentState)

    # Add supervisor node
    graph.add_node("supervisor", supervisor_node)

    # Add worker agent nodes
    graph.add_node("data_collection", data_collection_node)
    graph.add_node("model_selection", model_selection_node)
    graph.add_node("training", training_node)
    graph.add_node("evaluation", evaluation_node)
    graph.add_node("deployment", deployment_node)
    graph.add_node("monitoring", monitoring_node)

    # Set supervisor as entry point
    graph.set_entry_point("supervisor")

    # After supervisor makes decision, route to appropriate agent
    graph.add_conditional_edges("supervisor", supervisor_router)

    # After each agent completes, return to supervisor for next decision
    graph.add_edge("data_collection", "supervisor")
    graph.add_edge("model_selection", "supervisor")
    graph.add_edge("training", "supervisor")
    graph.add_edge("evaluation", "supervisor")
    graph.add_edge("deployment", "supervisor")
    graph.add_edge("monitoring", "supervisor")

    app = graph.compile()

    # --- Run the Graph ---
    # User input with rich task description
    initial_state = {
        # User's natural language input
        "user_request": """I want to upgrade my antenna system from fixed to movable antenna array.
The goal is to optimize both antenna positions and beamforming weights to maximize signal gain from multiple UAVs.

Current setup:
- Number of antennas: 8
- Number of UAV targets: 3
- I have CSI (Channel State Information) data available

Please help me:
1. Estimate the DOA (Direction of Arrival) angles from CSI
2. Design an optimal movable antenna array using reinforcement learning
3. Compare the performance with traditional fixed antenna array
4. Save the results for deployment
        """,

        # Task constraints
        "constraints": {
            "antenna_count": N,
            "minimum_spacing_lambda": 0.4,
            "total_aperture_lambda": 10.0,
            "k_uavs": K,
            "optimization_method": "PPO (Proximal Policy Optimization)",
            "training_timesteps": 20000
        },

        # Available tools (for supervisor's reference)
        "available_tools": [
            "estimate_doas_from_csi_file",
            "train_ppo_directly",
            "compute_fixed_antenna_baseline",
            "compare_and_plot_results",
            "save_optimization_results",
            "monitor_doa_drift"  # Continuous monitoring
        ],

        # System parameters
        "k_uavs": K,
        "antenna_count": N,
        "csi_data_path": "csi_data.npy",

        # Initialize state fields
        "task_history": [],
        "retries": 0,
        "estimated_doas": [],
        "model_description": "",
        "model_config": {},
        "training_code": "",
        "training_results": {},
        "fixed_baseline_results": {},
        "code_execution_logs": "",
        "evaluation_report": {},
        "deployment_readme": "",
        "monitoring_report": {},
        "supervisor_decision": {},

        # Monitoring and re-optimization control
        "optimization_cycle": 0,
        "max_optimization_cycles": 3,  # Maximum 3 re-optimization cycles
        "monitoring_iterations": 0,
        "drift_detected": False,
        "all_data_processed": False
    }

    print("\n\n" + "=" * 50, "\n      STARTING FULLY AUTOMATED WORKFLOW\n" + "=" * 50)

    # Stream the workflow execution and capture final state
    final_state = None
    for s in app.stream(initial_state, {"recursion_limit": 80}):
        print(s)
        print("---")
        # Update final_state with the latest state
        final_state = s

    print("\n\n" + "=" * 50, "\n      WORKFLOW COMPLETE - FINAL STATE\n" + "=" * 50)
    print(f"Final task history: {final_state.get(list(final_state.keys())[0], {}).get('task_history', [])}")

    # Extract the actual state from the last stream output
    if final_state:
        state_key = list(final_state.keys())[0]
        actual_final_state = final_state[state_key]
        print("\n" + "=" * 50, "\n      DEPLOYMENT README\n" + "=" * 50)
        print(actual_final_state.get('deployment_readme', 'No README was generated.'))
