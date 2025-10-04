# From Agentification to Self-Evolving Agentic AI for Wireless Networks: Concepts, Approaches, and Research Directions

This is the official implementation of *From Agentification to Self-Evolving Agentic AI for Wireless Networks: Concepts, Approaches, and Research Directions*.

## LangGraph Multi-Agent Antenna Optimization System

A sophisticated multi-agent system using LangGraph for adaptive antenna array optimization with continuous monitoring and automatic self-evolution.

## Features

- 🤖 **Multi-Agent Architecture**: Supervisor-coordinated workflow with specialized agents
- 📡 **DOA Estimation**: MUSIC algorithm for Direction of Arrival estimation
- 🎯 **PPO Optimization**: Reinforcement learning-based antenna position and weight optimization
- 📊 **Continuous Monitoring**: Real-time performance tracking with drift detection
- 🔄 **Auto Self-evolution**: Automatic self-evolution on DOA drift or performance degradation
- 📈 **Performance Comparison**: Built-in baseline comparison with fixed antenna arrays

## Workflow Stages

1. **Data Collection**: Estimate DOAs from CSI data using MUSIC algorithm
2. **Model Selection**: Choose neural network architecture for PPO
3. **Training**: Optimize antenna positions and beamforming weights using PPO
4. **Evaluation**: Compare with fixed antenna baseline (MRC)
5. **Deployment**: Save optimized configuration
6. **Monitoring**: Continuous DOA drift and performance monitoring

## Re-optimization Triggers

The system automatically triggers re-optimization when:
- **DOA Drift**: `max_drift > 3.0°`
- **Performance Degradation**: `movable_gain < fixed_gain`
