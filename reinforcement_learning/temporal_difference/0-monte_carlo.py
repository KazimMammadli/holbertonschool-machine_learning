#!/usr/bin/env python3
"""
Module for Monte Carlo algorithm
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append([state, reward])
            if terminated or truncated:
                break
            state = next_state

        # Convert episode to numpy array for easier indexing if needed,
        # but iterating backwards is more operation-efficient.
        G = 0
        # Track states visited in this episode to ensure First-Visit logic
        episode_states = [step[0] for step in episode]
        
        for t in range(len(episode) - 1, -1, -1):
            state_t, reward_t = episode[t]
            G = gamma * G + reward_t
            
            # First-visit check: t is the first occurrence of state_t
            if state_t not in episode_states[:t]:
                V[state_t] = V[state_t] + alpha * (G - V[state_t])
                
    return V
