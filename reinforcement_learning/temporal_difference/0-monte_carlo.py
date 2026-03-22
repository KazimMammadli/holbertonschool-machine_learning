#!/usr/bin/env python3
"""
Monte Carlo algorithm for state-value function estimation.
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to update value estimates.

    Parameters:
    env (gym.Env): environment instance
    V (numpy.ndarray): value estimates of shape (s,)
    policy (function): maps state -> action
    episodes (int): number of episodes
    max_steps (int): max steps per episode
    alpha (float): learning rate
    gamma (float): discount factor

    Returns:
    numpy.ndarray: updated value estimates
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        # Generate episode
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode.append((state, reward))

            state = next_state

            if terminated or truncated:
                break

        # Compute returns and update V
        G = 0
        for state, reward in reversed(episode):
            G = gamma * G + reward
            V[state] += alpha * (G - V[state])

    return V