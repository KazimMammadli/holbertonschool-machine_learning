#!/usr/bin/env python3
"""
SARSA(lambda) algorithm implementation.
"""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(lambda) algorithm.

    Parameters:
    env (gym.Env): environment instance
    Q (np.ndarray): Q-table of shape (s, a)
    lambtha (float): eligibility trace factor
    episodes (int): number of episodes
    max_steps (int): max steps per episode
    alpha (float): learning rate
    gamma (float): discount factor
    epsilon (float): initial epsilon
    min_epsilon (float): minimum epsilon
    epsilon_decay (float): decay rate

    Returns:
    np.ndarray: updated Q-table
    """

    def epsilon_greedy(state):
        """Select action using epsilon-greedy policy."""
        if np.random.uniform() < epsilon:
            return np.random.randint(Q.shape[1])
        return np.argmax(Q[state])

    for _ in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(state)

        # Eligibility traces
        E = np.zeros_like(Q)

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy(next_state)

            # TD error
            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]

            # Increase eligibility
            E[state, action] += 1

            # Update all Q values
            Q += alpha * delta * E

            # Decay eligibility traces
            E *= gamma * lambtha

            state = next_state
            action = next_action

            if terminated or truncated:
                break

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q
