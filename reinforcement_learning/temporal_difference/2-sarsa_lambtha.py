#!/usr/bin/env python3
"""Module for SARSA(λ) reinforcement learning algorithm."""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Perform SARSA(λ) to update a Q table.

    Args:
        env: The environment instance.
        Q: numpy.ndarray of shape (s, a) containing the Q table.
        lambtha: The eligibility trace factor.
        episodes: Total number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount rate.
        epsilon: Initial threshold for epsilon greedy.
        min_epsilon: Minimum value that epsilon should decay to.
        epsilon_decay: Decay rate for updating epsilon between episodes.

    Returns:
        Q: The updated Q table.
    """
    init_epsilon = epsilon

    for ep in range(episodes):
        state, _ = env.reset()
        E = np.zeros_like(Q)

        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if np.random.uniform() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])

            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            E[state, action] += 1
            Q += alpha * delta * E
            E *= gamma * lambtha

            if done:
                break
            state, action = next_state, next_action

        epsilon = (min_epsilon + (init_epsilon - min_epsilon)
                   * np.exp(-epsilon_decay * ep))

    return Q