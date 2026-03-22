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
    """

    def epsilon_greedy(state, eps):
        if np.random.uniform() < eps:
            return np.random.randint(Q.shape[1])
        return np.argmax(Q[state])

    for _ in range(episodes):
        state, _ = env.reset()
        eps = epsilon

        action = epsilon_greedy(state, eps)

        E = np.zeros_like(Q)

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)

            if not (terminated or truncated):
                next_action = epsilon_greedy(next_state, eps)


            if terminated or truncated:
                delta = reward - Q[state, action]
            else:
                delta = reward + gamma * Q[next_state, next_action] - Q[state, action]

            E[state, action] = 1

            # Update
            Q += alpha * delta * E

            # Decay traces
            E *= gamma * lambtha

            if terminated or truncated:
                break

            state = next_state
            action = next_action

        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q