#!/usr/bin/env python3
"""SARSA(lambda) algorithm for Q-table estimation"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Performs the SARSA(lambda) algorithm to update a Q table.

    Args:
        env: openAI environment instance
        Q (numpy.ndarray): shape (s, a) containing the Q table
        lambtha (float): eligibility trace factor
        episodes (int): total number of episodes to train over
        max_steps (int): maximum number of steps per episode
        alpha (float): learning rate
        gamma (float): discount rate
        epsilon (float): initial threshold for epsilon greedy
        min_epsilon (float): minimum value epsilon should decay to
        epsilon_decay (float): decay rate for updating epsilon per episode

    Returns:
        Q (numpy.ndarray): the updated Q table
    """
    n_states, n_actions = Q.shape

    def epsilon_greedy(state, eps):
        """Select action using epsilon-greedy policy."""
        if np.random.uniform() < eps:
            return env.action_space.sample()
        return np.argmax(Q[state])

    for _ in range(episodes):
        state = env.reset()
        action = epsilon_greedy(state, epsilon)
        Et = np.zeros_like(Q)

        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(next_state, epsilon)

            delta = (reward + gamma * Q[next_state, next_action] * (1 - done)
                     - Q[state, action])
            Et[state, action] += 1

            Q += alpha * delta * Et
            Et *= gamma * lambtha

            state = next_state
            action = next_action
            if done:
                break

        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q
