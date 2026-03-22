#!/usr/bin/env python3
"""
Module for Monte Carlo algorithm.
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to estimate the value function.

    Args:
        env: The gymnasium environment instance.
        V: numpy.ndarray of shape (s,) containing the value estimate.
        policy: Function that takes in a state and returns the next action.
        episodes: Total number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount rate.

    Returns:
        V: The updated value estimate.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        # Generate an episode
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            if terminated or truncated:
                break
            state = next_state

        # Convert episode to states and rewards
        states_in_episode = [step[0] for step in episode]
        rewards_in_episode = [step[1] for step in episode]

        G = 0
        # Iterate backwards through the episode to calculate returns
        visited_states = set()
        for t in reversed(range(len(episode))):
            s_t = states_in_episode[t]
            r_t = rewards_in_episode[t]
            G = gamma * G + r_t

            # First-visit Monte Carlo: only update if it's the 
            # first time the state is seen in this episode
            if s_t not in states_in_episode[:t]:
                V[s_t] = V[s_t] + alpha * (G - V[s_t])

    return V
