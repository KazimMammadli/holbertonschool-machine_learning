#!/usr/bin/env python3
"""
Module to have a trained agent play an episode
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode and returns the
    rendered board at each step.

    Args:
        env: the FrozenLakeEnv instance
        Q: a numpy.ndarray containing the Q-table
        max_steps: the maximum number of steps in the episode

    Returns:
        total_reward, rendered_outputs
    """
    state, _ = env.reset()
    # Capture the initial state of the board
    rendered_outputs = [env.render()]
    total_reward = 0

    for _ in range(max_steps):
        # Always exploit: pick the best action according to Q-table
        action = np.argmax(Q[state])

        # Step the environment
        state, reward, terminated, truncated, _ = env.step(action)

        # Capture the new state of the board
        rendered_outputs.append(env.render())

        total_reward += reward

        if terminated or truncated:
            break

    return total_reward, rendered_outputs
