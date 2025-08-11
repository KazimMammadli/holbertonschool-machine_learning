#!/usr/bin/env python3
"""This module defines bars function."""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Show bars for diff people."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    fruits = ["apples", "bananas", "oranges", "peaches"]
    people = ["Farrah", "Fred", "Felicia"]
    colors = ["red", "yellow", "#ff8000", "#ffe5b4"]

    fruit.reshape(4, 3)
    x = np.arange(len(people))
    bottom = np.zeros(len(people))
    for i in range(len(fruits)):
        plt.bar(x, fruit[i], bottom=bottom,
                color=colors[i], width=0.5,
                label=fruits[i])
        bottom += fruit[i]
    plt.title("Number of Fruit per Person")
    plt.yticks(np.arange(0, 81, 10))
    plt.xticks(x, people)
    plt.ylabel("Quantity of Fruit")
    plt.legend()
    plt.show()
