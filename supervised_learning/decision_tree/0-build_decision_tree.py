#!/usr/bin/env python3
"""Decision tree - max depth."""
import numpy as np


class Node:
    """Node class representing an internal decision point in the tree."""

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None,
                 is_root=False, depth=0):
        """Initialize a Node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
    """Return maximum depth of all descendants below this node."""
    if self.is_leaf:
        return self.depth

    left = self.left_child.max_depth_below() if self.left_child else self.depth
    right = self.right_child.max_depth_below() if self.right_child else self.depth
    
    return max(left, right)


class Leaf(Node):
    """Leaf class representing a terminal node in the tree."""

    def __init__(self, value, depth=None):
        """Initialize a Leaf node."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of this leaf node."""
        return self.depth


class DecisionTree:
    """Decision tree class."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize a DecisionTree."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return maximum depth of the tree."""
        return self.root.max_depth_below()

