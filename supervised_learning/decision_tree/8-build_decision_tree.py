#!/usr/bin/env python3
"""Decision tree implementation using Gini impurity for splitting."""
import numpy as np


class Node:
    """Represents an internal node in the decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
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
        """Return the maximum depth below this node."""
        if self.is_leaf:
            return self.depth
        left = (self.left_child.max_depth_below()
                if self.left_child else self.depth)
        right = (self.right_child.max_depth_below()
                 if self.right_child else self.depth)
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        """Count nodes below this node. Count only leaves if specified."""
        if self.is_leaf:
            return 1
        if only_leaves:
            left = (self.left_child.count_nodes_below(True)
                    if self.left_child else 0)
            right = (self.right_child.count_nodes_below(True)
                     if self.right_child else 0)
            return left + right
        else:
            left = (self.left_child.count_nodes_below(False)
                    if self.left_child else 0)
            right = (self.right_child.count_nodes_below(False)
                     if self.right_child else 0)
            return 1 + left + right

    def get_leaves_below(self):
        """Return a list of all leaves below this node."""
        if self.is_leaf:
            return [self]
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """Update feature bounds for this node and all children recursively."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}
        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.upper = self.upper.copy()
                child.lower = self.lower.copy()
                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                else:
                    child.upper[self.feature] = self.threshold
        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def left_child_add_prefix(self, text):
        """Format left child in ASCII tree representation."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Format right child in ASCII tree representation."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Return an ASCII representation of this node and its children."""
        label = f"root [feature={self.feature}, threshold={self.threshold}]" \
                if self.is_root else f"node [feature={self.feature},\
                                             threshold={self.threshold}]"
        result = label
        if self.left_child:
            result += "\n" + self.left_child_add_prefix(
                str(self.left_child)
            ).rstrip("\n")
        if self.right_child:
            result += "\n" + self.right_child_add_prefix(
                str(self.right_child)
            ).rstrip("\n")
        return result

    def update_indicator(self):
        """Create a function that identifies which samples belong to node."""
        def is_large_enough(x):
            if not self.lower:
                return np.ones(x.shape[0], dtype=bool)
            conditions = []
            for key in self.lower.keys():
                if key < x.shape[1]:
                    conditions.append(np.greater(x[:, key], self.lower[key]))
                else:
                    conditions.append(np.ones(x.shape[0], dtype=bool))
            return np.all(np.array(conditions), axis=0) if conditions else \
                np.ones(x.shape[0], dtype=bool)

        def is_small_enough(x):
            if not self.upper:
                return np.ones(x.shape[0], dtype=bool)
            conditions = []
            for key in self.upper.keys():
                if key < x.shape[1]:
                    conditions.append(np.less_equal(x[:, key],
                                                    self.upper[key]))
                else:
                    conditions.append(np.ones(x.shape[0], dtype=bool))
            return np.all(np.array(conditions), axis=0) if conditions else \
                np.ones(x.shape[0], dtype=bool)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """Predict class for a single sample."""
        return self.left_child.pred(x) if x[self.feature] > self.threshold \
            else self.right_child.pred(x)


class Leaf(Node):
    """Leaf node containing a predicted class value."""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1

    def get_leaves_below(self):
        return [self]

    def update_bounds_below(self):
        pass

    def __str__(self):
        return f"-> leaf [value={self.value}]"

    def pred(self, x):
        return self.value


class Decision_Tree:
    """Decision tree supporting random or Gini-based splits."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize the decision tree."""
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        return self.root.get_leaves_below()

    def update_bounds(self):
        self.root.update_bounds_below()

    def update_predict(self):
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_func(A):
            n_individuals = A.shape[0]
            predictions = np.zeros(n_individuals, dtype=int)
            for leaf in leaves:
                mask = leaf.indicator(A)
                predictions[mask] = leaf.value
            return predictions

        self.predict = predict_func

    def pred(self, x):
        return self.root.pred(x)

    def np_extrema(self, arr):
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def possible_thresholds(self, node, feature):
        values = np.unique(self.explanatory[:, feature][node.sub_population])
        if values.size < 2:
            return np.array([])
        return (values[1:] + values[:-1]) / 2
