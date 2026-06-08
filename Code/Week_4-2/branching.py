from abc import ABC, abstractmethod

import numpy as np


class BranchingStrategy(ABC):
    """
    Base interface for branching rules.
    """

    @abstractmethod
    def select(self, lp_solution: np.ndarray) -> int:
        pass


class MostFractionalBranching(BranchingStrategy):
    """
    Classical branching rule.

    Choose the variable closest to 0.5.
    """

    def select(self, lp_solution: np.ndarray) -> int:

        fractional_vars = []

        for i, value in enumerate(lp_solution):

            if 1e-6 < value < 1 - 1e-6:
                fractional_vars.append(i)

        if not fractional_vars:
            return -1

        return min(
            fractional_vars,
            key=lambda i: abs(lp_solution[i] - 0.5)
        )
        
class MWUABranching(BranchingStrategy):

    def __init__(self, mwua_certainty):
        self.mwua_certainty = mwua_certainty

    def select(self, lp_solution):

        fractional = [
            i
            for i, x in enumerate(lp_solution)
            if 1e-6 < x < 1 - 1e-6
        ]

        if not fractional:
            return -1

        return max(
            fractional,
            key=lambda i:
            self.mwua_certainty[i]
        )
        
import numpy as np

#from branching import MostFractionalBranching

x = np.array([
    1.0,
    0.3,
    0.52,
    0.9,
    0.48,
])

brancher = MostFractionalBranching()

print(brancher.select(x))