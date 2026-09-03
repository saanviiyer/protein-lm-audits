"""crosstalk -- RL environments for protein binding specificity.

Binding specificity is a two-sided objective: a design must bind its target and
must not bind everything else. This package turns exhaustively measured
two-partner binding landscapes into budgeted decision problems, so that a
specificity objective can be optimized, and audited, under a realistic noisy
oracle.
"""
from .landscape import AA, Landscape, load_pard3
from .objectives import OBJECTIVES, Objective, is_specific, make
from .envs import BudgetEnv, WalkEnv, register_gymnasium
from .solve import best_design, optimal_walk_value, ruggedness
from .metrics import evaluate

__version__ = "0.1.0"
__all__ = [
    "AA", "Landscape", "load_pard3", "Objective", "OBJECTIVES", "make",
    "is_specific", "BudgetEnv", "WalkEnv", "register_gymnasium",
    "best_design", "optimal_walk_value", "ruggedness", "evaluate",
]
