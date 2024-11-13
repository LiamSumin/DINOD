from .solver import BaseSolver
from .dinod_solver import DINODSolver
from typing import Dict

TASKS : Dict[str, BaseSolver] = {
    'detection' : DINODSolver,
}