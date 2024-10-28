from .solver import BaseSolver
from .dinod_solver import DinodSolver

from typing import Dict

TASKS : Dict[str, BaseSolver] = {
    'detection': DinodSolver,
                                 }
