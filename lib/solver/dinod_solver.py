import time
import json
import datetime

import time
from .solver import BaseSolver

class DINODSolver(BaseSolver):
    def fit(self, ):
        print("Start training")
        self.train()