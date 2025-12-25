# gurobipy.pyi
from typing import Any

class Model:
    def addVar(self, name: str) -> Any:
        ...

class Var:
    x: float