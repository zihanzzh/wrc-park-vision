"""Reserved behavior-module contract; behavior inference is not implemented yet."""

from __future__ import annotations

from ..schemas import Observation, ValidatedImage
from .base import TaskModule


class BehaviorModule(TaskModule):
    def __init__(self, module_id: str, task_group: str) -> None:
        self.module_id = module_id
        self.task_group = task_group

    def load(self) -> None:
        raise NotImplementedError("behavior module is reserved but not implemented")

    def run(self, image: ValidatedImage) -> list[Observation]:
        raise NotImplementedError("behavior module is reserved but not implemented")

