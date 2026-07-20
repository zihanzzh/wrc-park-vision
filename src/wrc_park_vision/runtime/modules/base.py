"""Task-module contract used by the runtime orchestrator."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import Observation, ValidatedImage


class TaskModule(ABC):
    module_id: str
    task_group: str

    @abstractmethod
    def load(self) -> None:
        """Load all module resources once at pipeline startup."""

    @abstractmethod
    def run(self, image: ValidatedImage) -> list[Observation]:
        """Return normalized observations without backend-specific objects."""

    def close(self) -> None:
        """Release module resources when supported."""

