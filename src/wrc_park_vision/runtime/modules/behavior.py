"""Candidate generation and VLM-confirmed single-image behavior observations."""

from __future__ import annotations

from ..config import BehaviorSettings
from ..schemas import (
    BehaviorCandidate,
    BehaviorClassSummary,
    Observation,
    ObservationSource,
    ReviewSummary,
    ValidatedImage,
)
from .base import TaskModule


class BehaviorPipeline:
    """Keep behavior rules outside the main runtime orchestrator."""

    def __init__(self, settings: BehaviorSettings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def class_summaries(self) -> list[BehaviorClassSummary]:
        if not self.enabled:
            return []
        return [
            BehaviorClassSummary(
                class_id=item.class_id,
                class_name=item.class_name,
                required_object_classes=list(item.required_object_classes),
            )
            for item in self.settings.classes
        ]

    def generate_candidates(self, observations: list[Observation]) -> list[BehaviorCandidate]:
        if not self.enabled:
            return []
        by_class: dict[str, list[Observation]] = {}
        for observation in observations:
            if observation.kind != "detection":
                continue
            if observation.task_group != self.settings.object_task_group:
                continue
            by_class.setdefault(observation.class_name, []).append(observation)

        candidates: list[BehaviorCandidate] = []
        for behavior_class in self.settings.classes:
            required = behavior_class.required_object_classes
            if any(class_name not in by_class for class_name in required):
                continue
            evidence = sorted(
                (
                    observation
                    for class_name in required
                    for observation in by_class[class_name]
                ),
                key=lambda observation: observation.id,
            )
            candidates.append(
                BehaviorCandidate(
                    id=f"behavior-candidate-{len(candidates) + 1:04d}",
                    class_id=behavior_class.class_id,
                    class_name=behavior_class.class_name,
                    evidence_observation_ids=[observation.id for observation in evidence],
                    evidence_class_names=list(required),
                )
            )
        return candidates

    def build_observations(self, review: ReviewSummary) -> list[Observation]:
        if not self.enabled or review.status != "completed":
            return []
        class_by_name = {item.class_name: item for item in self.settings.classes}
        observations: list[Observation] = []
        seen_classes: set[str] = set()
        for decision in review.behaviors:
            if decision.verdict != "confirmed" or decision.class_name in seen_classes:
                continue
            behavior_class = class_by_name.get(decision.class_name)
            if behavior_class is None:
                continue
            seen_classes.add(decision.class_name)
            observations.append(
                Observation(
                    id=f"behavior-{len(observations) + 1:04d}",
                    kind="behavior",
                    task_group=self.settings.task_group,
                    class_id=behavior_class.class_id,
                    class_name=behavior_class.class_name,
                    confidence=decision.confidence if decision.confidence is not None else 0.0,
                    source=ObservationSource(
                        module_id=self.settings.module_id,
                        backend=review.provider or "vlm",
                        model_id=review.model_id or "unknown",
                    ),
                    geometry=None,
                    evidence_observation_ids=list(decision.evidence_observation_ids),
                    reasoning=decision.reasoning,
                    metadata={
                        "behavior_review_id": decision.id,
                        "candidate_id": decision.candidate_id,
                    },
                )
            )
        return observations


class BehaviorModule(TaskModule):
    """Reserved future image-level behavior model module."""

    def __init__(self, module_id: str, task_group: str) -> None:
        self.module_id = module_id
        self.task_group = task_group

    def load(self) -> None:
        raise NotImplementedError("behavior module is reserved but not implemented")

    def run(self, image: ValidatedImage) -> list[Observation]:
        raise NotImplementedError("behavior module is reserved but not implemented")
