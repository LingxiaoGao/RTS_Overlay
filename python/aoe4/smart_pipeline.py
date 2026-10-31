"""Smart HUD pipeline: cadence-gated OCR read, parse, confirm, and state ingest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from aoe4.smart_capture import CapturedFrame
from aoe4.smart_confirmation import HudEvidenceConfirmer
from aoe4.smart_hud import parse_hud_text
from aoe4.smart_hud_reader import HudTextEvidence, text_mapping
from aoe4.smart_runtime import GameEvent, GameState, Observation, StateEstimator
from aoe4.smart_sampling import ObservationScheduler


@dataclass(frozen=True)
class PipelineResult:
    sampled: bool
    evidence: Mapping[str, HudTextEvidence]
    observations: tuple[Observation, ...]
    verified_observations: tuple[Observation, ...]
    state: GameState
    events: tuple[GameEvent, ...]



def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _require_methods(component: object, name: str, methods: tuple[str, ...]) -> None:
    for method in methods:
        if not callable(getattr(component, method, None)):
            raise TypeError(f"{name} must expose a callable {method} method")


class SmartHudPipeline:
    """Orchestrate scheduler, reader, parser, confirmer, and estimator."""

    def __init__(
        self,
        reader: object,
        scheduler: ObservationScheduler | None = None,
        estimator: StateEstimator | None = None,
        confirmer: HudEvidenceConfirmer | None = None,
        profile_validated: bool = False,
        age_signal_validated: bool = False,
    ) -> None:
        read = getattr(reader, "read", None)
        if not callable(read):
            raise TypeError("reader must expose a callable read(frame)")
        self._reader = reader
        if scheduler is None:
            scheduler = ObservationScheduler()
        _require_methods(scheduler, "scheduler", ("due", "mark_sample", "reset"))
        if estimator is None:
            estimator = StateEstimator()
        _require_methods(estimator, "estimator", ("ingest", "mark_stale", "reset"))
        if not hasattr(estimator, "state"):
            raise TypeError("estimator must expose state")
        if confirmer is None:
            confirmer = HudEvidenceConfirmer()
        _require_methods(
            confirmer, "confirmer", ("checkpoint", "accept", "restore", "reset")
        )
        self._scheduler = scheduler
        self._estimator = estimator
        self._confirmer = confirmer
        self._profile_validated = _require_bool(
            profile_validated, "profile_validated"
        )
        self._age_signal_validated = _require_bool(
            age_signal_validated, "age_signal_validated"
        )
        self._sample_index = 0

    @property
    def scheduler(self) -> ObservationScheduler:
        return self._scheduler

    @property
    def estimator(self) -> StateEstimator:
        return self._estimator

    @property
    def confirmer(self) -> HudEvidenceConfirmer:
        return self._confirmer

    @property
    def profile_validated(self) -> bool:
        return self._profile_validated

    @property
    def age_signal_validated(self) -> bool:
        return self._age_signal_validated

    @property
    def state(self) -> GameState:
        return self._estimator.state

    def set_validation(
        self,
        profile_validated: bool,
        age_signal_validated: bool = False,
    ) -> None:
        profile = _require_bool(profile_validated, "profile_validated")
        age = _require_bool(age_signal_validated, "age_signal_validated")
        if profile != self._profile_validated or age != self._age_signal_validated:
            self._profile_validated = profile
            self._age_signal_validated = age
            self._confirmer.reset()

    def process(self, frame: CapturedFrame) -> PipelineResult:
        if not isinstance(frame, CapturedFrame):
            raise TypeError(
                f"frame must be a CapturedFrame, got {type(frame).__name__}"
            )
        now = frame.captured_at

        if not self._scheduler.due(now):
            state = self._estimator.mark_stale(now)
            return PipelineResult(
                sampled=False,
                evidence={},
                observations=(),
                verified_observations=(),
                state=state,
                events=(),
            )

        raw = self._reader.read(frame)
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"reader.read must return a Mapping, got {type(raw).__name__}"
            )
        evidence = raw

        texts = text_mapping(evidence)
        observations = parse_hud_text(texts, frame.frame_id, now)

        checkpoint = self._confirmer.checkpoint()
        try:
            verified = self._confirmer.accept(
                evidence,
                observations,
                self._sample_index,
                self._profile_validated,
                self._age_signal_validated,
            )
            state, events = self._estimator.ingest(verified, now)
            state = self._estimator.mark_stale(now)
            self._scheduler.mark_sample(now)
        except Exception:
            self._confirmer.restore(checkpoint)
            raise
        self._sample_index += 1

        return PipelineResult(
            sampled=True,
            evidence=evidence,
            observations=observations,
            verified_observations=verified,
            state=state,
            events=events,
        )

    def reset(self) -> GameState:
        self._scheduler.reset()
        self._confirmer.reset()
        self._sample_index = 0
        return self._estimator.reset()
