"""Deferred admission of runtime-spawned bodies.

Description:
    A click or a generator can drop a body straight on top of the pile. Feeding
    that deep initial overlap to the XPBD solver turns the one-shot de-penetration
    push into a huge velocity, and the body explodes outward. Instead a spawned
    body waits in this queue and only enters the world on a frame where it fits
    without significant overlap. A body that never fits is discarded after a
    budget of tries, so clicking on a solid pile quietly does nothing rather than
    detonating it. Scene bodies still go straight in through engine.add_body; only
    runtime spawns are routed through the queue.
"""
import logging
from typing import List

from .bodies import RigidBody
from .collisions import detect_collision

logger = logging.getLogger(__name__)

# A queued spawn that still clashes after this many frames is discarded.
MAX_SPAWN_TRIES = 60
# Penetration depth (world units) a candidate may have and still be admitted.
SPAWN_CLEARANCE = 0.05
# How many separation nudges to try when easing a spawn out of an overlap.
SPAWN_NUDGE_STEPS = 8


def spawn_overlaps(body: RigidBody, bodies: List[RigidBody], clearance: float) -> bool:
    """True when body penetrates any body in bodies by more than clearance."""
    for other in bodies:
        if other is body or body.aabb.disjoint(other.aabb):
            continue
        collision = detect_collision(body, other)
        if collision is not None and collision.depth > clearance:
            return True
    return False


def _deepest_overlap(body, bodies, clearance):
    """Return the deepest collision pushing body out of bodies, or None."""
    worst = None
    for other in bodies:
        if other is body or body.aabb.disjoint(other.aabb):
            continue
        collision = detect_collision(body, other)
        if collision is not None and collision.depth > clearance:
            if worst is None or collision.depth > worst.depth:
                worst = collision
    return worst


def fit_position(body, bodies, clearance, steps) -> bool:
    """Nudge body out of its deepest overlaps; return True if it ends up clear.

    Description:
        Each step pushes the body along the minimum-translation normal of its
        deepest overlap, the same direction the solver would separate it, so a
        body dropped onto a floor or pile is eased onto it instead of being
        rejected. detect_collision(body, other) orients its normal toward other,
        so the solver moves body along -normal; this mirrors that.
    """
    for _ in range(steps):
        collision = _deepest_overlap(body, bodies, clearance)
        if collision is None:
            return True
        body.move(collision.normal * -(collision.depth + clearance))
    return _deepest_overlap(body, bodies, clearance) is None


class SpawnQueue:
    """Holds runtime-spawned bodies until they can enter the world without overlap."""

    def __init__(self, max_tries: int = MAX_SPAWN_TRIES,
                 clearance: float = SPAWN_CLEARANCE,
                 nudge_steps: int = SPAWN_NUDGE_STEPS):
        """Configure the retry budget, admissible overlap depth, and nudge budget."""
        self.max_tries = max_tries
        self.clearance = clearance
        self.nudge_steps = nudge_steps
        self.pending: List[List] = []

    def enqueue(self, body: RigidBody):
        """Queue a body to wait for an overlap-free frame."""
        self.pending.append([body, 0])

    def process(self, bodies: List[RigidBody]) -> List[RigidBody]:
        """Admit queued bodies that now fit, age the rest, and return the admitted.

        Description:
            A candidate that clashes is first nudged along its separation normals
            to ease it onto whatever it landed on; the nudged pose is kept only if
            it ends up clear, otherwise the body is restored to its click position.
            Each admitted body joins the set the remaining candidates test against,
            so two spawns queued together are never admitted into each other. A
            body that still clashes after max_tries frames is dropped with a log
            line instead of waiting forever.
        """
        admitted: List[RigidBody] = []
        still: List[List] = []
        for body, tries in self.pending:
            others = bodies + admitted
            if not spawn_overlaps(body, others, self.clearance):
                admitted.append(body)
                continue
            origin = body.position.copy()
            if fit_position(body, others, self.clearance, self.nudge_steps):
                admitted.append(body)
                continue
            body.move_to(origin)
            if tries + 1 >= self.max_tries:
                logger.info("Discarded spawn that never found space: %r", body)
            else:
                still.append([body, tries + 1])
        self.pending = still
        return admitted
