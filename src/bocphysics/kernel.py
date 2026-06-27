"""Edge colouring and body SoA packing for the colour-batched XPBD kernel.

Description:
    greedy_edge_color groups constraints so that every constraint in a colour
    touches disjoint movable bodies; pack_bodies stacks the touched bodies'
    velocity, spin, and inverse mass / inertia into SoA Matrix blocks. Both are
    shared helpers consumed by the colour-batched XPBD kernel in xpbd_kernel.py.
"""

from typing import Dict, List, Tuple

from bocpy import Matrix


def greedy_edge_color(items: List[Tuple[int, int]]) -> Tuple[List[int], int]:
    """Greedily colour edges so two sharing an endpoint never share a colour.

    Description:
        Each item is an (endpoint_a, endpoint_b) pair; this is greedy colouring
        of the line graph in input order -- assign each item the smallest colour
        not already used by an item incident to either endpoint. Input order is
        deterministic, so the colouring is worker-count independent.
    """
    used: Dict[int, set] = {}
    colors = [0] * len(items)
    ncolors = 0
    for idx, (end_a, end_b) in enumerate(items):
        taken = used.get(end_a, set()) | used.get(end_b, set())
        color = 0
        while color in taken:
            color += 1

        colors[idx] = color
        used.setdefault(end_a, set()).add(color)
        used.setdefault(end_b, set()).add(color)
        ncolors = max(ncolors, color + 1)

    return colors, ncolors


def pack_bodies(bodies: list):
    """Stack per-body velocity, spin, and inverse mass / inertia into SoA blocks.

    Description:
        Static bodies get a zeroed velocity row to match their zero contribution,
        and their inverse mass / inertia are already zero so any scatter delta
        onto their row is exactly zero.
    """
    n = len(bodies)
    vel = Matrix(n, 2, [c for b in bodies for c in
                        ((b.linear_velocity.x, b.linear_velocity.y)
                         if b.physics else (0.0, 0.0))])
    spin = Matrix(n, 1, [b.angular_velocity if b.physics else 0.0
                         for b in bodies])
    inv_m = Matrix(n, 1, [b.inv_mass for b in bodies])
    inv_i = Matrix(n, 1, [b.inv_inertia for b in bodies])
    return vel, spin, inv_m, inv_i
